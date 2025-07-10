import os
import shutil
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex, MeanMetric
from typing import List, Tuple
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Ensure the 'original_retraining' directory exists
os.makedirs('variants_retraining/noise_cyclegan/', exist_ok=True)

class AI4MARSDataset(Dataset):
    def __init__(self, images_path: str, masks_path: str, additional_set: str = None, dataset_size: int = 500):
        self.images_path = images_path
        self.masks_path = masks_path
        self.additional_set_path = additional_set
        self.dataset_size = dataset_size
        
        images = set(os.listdir(images_path))
        self.masks = [mask for mask in os.listdir(masks_path) if mask[:-4] + ".JPG" in images][:dataset_size]

        # Include additional set
        if additional_set:
            set_images_path = os.path.join(additional_set, 'realistic')
            set_masks_path = os.path.join(additional_set, 'labels_DNN')
            set_images = set(os.listdir(set_images_path))
            self.masks += [mask for mask in os.listdir(set_masks_path) if mask in set_images]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
        ])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_name = self.masks[idx]
        if os.path.exists(os.path.join(self.images_path, mask_name[:-4] + ".JPG")):
            image_path = os.path.join(self.images_path, mask_name[:-4] + ".JPG")
            mask_path = os.path.join(self.masks_path, mask_name)
        else:
            additional_set_path = self.additional_set_path
            image_path = os.path.join(additional_set_path, 'realistic', mask_name)
            mask_path = os.path.join(additional_set_path, 'labels_DNN', mask_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: unable to read image {image_path}. Skipping this sample.")
            return self.__getitem__((idx + 1) % len(self))  # Skip this image and get the next one

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"Warning: unable to read mask {mask_path}. Skipping this sample.")
            return self.__getitem__((idx + 1) % len(self))  # Skip this mask and get the next one

        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask, dtype=np.uint8)
        mask[mask == 255] = 4
        mask = torch.from_numpy(mask)
        mask = mask.long()

        return image, mask

class AI4MARSDataModule(pl.LightningDataModule):
    def __init__(self, images_path: str, masks_path: str, additional_set: str = None, dataset_size: int = 5000, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.additional_set = additional_set
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = AI4MARSDataset(self.images_path, self.masks_path, self.additional_set, self.dataset_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

class ImageSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes: int = 5, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)
        self.miou = MeanMetric()

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        preds = torch.argmax(preds, dim=1)
        
        # Compute IoU
        iou = self.iou(preds, y)
        self.miou(iou)
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(preds, y))
        self.log('train_iou', self.iou(preds, y))
        self.log('train_miou', self.miou.compute())
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        preds = torch.argmax(preds, dim=1)
        
        # Compute IoU
        iou = self.iou(preds, y)
        self.miou(iou)

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.accuracy(preds, y), on_step=True, on_epoch=True)
        self.log('val_iou', self.iou(preds, y), on_step=True, on_epoch=True)
        self.log('val_miou', self.miou.compute(), on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        # Reset the metric after each validation epoch
        self.miou.reset()

def train_single_run(data_module: AI4MARSDataModule, checkpoint_path: str, retrain_index: int, epochs: int = 100):
    data_module.setup()

    # Split the dataset into training and validation sets
    dataset_size = len(data_module.dataset)
    print(f"Total dataset size: {dataset_size}")
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    print(f"Training set size: {train_size}, Validation set size: {val_size}")
    train_dataset, val_dataset = random_split(data_module.dataset, [train_size, val_size])

    # Set up data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=data_module.batch_size, shuffle=True, num_workers=data_module.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=data_module.batch_size, num_workers=data_module.num_workers)

    # Initialize the Lightning model
    model = ImageSegmentationModel()

    # Check if checkpoint exists before loading
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch.")
    else:
        print("Checkpoint does not exist. Training from scratch.")

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/retrain_{retrain_index}_deeplab",
        filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    # Set up the Lightning trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50
    )

    # Train and validate the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the model checkpoint after training
    save_path = f"variants_retraining/noise_cyclegan/retrain_{retrain_index}_deeplab/retrained_model_checkpoint.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


# Paths
IMAGES_PATH = "./data/msl/images/edr"
MASK_PATH_TRAIN = "./data/msl/labels/train"
CHECKPOINT_PATH = "./retrained_model_deeplab2.pth"
ADDITIONAL_SETS_PATH = "./retraining_sets/noise_cyclegan/"

# Hyperparameters
DATASET_SIZE = 17000
BATCH_SIZE = 32
EPOCHS = 20

# Perform 10 retraining sessions with different additional data sets
for i in range(11):
    if not os.path.exists(f"./variants_retraining/noise_cyclegan/retrain_{i}_deeplab/retrained_model_checkpoint.pth"):
        additional_set = os.path.join(ADDITIONAL_SETS_PATH, f"set{i+1}")
        print(f"Retraining session {i+1} with additional set: {additional_set}")
        data_module = AI4MARSDataModule(IMAGES_PATH, MASK_PATH_TRAIN, additional_set=additional_set, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE, num_workers=2)
        train_single_run(data_module, checkpoint_path=CHECKPOINT_PATH, retrain_index=i, epochs=EPOCHS)
