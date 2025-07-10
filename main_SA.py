
import os
import surprise_mars_cyclegan as search
import pickle
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
import random 
import numpy as np
GaGan = search.GaGan()
GaGan.begin_server("D:/MarsEnv2/WindowsNoEditor/Mars.exe", 'PhysXCar')
# GA.initiate_pipe()
# GaGan.assignIDs()
def load_activation_pattern(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Flatten a 4D tensor (batch_size, channels, height, width) to 2D (channels, height*width)
def flatten_activation(activation):
    return activation.squeeze().view(activation.shape[0], -1).cpu().numpy()
# Load and preprocess the image with larger input size
def load_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Increase size to 512x512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Hook function to capture the activations from a specific layer
def get_activation_pattern(model, image, layer):
    activations = []

    def hook(module, input, output):
        activations.append(output)

    # Register the hook on the specific layer
    handle = layer.register_forward_hook(hook)

    # Forward pass through the model
    model(image)

    # Remove the hook after forward pass
    handle.remove()

    # Return the captured activation
    return activations[0].detach()
def load_train_activations():
        # Directory containing saved activation patterns
        activation_dir = "D:/pix2pixHD/datasets/train_activations/"

        # Load all training activation patterns from disk and flatten them
        train_activations = []
        subset_activations = select_subset(os.listdir(activation_dir), subset_ratio=0.05)

        for file_name in subset_activations:
            if file_name.endswith("_activation.pkl"):
                file_path = os.path.join(activation_dir, file_name)
                train_pattern = load_activation_pattern(file_path)
                flattened_train_pattern = flatten_activation(train_pattern)
                train_activations.append(flattened_train_pattern)

        # Stack training activations into a single 2D array for LSA computation
        return np.vstack(train_activations)

def fit_kde(train_activations):
        # Fit KDE on training activations for LSA computation
        return KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train_activations)
def select_subset(train_activations, subset_ratio=0.25):
    total = len(train_activations)
    subset_size = int(total * subset_ratio)  # Determine the size of the subset
    return random.sample(train_activations, subset_size)  # Randomly sample the subset
if not os.path.exists('road_positions.csv'):
    GaGan.collect_data_to_csv(10000, 0.1)
    
    
# Load activations once during initialization
train_activations = load_train_activations()
kde = fit_kde(train_activations)

for i in range(1, 11):    
    GaGan.searchAlgo(100, 12, i, kde)
# GaGan.searchAlgo(100, 12, 1, kde)