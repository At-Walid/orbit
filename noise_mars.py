import shutil
from ast import Break
import pymoo
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.algorithms.so_genetic_algorithm import comp_by_cv_and_fitness, FitnessSurvival
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
import setup_path
import airsim
import cv2
from turtle import ycor
import csv
import numpy as np
import os
import time
import tempfile
import subprocess
import param_car
import math
import random
from airsim.types import Pose

import pymoo.problems.multi
from pymoo.util.dominator import Dominator

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.algorithms.so_genetic_algorithm import GA as SGA
from pymoo.visualization.scatter import Scatter

import pix2pixHD
import os

import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
# using cpu
ctx = mx.gpu(0)

# ctx = mx.cpu()
import torch

import numpy as np
import scipy.misc as m
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as skm
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

from torchvision.utils import save_image
import torchvision
from gluoncv.data.transforms.presets.segmentation import test_transform
from mxnet import image as mx_img

from PIL import Image

import os
from collections import OrderedDict
from torch.autograd import Variable
from pix2pixHD.options.test_options import TestOptions
from pix2pixHD.data.data_loader import CreateDataLoader
from pix2pixHD.models.models import create_model
import pix2pixHD.util.util as util
from pix2pixHD.util.visualizer import Visualizer
from pix2pixHD.util import html
from pix2pixHD.data.base_dataset import BaseDataset, get_params, get_transform, normalize
import torch
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics.pairwise import euclidean_distances
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex
import pytorch_lightning as pl

# Adding Gaussian Noise function
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.08):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0., 1.)  # Ensure pixel values stay within [0, 1]
        return noisy_img

opt = TestOptions().parse(save=False)
opt.nThreads = 2  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.verbose = False
opt.label_nc = 0
opt.resize_or_crop = 'none'
opt.no_instance = True
opt.isTrain = False
opt.phase = 'test'
opt.which_epoch = 'latest'
opt.use_encoded_image = False
opt.output_nc = 3
opt.onnx = None
opt.ngf = 64
opt.nef = 16
opt.instance_feat = False
opt.input_nc = 3
opt.gpu_ids = [0]
opt.data_type = 32
opt.aspect_ratio = 1.0

visualizer = Visualizer(opt)
web_dir = os.path.join('D:/results/')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

opt.checkpoints_dir = './checkpoints/'
opt.name = 'ai4mars' 
def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-24)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding
def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop[a].get("rank"), b, pop[b].get("rank"),
                               method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)
class ImageSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes: int = 5, learning_rate: float = 1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Using DeepLabV3 with ResNet backbone
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        preds = torch.argmax(preds, dim=1)

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.accuracy(preds, y), on_step=True, on_epoch=True)
        self.log('val_iou', self.iou(preds, y), on_step=True, on_epoch=True)

class GaGan(object):
    pop = []
    algo = SGA
    params = {'image': ['x_val', 'y_val', 'z_val', 'Speed', 'Gear', 'Throttle', 'Brake', 'Steering']}

    def __init__(self):
        self.server = None
        self.vehicle_name = None
        self.client = airsim.CarClient()
        self.carControls = airsim.CarControls()
        self.airsim = airsim
        self.dir = 'D:/search_sets/mars/multi/noise2/'

    def begin_server(self, server, vehicle_name):
        self.server = server
        self.vehicle_name = vehicle_name
        print('begin')
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.simEnableWeather(True)
            self.client.reset()
        except:
            try:
                subprocess.Popen(self.server)
                time.sleep(10)
                self.client = airsim.CarClient()
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                self.client.simEnableWeather(True)
            except FileNotFoundError:
                print("Airsim not found: " + self.server)
            except Exception as e:
                print("Error occurred while starting Airsim: " + str(e))

        print("Airsim started successfully!")

    def assignIDs(self):
        print('assigning')
        obj_list = self.client.simListSceneObjects()
        labels = [1, 2, 3, 3, 3, 3, 3, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 9, 10, 11, 12, 14, 14, 15, 16, 17, 18, 19, 3, 6, 6,
                  14, 3, 2, 19, 19, 3, 1, 3, 3, 6, 6, 3, 8]

        class_map = ['road', 'sidewalk', 'building', 'house', 'wall', 'window', 'garage', 'fence', 'hedge', 'tree',
                     'birch', 'pavement', 'vehicle', 'car', 'suv', 'sedan', 'coupe', 'terrain', 'sky', 'person',
                     'rider', 'motorcycle', 'bike', 'autorickshaw', 'truck', 'bus', 'sign', 'skyscraper', 'grass',
                     'oak', 'bicycle', 'apartment', 'rock', 'trafficsign', 'light', 'house', 'leave', 'porch', 'roof',
                     'fir_', 'foliage', 'tower', 'saloon']
        assigned_objs = {}

        for obj_name in obj_list:
            for i in range(len(class_map)):
                obj_id = self.client.simGetSegmentationObjectID(obj_name.lower())
                if  class_map[i] in obj_name.lower():
                    print(obj_name, obj_id)
                    success = self.client.simSetSegmentationObjectID(obj_name, labels[i]);

        for obj_name in obj_list:
            if  'fir_' in obj_name.lower():
                obj_ID_before = self.client.simGetSegmentationObjectID(obj_name.lower())
                print(obj_name, obj_ID_before)
                success = self.client.simSetSegmentationObjectID(obj_name, 6);

        for obj_name in obj_list:
            if  'hedge' in obj_name.lower():
                obj_ID_before = self.client.simGetSegmentationObjectID(obj_name.lower())
                print(obj_name, obj_ID_before)
                success = self.client.simSetSegmentationObjectID(obj_name, 6);

        for obj_name in obj_list:
            if  'house' in obj_name.lower():
                obj_ID_before = self.client.simGetSegmentationObjectID(obj_name.lower())
                print(obj_name, obj_ID_before)
                success = self.client.simSetSegmentationObjectID(obj_name, 3);

        for obj_name in obj_list:
            if  'wall_low' in obj_name.lower():
                obj_ID_before = self.client.simGetSegmentationObjectID(obj_name.lower())
                print(obj_name, obj_ID_before)
                success = self.client.simSetSegmentationObjectID(obj_name, 5);
        pass

    def collect_data_to_csv(self, total=10000, freq=0.1):
        road_pos = []
        self.client.confirmConnection()
        for i in range(total):
            pos = self.client.simGetVehiclePose(self.vehicle_name)
            info = self.client.simGetGroundTruthEnvironment()
            pos_dict = {
                'x_pos': pos.position.x_val,
                'y_pos': pos.position.y_val,
                'z_pos': pos.position.z_val,
                'w_ori': pos.orientation.w_val,
                'x_ori': pos.orientation.x_val,
                'y_ori': pos.orientation.y_val,
                'z_ori': pos.orientation.z_val,
                'p_r_y': airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
            }
            road_pos.append(pos_dict)
            time.sleep(freq)
        with open('road_positions.csv', 'w', newline='') as csvfile:
            fieldnames = ['x_pos', 'y_pos', 'z_pos', 'w_ori', 'x_ori', 'y_ori', 'z_ori', 'p_r_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for pos in road_pos:
                writer.writerow(pos)
        print('collected' + str(len(road_pos)) + 'points')

    def control_car_from_csv(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        with open('road_positions.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [row for row in reader]
        random.shuffle(rows)
        for row in rows:
            pos_dict = {'x_pos': float(row['x_pos']), 'y_pos': float(row['y_pos']),
                    'z_pos': float(row['z_pos']),
                    'w_ori': float(row['w_ori']),
                    'x_ori': float(row['x_ori']),
                    'y_ori': float(row['y_ori']),
                    'z_ori': float(row['z_ori']),
                    'p_r_y': list(row['p_r_y'])}
            position = airsim.Vector3r(x_val=pos_dict['x_pos'], y_val=pos_dict['y_pos'], z_val=pos_dict['z_pos'])
            orientation = airsim.Quaternionr(w_val=pos_dict['w_ori'], x_val=pos_dict['x_ori'], y_val=pos_dict['y_ori'],
                                             z_val=pos_dict['z_ori'])
            pos = Pose(position_val=position, orientation_val=orientation)
            self.client.simSetVehiclePose(pos, True, self.vehicle_name)
            time.sleep(1)
    
    
    def retrieveImages(self, x, numSet):
        self.dir = "D:/search_sets/mars/multi/noise2/set" + str(numSet)
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),  # label
            airsim.ImageRequest("1", airsim.ImageType.Scene),  # scene vision image in png format #img
        ])
        files = []
        label = ['L', 'S']
        for response_idx, response in enumerate(responses):
            filename = os.path.join(self.dir, label[response_idx] + '_' + str(x[0]) + "_" + str(x[1]))
            if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress:  # png format
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else:  # uncompressed array
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 3 channels
                
                if response_idx == 1:  # Convert only the scene image to grayscale
                    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # convert to grayscale
                    cv2.imwrite(os.path.normpath(filename + '.png'), img_gray)  # write grayscale image to png
                else:
                    cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb)  # write original RGB image to png
                
            files.append(os.path.normpath(filename + '.png'))
        return files

    def segmentImages(self, ID):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
                                              airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        segment = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height,
                                                                                       responses[0].width, 3)
        airsim.write_png(os.path.join(self.out_path, 'segment_' + str(ID) + '.png'), segment)
        scene = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height,
                                                                                     responses[0].width, 3)
        airsim.write_png(os.path.join(self.out_path, 'scene_' + str(ID) + '.png'), scene)
        return scene, segment
        pass

    def reset(self):
        car_controls = self.airsim.CarControls()
        car_controls.throttle = 0
        car_controls.steering = 0
        car_controls.speed = 0
        self.client.setCarControls(car_controls)
        self.client.reset()

    def processImages(self):
        pass


    def searchAlgo(self, n_gen, pop_size, numSet):
        path = "D:/search_sets/mars/multi/noise2/set" + str(numSet)
        os.makedirs(path, exist_ok=True)
        with open('road_positions.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [row for row in reader]
        GAN = getGAN()
        DNN = gluoncv.model_zoo.get_deeplab(dataset='citys', backbone='resnet50', pretrained=True, ctx=mx.gpu(0))
        MyProblem = GanProblem(n_gen, pop_size, rows, self, DNN, GAN, numSet)
        genetic_algo = NSGA2(pop_size=pop_size, sampling=FloatRandomSampling(),
                selection=TournamentSelection(func_comp=binary_tournament),
                crossover=SimulatedBinaryCrossover(eta=15, prob=0.7),
                mutation=PolynomialMutation(prob=0.3, eta=20))
        print(n_gen, pop_size)
        minimize(MyProblem, genetic_algo, ('n_gen', n_gen), seed=1, verbose=True)
        print(len(MyProblem.forbidden))
        with open('ga_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['img', 'individual', 'entry', 'rot', 'F', 'PixAcc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for pos in MyProblem.archive:
                writer.writerow(pos)

def compute_color_proportion(image_path, target_color):
    img = Image.open(image_path)
    img_np = np.array(img)
    target_color_np = np.array(target_color, dtype=np.uint8)
    flattened_img = img_np.reshape((-1, 3))
    num_target_color_pixels = np.sum(np.all(flattened_img == target_color_np, axis=1))
    total_pixels = img_np.shape[0] * img_np.shape[1]
    proportion = num_target_color_pixels / total_pixels
    return proportion

def compute_all_color_proportion(image_path):
    img = Image.open(image_path)
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_np = np.stack((img_np,) * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    flattened_img = img_np.reshape((-1, 3))
    unique_colors, counts = np.unique(flattened_img, axis=0, return_counts=True)
    total_pixels = img_np.shape[0] * img_np.shape[1]
    proportions = counts / total_pixels
    for color, proportion in zip(unique_colors, proportions):
        if proportion > 0.70:
            return True, color.tolist(), proportion
    return False, None, None
def distance(input_image_path, archive, type_distance):
    if(type_distance == "PixelAccuracy"):
        closest_image, closest_distance = find_highest_pixel_accuracy(input_image_path, archive)
        closest_index = archive.index(closest_image)
    else:
        pairwise_distances = euclidean_distances(archive, [input_image_path])
        closest_index = np.argmin(pairwise_distances)
        # closest_image_path = archive2[closest_index]
        closest_distance = pairwise_distances[closest_index][0]
    return closest_distance, closest_index
def preprocess_image(image_path: str, return_tensor: bool = False):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))
    image_normalized = np.asarray(image, dtype=np.float32) / 255.0
    if return_tensor:
        image_normalized = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)
        return image, image_tensor
    return image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the saved model
model = ImageSegmentationModel()
model.load_state_dict(torch.load("D:/ai4mars/retraining_results/variants_retraining2/noise/retrain_0_deeplab/retrained_model_checkpoint.pth"))
model.eval()
model.to(device)

def predict_img(f):
    test_image, test_image_tensor = preprocess_image(f, return_tensor=True)
    test_image_tensor = test_image_tensor.to(device)
    with torch.no_grad():
        prediction = model(test_image_tensor)
        predicted_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
    return predicted_mask

def is_car_stationary(self):
    car_state = self.client.getCarState(self.vehicle_name)
    car_velocity = car_state.kinematics_estimated.linear_velocity
    velocity_threshold = 0.1
    is_stationary = (abs(car_velocity.x_val) < velocity_threshold and
                     abs(car_velocity.y_val) < velocity_threshold and
                     abs(car_velocity.z_val) < velocity_threshold)
    return is_stationary

def compute_iou(mask1, mask2, num_classes):
    ious = []
    for cls in range(num_classes):
        intersection = np.logical_and(mask1 == cls, mask2 == cls).sum()
        union = np.logical_or(mask1 == cls, mask2 == cls).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

def compute_miou(ious):
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    if len(valid_ious) == 0:
        return float('nan')
    else:
        return np.mean(valid_ious)

def generate_dnn_mask(image_path):
    color_map_DNN = {
        (24, 175, 120): 3,    # big rock, red
        (146, 52, 70): 1,     # bedrock, grey
        (188, 18, 5): 2,      # sand, yellow
        (187, 70, 156): 0,    # soil, beige
        (249, 79, 73): 4      # null, black
    }
    image = Image.open(image_path).convert('RGB')
    rgb_array = np.array(image)
    dnn_mask = np.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            rgb_value = tuple(rgb_array[i, j])
            if rgb_value in color_map_DNN:
                dnn_mask[i, j] = color_map_DNN[rgb_value]
    return dnn_mask
def load_and_preprocess_image(image_path):
    # Define image transformations
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img)
    return img

# Function to extract features from an image
def extract_features(model_resnet, image):
    with torch.no_grad():
        image = image.unsqueeze(0)
        feature = model_resnet(image)
        feature = feature.squeeze().detach().cpu().numpy()
    return feature

# Function to find the closest image to the input image
def find_closest_image(input_image_path, directory):
    # Load and preprocess input image
    input_image = load_and_preprocess_image(input_image_path)
    input_feature = extract_features(model_resnet, input_image)

    # Load and preprocess all other images in the directory
    images = []
    image_paths = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        if img_path != input_image_path:  # Ignore the input image itself
            img = load_and_preprocess_image(img_path)
            images.append(img)
            image_paths.append(img_path)

    # Extract features for all images
    features = [extract_features(model_resnet, img) for img in images]

    # Calculate pairwise distances
    features = np.array(features)
    pairwise_distances = euclidean_distances(features, [input_feature])

    # Find the index of the closest image
    closest_index = np.argmin(pairwise_distances)
    closest_image_path = image_paths[closest_index]
    closest_distance = pairwise_distances[closest_index][0]

    return closest_image_path, closest_distance, image_paths
def generate_gan_mask(dnn_mask):
    color_map_GAN = {
        3: (254, 0, 0),       # big rock, red
        1: (128, 128, 128),   # bedrock, grey
        2: (255, 255, 0),     # sand, yellow
        0: (255, 240, 220),   # soil, beige
        4: (0, 0, 0)          # null, black
    }
    gan_mask = np.zeros((dnn_mask.shape[0], dnn_mask.shape[1], 3), dtype=np.uint8)
    for i in range(dnn_mask.shape[0]):
        for j in range(dnn_mask.shape[1]):
            label_value = dnn_mask[i, j]
            if label_value in color_map_GAN:
                gan_mask[i, j] = color_map_GAN[label_value]
    return gan_mask

def getGAN():
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
        model.opt = opt
    else:
        model = None
    print('Model loaded')
    return model

class GanProblem(pymoo.problems.multi.Problem):
    def __init__(self, n_gen, pop_size, rows, myGaGan, DNN, GAN, numSet, **kwargs):
        self.problemDict = []
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.t = time.time()
        self.forbidden = []
        self.counter = 0
        self.rows = rows
        self.GaGan = myGaGan
        self.DNN = DNN
        self.GAN = GAN
        self.archive = []
        self.numSet = numSet

        super().__init__(n_var=2, n_obj=1, n_constr=0, xl=[0, 0], xu=[1, 1], elementwise_evaluation=False, **kwargs, type_var=np.float64)

    def _evaluate(self, X, out, *args, **kwargs):
        path = "D:/search_sets/mars/multi/noise2/set" + str(self.numSet) + "/"
        print('@@@@@', path)
        F = []
        import psutil
        
        camera_position = airsim.Vector3r(0, 0, -1)  # x, y, z coordinates above the car
        camera_orientation = airsim.to_quaternion(-0.5, 0, 0)  # roll, pitch (downward), yaw in radians
        self.GaGan.client.simSetCameraPose("0", airsim.Pose(camera_position, camera_orientation))
        self.GaGan.client.simSetCameraPose("1", airsim.Pose(camera_position, camera_orientation))
        
        # Diversity fitness setup
        archive2 = []
        archive_features = []
        model_resnet = resnet18(pretrained=True)
        model_resnet.eval()
        type_distance = 'FeatureDistance'
        threshold_diversity = 12
        
        for x in X:
            t1 = time.time()
            row = self.rows[int(x[0] * len(self.rows))]
            pos_dict = {'x_pos': float(row['x_pos']), 'y_pos': float(row['y_pos']), 'z_pos': float(row['z_pos']),
                        'w_ori': float(row['w_ori']), 'x_ori': float(row['x_ori']), 'y_ori': float(row['y_ori']),
                        'z_ori': float(row['z_ori']), 'p_r_y': list(row['p_r_y'])}
            position = airsim.Vector3r(x_val=pos_dict['x_pos'], y_val=pos_dict['y_pos'], z_val=pos_dict['z_pos'])
            orientation = airsim.Quaternionr(w_val=x[1], x_val=pos_dict['x_ori'], y_val=pos_dict['y_ori'],
                                             z_val=pos_dict['z_ori'])
            pos = Pose(position_val=position, orientation_val=orientation)
            print(self.GaGan.vehicle_name)            
            self.GaGan.client.simSetVehiclePose(pos, True, self.GaGan.vehicle_name)

            start_time = time.time()
            while time.time() - start_time > 10 and not is_car_stationary(self.GaGan):
                print("Waiting for the car to settle...")
                time.sleep(1)  # Check every second

            if not is_car_stationary(self.GaGan):
                self.GaGan.client.setCarControls(airsim.CarControls(brake=1.0))
                print("Applying brakes...")
                time.sleep(2)
                self.GaGan.client.setCarControls(airsim.CarControls(brake=0.0))

            f = 1
            f2 = 1 
            if True:
                save_path = os.path.join(path, "R_" + str(x[0]) + "_" + str(x[1]) + ".png")
                simulated_path = os.path.join(path, "S_" + str(x[0]) + "_" + str(x[1]) + ".png")
                label_path = os.path.join(path, "L_" + str(x[0]) + "_" + str(x[1]) + ".png")
                label_GAN, label_DNN = None, None

                if not os.path.isfile(save_path):
                    t1 = time.time()
                    img_path = self.GaGan.retrieveImages(x, self.numSet)
                    sim_img = cv2.imread(simulated_path)
                    img_gray = cv2.cvtColor(sim_img, cv2.COLOR_RGB2GRAY)
                    cv2.imwrite(simulated_path, img_gray)
                    t4 = time.time()
                    label_DNN = generate_dnn_mask(label_path)
                    label_GAN = generate_gan_mask(label_DNN)
                    print("Label:", time.time()-t4)

                    img = Image.fromarray((label_GAN * 1).astype(np.uint8))
                    img.save(os.path.join(path, "G_" + str(x[0]) + "_" + str(x[1]) + ".png"))

                    t6 = time.time()
                    transform_A = get_transform(opt, get_params(opt, img.size))
                    img_A = transform_A(img.convert('RGB')).unsqueeze(0)
                    generated = self.GAN.inference(img_A, torch.tensor([0]), None)
                    util.save_image(util.tensor2im(generated.data[0]), save_path)
                    print("GAN:", time.time()-t6)

                t3 = time.time()
                # Preprocess image into tensor
                test_image, test_image_tensor = preprocess_image(save_path, return_tensor=True)

                pred = predict_img(save_path)

                # Apply light noise instead of flipping
                noise_transform = AddGaussianNoise()
                noisy_image_tensor = noise_transform(test_image_tensor)

                # Save noisy image
                noisy_image_path = os.path.join(path, "noisy_R_" + str(x[0]) + "_" + str(x[1]) + ".png")
                noisy_image = noisy_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
                cv2.imwrite(noisy_image_path, noisy_image)

                noisy_pred = predict_img(noisy_image_path)

                # Calculate mIoU between the noisy prediction and the original prediction
                miou_noisy_comparison = compute_miou(compute_iou(noisy_pred, pred, 5))
                print(f"mIoU between noisy prediction and non-noisy prediction: {miou_noisy_comparison}")

                # Calculate mIoU between the noisy prediction and the ground truth
                miou_gt_comparison = compute_miou(compute_iou(noisy_pred, label_DNN, 5))
                print(f"mIoU between noisy prediction and ground truth: {miou_gt_comparison}")

                f = miou_noisy_comparison

                f2 = 1  # Default diversity score
                proportion = compute_all_color_proportion(label_path)

                if(proportion[0] == True):
                    f = 2
                    f2 = 2
                    print('proportion too high/low', proportion[1], proportion[2])
               
            
                if len(archive2) == 0:
                    archive2.append(save_path)
                    img = load_and_preprocess_image(save_path)
                    features = extract_features(model_resnet, img)
                    archive_features.append(features)
                elif(f <):
                    img = load_and_preprocess_image(save_path)
                    features = extract_features(model_resnet, img)
                    closest_distance, _ = distance(features, archive_features, type_distance)
                    
                    if closest_distance > threshold_diversity:
                        archive_features.append(features)
                        archive2.append(save_path)
                        f2 = 1 / (1 + closest_distance)
                    else:
                        f2 = 2
                
                print("total:", time.time()-t1)
                F.append([f, f2])
                self.problemDict.append({'entry': x[0], 'rot': x[1], 'F': [f, f2]})
                
                with open(path + 'archive.txt', 'w') as fp:
                    for item in archive2:
                        fp.write("%s\n" % item)

            # print("total:", time.time()-t1)
            # F.append([f, f2])
            # self.problemDict.append({'entry': x[0], 'rot': x[1], 'F': [f, f2]})
            # with open(path + 'archive.txt', 'w') as fp:
            #     for item in self.archive:
            #         fp.write("%s\n" % item)
         
            
        
        if out is None:
            return F
        else:
            out["F"] = np.array(F)


