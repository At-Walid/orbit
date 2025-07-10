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
from torchvision import models, transforms
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


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import pickle
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
import random 

import torch
torch.cuda.empty_cache()


# print(os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'])
# os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
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
#data_loader = CreateDataLoader(opt)
#dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join('D:/results/')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


# Path to the pre-trained weights
opt.checkpoints_dir = './checkpoints/'  # directory where your .pth files are stored
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

        # Load the DeepLabV3 ResNet50 model and adjust for the number of classes
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x, mc_dropout=False, num_samples=10):
        if mc_dropout:
            # Enable dropout but keep batch norm in eval mode
            enable_mc_dropout(self.model)
            predictions = []
            for _ in range(num_samples):
                preds = self.model(x)['out']
                predictions.append(preds.unsqueeze(0))
            return torch.cat(predictions, dim=0)
        else:
            return self.model(x)['out']
class GaGan(object):
    pop = []
    algo = NSGA2
    params = {'image': ['x_val', 'y_val', 'z_val', 'Speed', 'Gear', 'Throttle', 'Brake', 'Steering']}

    def __init__(self):
        self.server = None
        self.vehicle_name = None
        self.client = airsim.CarClient()
        self.carControls = airsim.CarControls()
        self.airsim = airsim
        self.dir = 'D:/search_sets/mars/uncertainty/'

    def begin_server(self, server, vehicle_name):
        self.server = server
        self.vehicle_name = vehicle_name
        print('begin')
        # print(self.client.confirmConnection())
        # print(self.client.ping())
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
                self.assignIDs()
            except FileNotFoundError:
                print("Airsim not found: " + self.server)
            except Exception as e:
                print("Error occurred while starting Airsim: " + str(e))

        print("Airsim started successfully!")
    

    def collect_data_to_csv(self, total=10000, freq=0.1):
        road_pos = []
        self.client.confirmConnection()
        # Collect 10,000 data points
        for i in range(total):
            pos = self.client.simGetVehiclePose(self.vehicle_name)
            info = self.client.simGetGroundTruthEnvironment()
            # Convert Pose data into dictionary form for easier CSV writing
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
        # Write data to CSV file
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
        self.client.simEnableWeather(True)
        # Read data from CSV file
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
            # Assuming you have a method to convert pos_dict to Pose object
            position = airsim.Vector3r(x_val=pos_dict['x_pos'], y_val=pos_dict['y_pos'], z_val=pos_dict['z_pos'])
            orientation = airsim.Quaternionr(w_val=pos_dict['w_ori'], x_val=pos_dict['x_ori'], y_val=pos_dict['y_ori'],
                                             z_val=pos_dict['z_ori'])
            pos = Pose(position_val=position, orientation_val=orientation)
            self.client.simSetVehiclePose(pos, True, self.vehicle_name)
            time.sleep(1)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, random.randint(0, 1))
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, random.randint(0, 1))

    def retrieveImages(self, x, numSet):
        self.dir = "D:/search_sets/mars/uncertainty/set" + str(numSet)
        responses = self.client.simGetImages([

            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),  # label
            #airsim.ImageRequest("0", airsim.ImageType.DepthVis),  # depth visualization image #inst
            #airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
            airsim.ImageRequest("1", airsim.ImageType.Scene),  # scene vision image in png format #img
            #airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  # vision image in uncompressed RGB array
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
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 3 channels
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb)  # write to png
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
        path = "D:/search_sets/mars/uncertainty/set" + str(numSet)
        os.makedirs(path, exist_ok=True)
        with open('road_positions.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [row for row in reader]
        GAN = getGAN()
        DNN = gluoncv.model_zoo.get_deeplab(dataset='citys', backbone='resnet50', pretrained=True, ctx=mx.gpu(0))

        MyProblem = GanProblem(n_gen, pop_size, rows, self, DNN, GAN, numSet)
        print(MyProblem)
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
                # shutil.copy(pos['img'], 'D:/results/archive/'+os.path.basename(pos["img"]))


def enable_mc_dropout(model):
    """Enable dropout layers during test time without affecting batch normalization layers."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Keep dropout active
        elif isinstance(module, nn.BatchNorm2d):
            module.eval()   # Ensure BatchNorm stays in eval mode
def compute_pixelwise_variance(predictions):
    variance = np.var(predictions, axis=0)
    return variance


def compute_overall_uncertainty(variance_map, method='median'):
    if method == 'median':
        return np.median(variance_map)
    elif method == 'mean':
        return np.mean(variance_map)
def mc_dropout_inference(model, image, num_samples=5, device='cpu'):
    predictions = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure the model is on the correct device
    model.to(device)
    
    with torch.no_grad():
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Check if the image is already a tensor
        if isinstance(image, torch.Tensor):
            input_tensor = image.unsqueeze(0)  # Add batch dimension if needed
        else:
            input_tensor = preprocess(image).unsqueeze(0)  # Apply preprocessing

        # Move input_tensor to the same device as the model
        input_tensor = input_tensor.to(device)
        
        for _ in range(num_samples):
            output = model(input_tensor, mc_dropout=True)  # Forward pass with MC Dropout
            predictions.append(output.cpu().numpy())  # Move output back to CPU for processing

    # Stack predictions along the first dimension
    predictions = np.stack(predictions, axis=0)  # Shape: (num_samples, batch_size, num_classes, H, W)
    
    return predictions

def calculate_pixel_accuracy(target_mask, compared_mask):
    """Calculate the pixel accuracy between two segmentation masks."""
    assert target_mask.shape == compared_mask.shape, "Masks must have the same dimensions"
    correct_predictions = np.sum(np.all(target_mask == compared_mask, axis=-1))
    total_pixels = target_mask.shape[0] * target_mask.shape[1]
    pixel_accuracy = correct_predictions / total_pixels
    return pixel_accuracy

def find_highest_pixel_accuracy(target_mask_path, archive):
    """Find the mask with the highest pixel accuracy compared to the target mask and print the result."""
    target_mask = np.array(Image.open(target_mask_path))
    
    highest_accuracy = 0
    highest_accuracy_filename = ""
    
    for filename in archive:
        current_mask_path = filename
        
        if current_mask_path == target_mask_path or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        compared_mask = np.array(Image.open(current_mask_path))
        
        try:
            pixel_accuracy = calculate_pixel_accuracy(target_mask, compared_mask)
            if pixel_accuracy > highest_accuracy:
                highest_accuracy = pixel_accuracy
                highest_accuracy_filename = filename
        except AssertionError:
            print(f"Skipped due to size mismatch: {filename}")

    return highest_accuracy_filename, highest_accuracy
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
def compute_color_proportion(image_path, target_color):
    # Read the image
    img = cv2.imread(image_path)

    # Convert target color to numpy array (BGR format)
    target_color_np = np.array(target_color, dtype=np.uint8)

    # Flatten the image to a 1D array
    flattened_img = img.reshape((-1, 3))

    # Count occurrences of the target color
    num_target_color_pixels = np.sum(np.all(flattened_img == target_color_np, axis=1))

    # Compute the proportion of target color pixels
    total_pixels = img.shape[0] * img.shape[1]  # Assuming 2D image
    proportion = num_target_color_pixels / total_pixels

    return proportion
# Function to load and preprocess an image
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
def generate_masks(rgb_array):
    # Define the color maps
    color_map_GAN = {
        3: (255, 0, 0),       # big rock, red
        1: (128, 128, 128),    # bedrock, grey
        2: (255, 255, 0),       # sand, yellow
        0: (255, 240, 220),   # soil, beige
        4: (0, 0, 0)           # null, black
    }

    color_map_DNN = {
        (24, 175, 120): 3,                 # big rock, red
        (146, 52, 70): 1,                  # bedrock, grey
        (188, 18, 5): 2,                   # sand, yellow
        (187, 70, 156): 0,                 # soil, beige
        (249, 79, 73): 4                   # null, black
    }
    gan_mask = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 3), dtype=np.uint8)
    dnn_mask = np.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8)
    
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            rgb_value = tuple(rgb_array[i, j])
            if rgb_value in color_map_GAN:
                gan_mask[i, j] = color_map_GAN[rgb_value]
            if rgb_value in color_map_DNN:
                dnn_mask[i, j] = color_map_DNN[rgb_value]
    
    return gan_mask, dnn_mask

def compute_color_proportion(image_path, target_color):
    # Read the image
    img = cv2.imread(image_path)

    # Convert target color to numpy array (BGR format)
    target_color_np = np.array(target_color, dtype=np.uint8)

    # Flatten the image to a 1D array
    flattened_img = img.reshape((-1, 3))

    # Count occurrences of the target color
    num_target_color_pixels = np.sum(np.all(flattened_img == target_color_np, axis=1))

    # Compute the proportion of target color pixels
    total_pixels = img.shape[0] * img.shape[1]  # Assuming 2D image
    proportion = num_target_color_pixels / total_pixels

    return proportion
def compute_all_color_proportion(image_path):
    # Load the image with PIL
    img = Image.open(image_path)

    # Convert the image to a numpy array
    img_np = np.array(img)

    # Ensure the image has 3 channels (RGB)
    if len(img_np.shape) == 2:  # Grayscale image
        img_np = np.stack((img_np,) * 3, axis=-1)
    elif img_np.shape[2] == 4:  # RGBA image, convert to RGB
        img_np = img_np[:, :, :3]

    # Flatten the image to a 1D array
    flattened_img = img_np.reshape((-1, 3))

    # Get unique colors and their counts
    unique_colors, counts = np.unique(flattened_img, axis=0, return_counts=True)

    # Total number of pixels in the image
    total_pixels = img_np.shape[0] * img_np.shape[1]  # Assuming 2D image

    # Compute the proportions for each unique color
    proportions = counts / total_pixels

    # Check if any color has a proportion higher than 70%
    for color, proportion in zip(unique_colors, proportions):
        if proportion > 0.70:
            return True, color.tolist(), proportion

    return False, None, None
# Preprocess images
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
model.load_state_dict(torch.load("D:/ai4mars/retraining_results/variants_retraining2/search2_feat/retrain_5_deeplab/retrained_model_checkpoint.pth"))
model.eval()
model.to(device)
def predict_img(f):
        test_image, test_image_tensor = preprocess_image(f, return_tensor=True)
        test_image_tensor = test_image_tensor.to(device)

        # Perform prediction
        with torch.no_grad():
            prediction = model(test_image_tensor)
            predicted_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

        # Load the ground truth segmentation
        

        return predicted_mask
def is_car_stationary(self):
        # Get the car's velocity
        car_state = self.client.getCarState(self.vehicle_name)
        car_velocity = car_state.kinematics_estimated.linear_velocity

        # Check if the car's velocity is near zero
        velocity_threshold = 0.1  # Adjust this threshold based on your scenario
        is_stationary = (abs(car_velocity.x_val) < velocity_threshold and
                         abs(car_velocity.y_val) < velocity_threshold and
                         abs(car_velocity.z_val) < velocity_threshold)

        return is_stationary
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
# Compute LSA (Likelihood-based Surprise Adequacy)
def compute_lsa(kde, test_activation):
    # KDE expects a 2D array, so ensure test_activation is flattened
    log_density = kde.score_samples(test_activation)
    return -log_density.mean()  # Use the mean log-likelihood as LSA score

# Compute DSA (Distance-based Surprise Adequacy)
def compute_dsa(test_activation, train_activations):
    # Compute distances between test activation and all training activations
    distances = [np.linalg.norm(test_activation - train_activation) for train_activation in train_activations]
    
    # Find the minimum distance (nearest neighbor in terms of activation)
    d_nearest = min(distances)
    
    # Optionally, normalize using the mean distance across all activations
    avg_distance = np.mean(distances)
    dsa_score = d_nearest / avg_distance

    return dsa_score
def select_subset(train_activations, subset_ratio=0.25):
    total = len(train_activations)
    subset_size = int(total * subset_ratio)  # Determine the size of the subset
    return random.sample(train_activations, subset_size)  # Randomly sample the subset
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

        # # Load activations once during initialization
        # self.train_activations = self.load_train_activations()

        super().__init__(n_var=2, n_obj=1, n_constr=0, xl=[0, 0], xu=[1, 1], elementwise_evaluation=False, **kwargs, type_var=np.float64)

    def load_train_activations(self):
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

    def fit_kde(self, train_activations):
        # Fit KDE on training activations for LSA computation
        return KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train_activations)

        

    def _evaluate(self, X, out, *args, **kwargs):
        path = "D:/search_sets/mars/uncertainty/set" + str(self.numSet) + "/"
        os.makedirs(path, exist_ok = True) 
        F = []
        import psutil
        
        # with open('./archive.txt', "r") as f:
        #     file = f.read().split('\n')
        archive2 = []
        archive_features = []
        model_resnet = resnet18(pretrained=True)
        model_resnet.eval()
        # print(len(file), 'retrieved images from archive')
        # for f in file:
        #     input_image_path = f.split(',')[0].split(':')[1].split("'")[1]
        #     archive2.append(input_image_path)
        #     img = load_and_preprocess_image(input_image_path)
        #     features = extract_features(model_resnet, img)
        #     features = np.array(features)
        #     archive_features.append(features)  
        #     f11 = float(f.split(',')[5].split(': (')[1])
        #     f22 = float(f.split(',')[6].split(')')[0])
        #     F.append([f11, f22])
        #     entry = int(f.split(',')[3].split(':')[1])
        #     rot = int(f.split(',')[4].split(':')[1])
        #     self.problemDict.append({'entry': entry, 'rot': rot, 'F': [f11, f22]})
        #     x0 = float(input_image_path.split('R_')[1].split('_')[0])
        #     x1 = float(input_image_path.split('R_')[1].split('_')[1].replace('.png', ''))

        #     self.archive.append({'img': input_image_path, 'individual': (x0, x1), 'entry': entry, 'rot': rot, 'F': (f11, f22), 'PixAcc': 0})


        
        type_distance = 'FeatureDistance'
        threshold_diversity = 12
        print('@@@@@@@@@@@@', X)
        camera_position = airsim.Vector3r(0, 0, -1)  # x, y, z coordinates above the car
        camera_orientation = airsim.to_quaternion(-0.5, 0, 0)  # roll, pitch (downward), yaw in radians
        self.GaGan.client.simSetCameraPose("0", airsim.Pose(camera_position, camera_orientation))
        self.GaGan.client.simSetCameraPose("1", airsim.Pose(camera_position, camera_orientation))


        deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True)
        deeplab.eval()
        layer = deeplab.backbone.layer1[0].conv2
        dsa_threshold = 0.65
        lsa_threshold = 249670

        # # Directory containing saved activation patterns
        # activation_dir = "D:/pix2pixHD/datasets/train_activations/"

        # # Load all training activation patterns from disk and flatten them
        # # Load all training activation patterns from disk and flatten them
        # train_activations = []
        # subset_activations = select_subset(os.listdir(activation_dir), subset_ratio=0.05)

        # for file_name in subset_activations:
        #     if file_name.endswith("_activation.pkl"):
        #         file_path = os.path.join(activation_dir, file_name)
        #         train_pattern = load_activation_pattern(file_path)
        #         flattened_train_pattern = flatten_activation(train_pattern)
        #         train_activations.append(flattened_train_pattern)

        # # Stack training activations into a single 2D array for LSA computation
        # train_activations = np.vstack(train_activations)

        # # Fit KDE on training activations for LSA computation
        # kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train_activations)

        # print("Training Activations loaded")

        
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
            self.GaGan.client.simSetVehiclePose(pos, True, self.GaGan.vehicle_name)

            start_time = time.time()
            while time.time() - start_time > 10 and not is_car_stationary(self.GaGan):
                print("Waiting for the car to settle...")
                time.sleep(1)  # Check every second

            if not is_car_stationary(self.GaGan):
                self.GaGan.client.setCarControls(airsim.CarControls(brake=1.0))
                print("Applying brakes...")
                time.sleep(2)  # Apply brakes for 5 seconds
                self.GaGan.client.setCarControls(airsim.CarControls(brake=0.0))



            save_path = os.path.join(path, "R_" + str(x[0]) + "_" + str(x[1]) + ".png")
            # if(not save_path in archive2):
            if True:
                save_path = os.path.join(path, "R_" + str(x[0]) + "_" + str(x[1]) + ".png")
                simulated_path = os.path.join(path, "S_" + str(x[0]) + "_" + str(x[1]) + ".png")
                label_path = os.path.join(path, "L_" + str(x[0]) + "_" + str(x[1]) + ".png")
                label_GAN, label_DNN = None, None
                if not os.path.isfile(save_path):
                    t1 = time.time()
                    img_path = self.GaGan.retrieveImages(x, self.numSet)  # AirSim Response
                    sim_img = cv2.imread(simulated_path)
                    if sim_img is None:
                         raise ValueError(f"Image could not be loaded or is empty. Please check the image path.")

                    img_gray = cv2.cvtColor(sim_img, cv2.COLOR_RGB2GRAY)
                    cv2.imwrite(simulated_path, img_gray)
                    t4 = time.time()
                    label_DNN = generate_dnn_mask(label_path)
                    Image.fromarray(label_DNN.astype(np.uint8)).save(os.path.join(path, "D_" + str(x[0]) + "_" + str(x[1]) + ".png"))
                    label_GAN = generate_gan_mask(label_DNN)
                    # label_GAN = cv2.cvtColor(label_GAN, cv2.COLOR_BGR2RGB)
                    print("Label:", time.time()-t4)

                   

                    img = Image.fromarray((label_GAN * 1).astype(np.uint8))
                    img.save(os.path.join(path, "G_" + str(x[0]) + "_" + str(x[1]) + ".png"))

                    t6 = time.time()
                    # p.suspend()
                    transform_A = get_transform(opt, get_params(opt, img.size))
                    img_A = transform_A(img.convert('RGB')).unsqueeze(0)
                    generated = self.GAN.inference(img_A, torch.tensor([0]), None)
                    util.save_image(util.tensor2im(generated.data[0]), save_path)

                    print("GAN:", time.time()-t6)
                t3 = time.time()
                # image = load_image(save_path) 
                image = Image.open(save_path)    

                predictions = mc_dropout_inference(model, image, num_samples=5)
                variance_map = compute_pixelwise_variance(predictions)
                overall_uncertainty_score = compute_overall_uncertainty(variance_map, method='median')           

                
              
                print('Uncertainty = ', overall_uncertainty_score)
                print("DNN predict:", time.time()-t3)


                
                proportion = compute_all_color_proportion(label_path)

               
                if(proportion[0] == True):
                    f = 2
                    f2 = 2
                    print('proportion too high/low', proportion[1], proportion[2])
                else:
                    f = 1/(1+overall_uncertainty_score)
                    input_image_path = path + "R_" + str(x[0]) + "_" + str(x[1]) + ".png"
                    if(len(archive2) == 0):
                        
                        archive2.append(input_image_path)
                        if(type_distance != 'PixelAccuracy'):
                            img = load_and_preprocess_image(input_image_path)
                            features = extract_features(model_resnet, img)
                            features = np.array(features)
                            archive_features.append(features)     
                        f2 = 1               
                        self.archive.append({'img': save_path, 'individual': (x[0], x[1]), 'entry': int(x[0]*len(self.rows)), 'rot': int(x[1]*360), 'F': (f, f2), 'PixAcc': 0})
                        
                    else:

                        if(type_distance != 'PixelAccuracy'):
                            img = load_and_preprocess_image(input_image_path)
                            features = extract_features(model_resnet, img)
                            features = np.array(features)
                            closest_distance, closest_index = distance(features, archive_features, type_distance)
                        else:
                            closest_distance, closest_index = distance(input_image_path, archive2,type_distance)

                        

                        if(closest_distance > threshold_diversity): #  if PixelAccuracy use '<' if Euclidean use '>'
                            if(type_distance != 'PixelAccuracy'):
                                archive_features.append(features)

                            archive2.append(input_image_path)
                            print('added to archive', 'distance = ', closest_distance)
                            self.archive.append({'img': save_path, 'individual': (x[0], x[1]), 'entry': int(x[0]*len(self.rows)), 'rot': int(x[1]*360), 'F': (f, f2), 'PixAcc': 0})
                            f2 = 1 / (1 + closest_distance) # f2 = 1 / (1 + closest_distance) if Euclidean, f2 = closest_distance if PixAcc
                        else:
                            closest_image_path = archive2[closest_index]
                            closest_image = Image.open(closest_image_path) 


                            predictions = mc_dropout_inference(model, closest_image, num_samples=10)
                            variance_map = compute_pixelwise_variance(predictions)
                            overall_uncertainty_score = compute_overall_uncertainty(variance_map, method='median')  

                            f_closest = 1/(1+overall_uncertainty_score)

                             

                            if(f_closest > f):
                                self.archive[closest_index] = {'img': save_path, 'individual': (x[0], x[1]), 'entry': int(x[0]*len(self.rows)), 'rot': int(x[1]*360), 'F': (f, f2), 'PixAcc': 0}
                                print('replaced in archive', 'fitnesses=', f,f2)
                                archive2[closest_index] = input_image_path
                                f2 = 1
                                if(type_distance != 'PixelAccuracy'):
                                    archive_features[closest_index] = features
                            else:
                                f2 = 2
                                print('image not diverse')


               
            print("total:", time.time()-t1)
            F.append([f, f2])
            self.problemDict.append({'entry': x[0], 'rot': x[1], 'F': [f, f2]})
            with open(path + 'archive.txt', 'w') as fp:
                    for item in self.archive:
                        fp.write("%s\n" % item)
            # else:
            #     for i in range(len(self.archive)):
            #         if(self.archive[i]['img'] == save_path):
            #             F.append(self.archive[i]['F'])  
            #             print(F)
            #             break      
                
        if(type_distance != 'PixelAccuracy'):
            current_features = []
            currentX = []
            for i in range(len(X)):
                x = X[i]
                img_path = path + "R_" + str(x[0]) + "_" + str(x[1]) + ".png"
                if(img_path in archive2):
                    img_index = archive2.index(img_path)
                    current_features.append(archive_features[img_index])
                    currentX.append(x)
            for i in range(len(current_features)):
                current_features2 = [item for index, item in enumerate(current_features) if index != i]
                if(len(current_features2) > 0):
                    closest_distance, closest_index = distance(current_features[i], current_features2, type_distance)
                    F[i][1] = 1 / (1 + closest_distance)
                    x = currentX[i]
                    for j in range(len( self.problemDict)):
                        if( self.problemDict[j]['entry'] == x[0] and  self.problemDict[j]['rot'] == x[1]):
                            self.problemDict[j]['F'] = F
                            break


        

        else:
            current_paths = []
            currentX = []
            for i in range(len(X)):
                x = X[i]
                img_path = path + "R_" + str(x[0]) + "_" + str(x[1]) + ".png"
                if(img_path in archive2):
                    img_index = archive2.index(img_path)
                    current_paths.append(img_path)
                    currentX.append(x)
            for i in range(len(current_paths)):
                if(len(current_paths) > 1):
                    closest_distance, closest_index = distance(current_paths[i], current_paths, type_distance)
                    F[i][1] = closest_distance
                    x = currentX[i]
                    for j in range(len( self.problemDict)):
                        if( self.problemDict[j]['entry'] == x[0] and  self.problemDict[j]['rot'] == x[1]):
                            self.problemDict[j]['F'] = F
                            break






        if out is None: return F
        else: out["F"] = np.array(F)

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
    # Filter out 'nan' values and compute the mean
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

def generate_gan_mask(dnn_mask):
    color_map_GAN = {
        3: (255, 0, 0),       # big rock, red
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
def generate_label(path):
#    import numpy as np
#    import cv2
#    import os
#    from PIL import Image

    # Define the class mappings from RGB to Cityscapes
    class_map = {
        (187, 70, 156): (128, 64, 128), (112, 105, 191): (244, 35, 232),
        (89, 121, 72): (70, 70, 70), (0, 53, 65): (70, 70, 70),
        (28, 34, 108): (70, 70, 70), (49, 89, 160): (70, 70, 70),
        (190, 225, 64): (102, 102, 156), (206, 190, 59): (190, 153, 153),
        # (206, 190, 59):(107, 142, 35),
        (135, 169, 180): (153, 153, 153), (115, 176, 195): (244, 35, 232),
        (49, 89, 160): (70, 70, 70), (81, 13, 36): (107, 142, 35),
        (29, 26, 199): (70, 130, 180), (102, 16, 239): (70, 130, 180),
        (189, 135, 188): (220, 20, 60), (156, 198, 23): (255, 0, 0),
        (161, 171, 27): (0, 0, 142), (68, 218, 116): (0, 0, 70),
        (11, 236, 9): (0, 60, 100), (196, 30, 8): (0, 80, 100),
        (121, 67, 28): (0, 0, 230), (148, 66, 130): (0, 0, 142),
        (255, 0, 0): (107, 142, 35), (250, 170, 30): (244, 35, 232),
        (153, 108, 6): (128, 64, 128), (131, 182, 184): (153, 153, 153),
        (0, 0, 0): (70, 130, 180)
    }

    rgb_mask = np.array(Image.open(path))
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGRA2BGR)
    # Convert the RGB mask to Cityscapes format
    cityscapes_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1], 3), dtype=np.uint8)
    for i in range(rgb_mask.shape[0]):
        for j in range(rgb_mask.shape[1]):
            #         print(rgb_mask[i,j])
            rgb_val = tuple(rgb_mask[i, j, :])
            if rgb_val in class_map:
                cityscapes_mask[i, j] = np.array(class_map[rgb_val])
            else:
                cityscapes_mask[i, j] = np.array(rgb_val)

    # Save the Cityscapes segmentation mask
    Image.fromarray(cityscapes_mask).save(path)

def predict(model, path):
    print(path)
    t1 = time.time()
    img = mx_img.imread(path)
    t2 = time.time()
    img = test_transform(img, ctx)
    t3 = time.time()
    output = model.predict(img)
    t4 = time.time()
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    t5 = time.time()
    pred = get_color_pallete(predict, 'ade20k')
    t6 = time.time()
    print(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
    return pred


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    )  # .reshape(n_class, n_class)
    while (len(hist) > n_class * n_class):
        hist = np.delete(hist, -1)
    hist = hist.reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def miou(label_trues, label_preds, n_class):
    score = scores(label_trues, label_preds, n_class)
    cls_iu = score['Class IoU']
    # miou = []
    # for key in cls_iu:
    #     if (cls_iu[key] > 0):
    #         miou.append(cls_iu[key])
    # return np.average(miou)
    return cls_iu[14]

def mask_label(file):
    airsim_to_cityscapes = {
        (187, 70, 156): (128, 64, 128),
        (112, 105, 191): (244, 35, 232),
        (89, 121, 72):(70, 70, 70),
        (0,53,65):(70, 70, 70),
        (28,34,108):(70, 70, 70),
        (49,89,160):(70, 70, 70),    
        (190, 225, 64):(102, 102, 156),
        (206, 190, 59):(190, 153, 153),
        # (206, 190, 59):(107, 142, 35),
        (135, 169, 180):(153, 153, 153),
        (115, 176, 195):(244, 35, 232),
        (49, 89, 160):(70, 70, 70),
        (81, 13, 36):(107, 142, 35),
        (29, 26, 199):(70, 130, 180),
        (102, 16, 239):(70, 130, 180),
        (189, 135, 188):(220, 20, 60),
        (156, 198, 23):(255, 0, 0),
        (161, 171, 27):(0, 0, 142),
        (68, 218, 116):(0, 0, 70),
        (11, 236, 9):(0, 60, 100),
        (196, 30, 8):(0, 80, 100),
        (121, 67, 28):(0, 0, 230),
        (148,66,130):(0, 0, 142),
        (255,0,0):(107, 142, 35),
        (250,170,30):(244, 35, 232),
        (153,108,6): (128, 64, 128),
        (131,182,184): (153, 153, 153), 
        (0, 0, 0): (70, 130, 180)
        }
    airsim_rgb_to_cityscapes = {(128, 64, 128): 1,
     (244, 35, 232): 2,
     (70, 70, 70): 3,
     (102, 102, 156): 4,
     (190, 152, 153): 5,
     (153, 153, 153): 6,
     (250, 170, 30): 7,
     (220, 220, 0): 8,
     (107, 142, 35): 9,
     (152, 251, 152): 10,
     (70, 130, 180): 11,
     (220, 20, 60): 12,
     (255, 0, 0): 13,
     (0, 0, 142): 14,
     (0, 0, 70): 15,
     (0, 60, 100): 16,
     (0, 80, 100): 17,
     (0, 0, 230): 18,
     (119, 11, 32): 19,
    }

    # Example AirSim RGB segmentation mask (replace this with your actual RGB mask)
    # airsim_rgb_mask = np.random.randint(0, 256, size=(256, 256, 3))  # Example random RGB mask
    airsim_rgb_mask = np.array(Image.open(file))
    airsim_rgb_mask = cv2.cvtColor(airsim_rgb_mask, cv2.COLOR_BGRA2BGR)
    # Create a lookup table for quick color retrieval
    lookup_table = {}
    for airsim_color, cityscapes_color in airsim_to_cityscapes.items():
        lookup_table[str(airsim_color)] = cityscapes_color

    # Function to convert AirSim RGB mask to Cityscapes RGB colors
    def convert_to_cityscapes_rgb(rgb_mask):
        height, width, _ = rgb_mask.shape
        reshaped_mask = rgb_mask.reshape(-1, 3)
        mask_str = np.array([str(tuple(pixel)) for pixel in reshaped_mask])
        cityscapes_colors = np.array([lookup_table.get(color_str, [0, 0, 0]) for color_str in mask_str])
        return cityscapes_colors.reshape(height, width, 3)
    
    
    cityscapes_rgb_colors = convert_to_cityscapes_rgb(airsim_rgb_mask)
    
    height, width, _ = cityscapes_rgb_colors.shape
    cityscapes_labels = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            airsim_color = tuple(cityscapes_rgb_colors[y, x])
            if airsim_color in airsim_rgb_to_cityscapes:
                cityscapes_labels[y, x] = airsim_rgb_to_cityscapes[airsim_color]
            else:
                # Assign a default label for unknown colors
                cityscapes_labels[y, x] = -1  # Replace with your choice

    # Example output (displaying the converted colors)
    return cityscapes_rgb_colors, cityscapes_labels

def miou_label(file):
    airsim_to_cityscapes = {
        (187, 70, 156): (128, 64, 128),
        (112, 105, 191): (244, 35, 232),
        (89, 121, 72):(70, 70, 70),
        (0,53,65):(70, 70, 70),
        (28,34,108):(70, 70, 70),
        (49,89,160):(70, 70, 70),
        (190, 225, 64):(102, 102, 156),
        (206, 190, 59):(190, 153, 153),
        # (206, 190, 59):(107, 142, 35),
        (135, 169, 180):(153, 153, 153),
        (115, 176, 195):(244, 35, 232),
        (49, 89, 160):(70, 70, 70),
        (81, 13, 36):(107, 142, 35),
        (29, 26, 199):(70, 130, 180),
        (102, 16, 239):(70, 130, 180),
        (189, 135, 188):(220, 20, 60),
        (156, 198, 23):(255, 0, 0),
        (161, 171, 27):(0, 0, 142),
        (68, 218, 116):(0, 0, 70),
        (11, 236, 9):(0, 60, 100),
        (196, 30, 8):(0, 80, 100),
        (121, 67, 28):(0, 0, 230),
        (148,66,130):(0, 0, 142),
        (255,0,0):(107, 142, 35),
        (250,170,30):(244, 35, 232),
        (153,108,6): (128, 64, 128),
        (131,182,184): (153, 153, 153),
        (0, 0, 0): (70, 130, 180)
        }

    # Example AirSim RGB segmentation mask (replace this with your actual RGB mask)
    # airsim_rgb_mask = np.random.randint(0, 256, size=(256, 256, 3))  # Example random RGB mask
    airsim_rgb_mask = np.array(Image.open(file))
    airsim_rgb_mask = cv2.cvtColor(airsim_rgb_mask, cv2.COLOR_BGRA2BGR)
    # Create a lookup table for quick color retrieval
    lookup_table = {}
    for airsim_color, cityscapes_color in airsim_to_cityscapes.items():
        lookup_table[str(airsim_color)] = cityscapes_color

    # Function to convert AirSim RGB mask to Cityscapes RGB colors
    def convert_to_cityscapes_rgb(rgb_mask):
        height, width, _ = rgb_mask.shape
        reshaped_mask = rgb_mask.reshape(-1, 3)
        mask_str = np.array([str(tuple(pixel)) for pixel in reshaped_mask])
        cityscapes_colors = np.array([lookup_table.get(color_str, [0, 0, 0]) for color_str in mask_str])
        return cityscapes_colors.reshape(height, width, 3)

    # Convert AirSim RGB mask to Cityscapes RGB colors
    cityscapes_rgb_colors = convert_to_cityscapes_rgb(airsim_rgb_mask)

    # Example output (displaying the converted colors)
    return cityscapes_rgb_colors

def relabel(path):
    color_label_image = Image.open(path)
    color_to_label_mapping = {(128, 64, 128): 1,
                              (244, 35, 232): 2,
                              (70, 70, 70): 3,
                              (102, 102, 156): 4,
                              (190, 152, 153): 5,
                              (153, 153, 153): 6,
                              (250, 170, 30): 7,
                              (220, 220, 0): 8,
                              (107, 142, 35): 9,
                              (152, 251, 152): 10,
                              (70, 130, 180): 11,
                              (220, 20, 60): 12,
                              (255, 0, 0): 13,
                              (0, 0, 142): 14,
                              (0, 0, 70): 15,
                              (0, 60, 100): 16,
                              (0, 80, 100): 17,
                              (0, 0, 230): 18,
                              (119, 11, 32): 19,
                              }

    # Convert the color label image to a NumPy array
    color_label_array = np.array(color_label_image)

    # Initialize a NumPy array to store the transformed label mask
    label_mask = np.zeros((color_label_array.shape[0], color_label_array.shape[1]), dtype=np.uint8)

    # Map the RGB colors to class labels and create the label mask
    for i in range(color_label_array.shape[0]):
        for j in range(color_label_array.shape[1]):
            pixel_color = tuple(color_label_array[i, j][:3])

            if pixel_color in color_to_label_mapping:
                label_mask[i, j] = color_to_label_mapping[pixel_color]

    # Convert the NumPy array to a label mask image
    label_mask_image = Image.fromarray(label_mask)
    return label_mask
def getGAN():
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
        #if opt.verbose: print(model)
        model.opt = opt
    else:
        model = None
    print('Model loaded')
    return model
class RankAndCrowdingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, n_survive, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]
