# !pip install segmentation-models-pytorch torchinfo

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dataclasses import dataclass
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch.optim as TorchOptimizers
import torch.amp
from torchinfo import summary as torch_summary
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import time
import wandb
import pandas as pd
import albumentations as A
import json
import math
from typing import Callable
from pathlib import Path
import random

# from google.colab import drive
# drive.mount('/content/drive')

# wandb.login()

@dataclass
class Environment:
    train_images_folder: str
    train_labels_folder: str
    val_images_folder: str
    val_labels_folder: str
    saved_weights_filepath: str
    training_output_folder: str
    device: str

    def fetch_train_filenames(self):
        names = np.array([f.name for f in Path(self.train_images_folder).iterdir()])
        if config.fraction >= .99:
            return names
        fraction_of_names = random.choices(names, k=math.ceil(len(names) * config.fraction))
        return fraction_of_names

    def fetch_val_filenames(self):
        return np.array([f.name for f in Path(self.val_images_folder).iterdir()])


env: Environment = None
""" Set to appropriate environment before training/inference """;

local_env = Environment(
    train_images_folder='data/train_images/',
    train_labels_folder='data/train_labels/',
    val_images_folder='data/test_images/',
    val_labels_folder='data/test_labels/',
    saved_weights_filepath='data_gen/best.pt',
    training_output_folder='data_gen/',
    device='cpu',
)
colab_home_folder = '/content/drive/My Drive/Colab Notebooks/retina-segmentation/'
colab_env = Environment(
    train_images_folder=colab_home_folder + 'data/train_images/',
    train_labels_folder=colab_home_folder + 'data/train_labels/',
    val_images_folder=colab_home_folder + 'data/test_images/',
    val_labels_folder=colab_home_folder + 'data/test_labels/',
    saved_weights_filepath=colab_home_folder + 'data_gen/best.pt',
    training_output_folder=colab_home_folder + 'data_gen/',
    device='cuda',
)

class Config:
    def __init__(self, training, verbose=False, debug=False):
        self.verbose = verbose
        self.training = training
        if self.training:
            os.makedirs(env.training_output_folder, exist_ok=True)

        self.num_classes = 1
        self.seed = 8675309
        self.batch_size = 32
        self.starting_learning_rate = 1e-4
        self.max_epochs = 400
        self.patience = 100
        self.num_workers = 9 if env.device == 'cuda' and not debug else 0
        self.pin_memory = self.num_workers > 0
        self.use_amp = env.device == 'cuda'
        self.original_image_width = 768
        self.original_image_height = 576
        self.image_width = 768
        self.image_height = 576
        self.fraction = 0.1 if debug else 1

        self.imagenet_mean_cpu_tensor = torch.tensor(imagenet_mean_array)
        self.imagenet_std_cpu_tensor = torch.tensor(imagenet_std_array)
        self.channelwise_imagenet_mean_cpu_tensor = self.imagenet_mean_cpu_tensor.view(3, 1, 1)
        self.channelwise_imagenet_std_cpu_tensor = self.imagenet_std_cpu_tensor.view(3, 1, 1)
        self.imagenet_mean_gpu_tensor = gpu_tensor(imagenet_mean_array)
        self.imagenet_std_gpu_tensor = gpu_tensor(imagenet_std_array)
        self.channelwise_imagenet_mean_gpu_tensor = self.imagenet_mean_gpu_tensor.view(3, 1, 1)
        self.channelwise_imagenet_std_gpu_tensor = self.imagenet_std_gpu_tensor.view(3, 1, 1)

        self.encoder_name = 'resnet34'
        self.encoder_weights = 'imagenet'

        torch.manual_seed(self.seed)
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)

        self.train_transforms = A.Compose([
            A.Resize(self.image_height, self.image_width),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.03, 0.03),
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.OneOf([
                A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(0.15, 0.15, p=1.0),
                A.RandomGamma(gamma_limit=(85, 115), p=1.0),
            ], p=0.5),
            A.Normalize(mean=imagenet_mean_tuple, std=imagenet_std_tuple),
            A.ToTensorV2(),
        ])

        self.val_transforms = A.Compose([
            A.Resize(self.image_height, self.image_width),
            A.Normalize(mean=imagenet_mean_tuple, std=imagenet_std_tuple),
            A.ToTensorV2(),
        ])

        self.test_transforms = A.Compose([
            A.Normalize(mean=imagenet_mean_tuple, std=imagenet_std_tuple),
            A.ToTensorV2(),
        ])


config: Config = None
""" Create and assign before training/inference """;

imagenet_mean_tuple = (0.485, 0.456, 0.406)
imagenet_std_tuple = (0.229, 0.224, 0.225)
imagenet_mean_array = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std_array = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_COLORS = np.array([
    [  0,   0,   0], # 0: Black (Background)
    [255, 255, 255], # 1: White (Blood vessel)
])

def gpu_tensor(numpy_array):
    return torch.tensor(numpy_array, device=env.device)

def gpu_image_tensor_to_numpy_array(image_tensor):
    image = denormalize(image_tensor, config.channelwise_imagenet_mean_gpu_tensor, config.channelwise_imagenet_std_gpu_tensor)
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    return (image * 255).astype(np.uint8)

def gpu_mask_tensor_to_colored_mask_numpy_array(mask_tensor):
    mask = mask_tensor.cpu().numpy()
    mask = np.clip(mask, 0, len(CLASS_COLORS) - 1).astype(np.int32)
    return CLASS_COLORS[mask]

def visualize_image(image_tensor):
    image = gpu_image_tensor_to_numpy_array(image_tensor)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    plt.close()

def visualize_mask(mask_tensor):
    colored_mask = gpu_mask_tensor_to_colored_mask_numpy_array(mask_tensor)
    plt.imshow(colored_mask)
    plt.axis('off')
    plt.show()
    plt.close()

def visualize_mask_overlayed_over_image(image_tensor, mask_tensor, alpha=0.5):
    image_array = gpu_image_tensor_to_numpy_array(image_tensor)
    colored_mask = gpu_mask_tensor_to_colored_mask_numpy_array(mask_tensor)
    blended = (alpha * colored_mask + (1 - alpha) * image_array).astype(np.uint8)
    plt.imshow(blended)
    plt.axis('off')
    plt.show()
    plt.close()

def normalize(tensor, mean, std):
    return (tensor - mean) / std

def denormalize(tensor, mean, std):
    return tensor * std + mean

def print_model_torchinfo(model: nn.Module):
    print(torch_summary(model, input_size=(1, 3, config.image_width, config.image_height)))

def print_model(model: nn.Module):
    for name, module in model.named_modules():
        print(name, "->", module.__class__.__name__)

def create_dataloader(dataset, shuffle):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=config.pin_memory, generator=config.generator)

def num_batches_from_loader(dataloader):
    return math.ceil(len(dataloader.dataset) / config.batch_size)


class MetricsAccumulator:
    def __init__(self):
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.total_correct = 0
        self.total_pixels = 0
        self.vessel_intersection = 0
        self.vessel_union = 0
        self.bg_intersection = 0
        self.bg_union = 0

    @torch.no_grad()
    def update(self, logits, targets):
        targets = targets.unsqueeze(1).float()
        targets = (targets > 0.5).float()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        self.total_correct += (preds == targets).sum().item()
        self.total_pixels += targets.numel()

        v_inter = (preds * targets).sum().item()
        v_union = preds.sum().item() + targets.sum().item() - v_inter
        self.vessel_intersection += v_inter
        self.vessel_union += v_union

        preds_bg = 1.0 - preds
        targets_bg = 1.0 - targets
        bg_inter = (preds_bg * targets_bg).sum().item()
        bg_union = preds_bg.sum().item() + targets_bg.sum().item() - bg_inter
        self.bg_intersection += bg_inter
        self.bg_union += bg_union

    def compute(self):
        accuracy = self.total_correct / (self.total_pixels + 1e-6)
        vessel_iou = self.vessel_intersection / (self.vessel_union + 1e-6)
        bg_iou = self.bg_intersection / (self.bg_union + 1e-6)
        mean_iou = (vessel_iou + bg_iou) / 2.0
        return {
            'accuracy': accuracy,
            'vessel_iou': vessel_iou,
            'bg_iou': bg_iou,
            'mean_iou': mean_iou,
        }


def plot_training_metrics(history):
    num_epochs = len(history['train_loss'])
    epochs = list(range(1, num_epochs + 1))

    def _make_plot(title, series, ylabel, wandb_key):
        plt.figure(figsize=(10, 6))
        for label, key in series:
            plt.plot(epochs, history[key], label=label, marker='o', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        wandb.log({wandb_key: wandb.Image(plt.gcf())})
        plt.show()
        plt.close()

    _make_plot('Training and Validation Loss', [
        ('Train Loss', 'train_loss'),
        ('Val Loss', 'val_loss'),
    ], 'Loss', 'loss_plot')

    _make_plot('Accuracy', [
        ('Train', 'train_accuracy'),
        ('Val (fixed-size)', 'val_accuracy'),
        ('Fullsize Global', 'fullsize_global_accuracy'),
        ('Fullsize Per-Image', 'fullsize_per_image_accuracy'),
    ], 'Accuracy', 'accuracy_plot')

    _make_plot('Vessel IoU', [
        ('Train', 'train_vessel_iou'),
        ('Val (fixed-size)', 'val_vessel_iou'),
        ('Fullsize Global', 'fullsize_global_vessel_iou'),
        ('Fullsize Per-Image', 'fullsize_per_image_vessel_iou'),
    ], 'IoU', 'vessel_iou_plot')

    _make_plot('Background IoU', [
        ('Train', 'train_bg_iou'),
        ('Val (fixed-size)', 'val_bg_iou'),
        ('Fullsize Global', 'fullsize_global_bg_iou'),
        ('Fullsize Per-Image', 'fullsize_per_image_bg_iou'),
    ], 'IoU', 'bg_iou_plot')

    _make_plot('Mean IoU', [
        ('Train', 'train_mean_iou'),
        ('Val (fixed-size)', 'val_mean_iou'),
        ('Fullsize Global', 'fullsize_global_mean_iou'),
        ('Fullsize Per-Image', 'fullsize_per_image_mean_iou'),
    ], 'IoU', 'mean_iou_plot')

    _make_plot('Full-Size Validation Metrics', [
        ('Global Accuracy', 'fullsize_global_accuracy'),
        ('Per-Image Accuracy', 'fullsize_per_image_accuracy'),
        ('Global Vessel IoU', 'fullsize_global_vessel_iou'),
        ('Global BG IoU', 'fullsize_global_bg_iou'),
        ('Global Mean IoU', 'fullsize_global_mean_iou'),
        ('Per-Image Vessel IoU', 'fullsize_per_image_vessel_iou'),
        ('Per-Image BG IoU', 'fullsize_per_image_bg_iou'),
        ('Per-Image Mean IoU', 'fullsize_per_image_mean_iou'),
    ], 'Value', 'fullsize_all_metrics_plot')