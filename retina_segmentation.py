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

@dataclass
class Environment:
    train_images_folder: str
    train_labels_folder: str
    val_images_folder: str
    val_labels_folder: str
    saved_weights_filepath: str
    training_output_folder: str
    device: str


env: Environment = None
""" Set to appropriate environment before training/inference """

local_env = Environment(
    train_images_folder='data/train_images/',
    train_labels_folder='data/train_labels/',
    val_images_folder='data/test_images/',
    val_labels_folder='data/test_labels/',
    saved_weights_filepath='data_gen/best.pt',
    training_output_folder='data_gen/',
    device='cpu'
)

class Config:
    def __init__(self, training, verbose=False):
        self.verbose = verbose
        self.training = training
        if self.training:
            os.makedirs(env.training_output_folder, exist_ok=True)

        self.num_classes = 1
        self.seed = 8675309
        self.batch_size = 32
        self.starting_learning_rate = 1e-4
        self.max_epochs = 200
        self.patience = 50
        self.layers_to_unfreeze = 10
        self.num_workers = 9 if env.device == 'cuda' else 0
        self.pin_memory = self.num_workers > 0
        self.use_amp = env.device == 'cuda'
        self.original_image_width = 768
        self.original_image_height = 576
        self.image_width = 768
        self.image_height = 576

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

config: Config = None
""" Create and assign before training/inference """

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
    mask = np.clip(mask, 0, config.num_classes - 1).astype(np.int32)
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

def _num_batches(dataloader):
    return math.ceil(len(dataloader.dataset) / config.batch_size)

@dataclass
class SegmentationDataset(TorchDataset):
    images_root_folder: str
    masks_root_folder: str
    image_suffix: str
    mask_suffix: str
    image_transforms: A.Compose
    image_ids: np.ndarray
    has_masks: bool

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.cvtColor(cv2.imread(f'{self.images_root_folder}{image_id}.{self.image_suffix}'), cv2.COLOR_BGR2RGB)
        if self.has_masks:
            mask = cv2.imread(f'{self.masks_root_folder}{image_id}.{self.mask_suffix}', cv2.IMREAD_GRAYSCALE)
            transformed = self.image_transforms(image=image, mask=mask)
            return transformed['image'], transformed['mask']
        transformed = self.image_transforms(image=image)
        return transformed['image']


class RetinaSegModel(nn.Module):
    def __init__(self, saved_model_weights=None):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=3,
            classes=config.num_classes,
        )

        if saved_model_weights is not None:
            saved_model_weights = torch.load(saved_model_weights, weights_only=True, map_location='cpu')
            self.model.load_state_dict(saved_model_weights)

    def forward(self, x):
        return self.model(x)





def main():
    global env, config
    env = local_env
    config = Config(training=True)
    model = RetinaSegModel()
    print_model_torchinfo(model.model)
    print_model(model)

if __name__ == '__main__':
    main()