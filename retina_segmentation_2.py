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

def _num_batches(dataloader):
    return math.ceil(len(dataloader.dataset) / config.batch_size)

@dataclass
class SegmentationDataset(TorchDataset):
    images_root_folder: str
    masks_root_folder: str
    image_transforms: A.Compose
    image_filenames: np.ndarray
    has_masks: bool

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image = cv2.cvtColor(cv2.imread(f'{self.images_root_folder}{image_filename}'), cv2.COLOR_BGR2RGB)
        if self.has_masks:
            mask_stem = Path(image_filename).stem
            mask_path = next(Path(self.masks_root_folder).glob(f'{mask_stem}.*'))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            transformed = self.image_transforms(image=image, mask=mask)
            return transformed['image'], transformed['mask']
        transformed = self.image_transforms(image=image)
        return transformed['image']

def plot_val_images_and_masks(num_images_to_visualize=3):
    val_dataset = SegmentationDataset(
        images_root_folder=env.val_images_folder,
        masks_root_folder=env.val_labels_folder,
        image_transforms=config.val_transforms,
        image_filenames=env.fetch_val_filenames(),
        has_masks=True,
    )

    fig, axes = plt.subplots(num_images_to_visualize, 2, figsize=(10, 5 * num_images_to_visualize))
    axes[0, 0].set_title('Image')
    axes[0, 1].set_title('Mask')

    for i in range(num_images_to_visualize):
        image_tensor, mask_tensor = val_dataset[i]
        image = gpu_image_tensor_to_numpy_array(image_tensor)
        colored_mask = gpu_mask_tensor_to_colored_mask_numpy_array(mask_tensor)

        axes[i, 0].imshow(image)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(colored_mask)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

def val_mean_iou(pred_masks):
    val_filenames = env.fetch_val_filenames()
    gt_dataset = SegmentationDataset(
        images_root_folder=env.val_images_folder,
        masks_root_folder=env.val_labels_folder,
        image_transforms=config.test_transforms,
        image_filenames=val_filenames,
        has_masks=True,
    )

    # Accumulators for global mIoU (sum intersection/union across all images, then divide)
    global_intersection = np.zeros(2)
    global_union = np.zeros(2)

    # Per-image mIoU list
    ious_per_image = []

    for i in range(len(gt_dataset)):
        _, gt_mask = gt_dataset[i]
        gt_mask = gt_mask.numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask
        pred_mask = pred_masks[i].cpu().numpy() if isinstance(pred_masks[i], torch.Tensor) else np.array(pred_masks[i])

        # Resize pred to GT dimensions if they differ
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Binarize both masks
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)

        # Per-class IoU (background=0, vessel=1)
        class_ious = []
        for c in range(2):
            gt_c = (gt_binary == c)
            pred_c = (pred_binary == c)
            intersection = np.logical_and(gt_c, pred_c).sum()
            union = np.logical_or(gt_c, pred_c).sum()
            global_intersection[c] += intersection
            global_union[c] += union
            iou = intersection / (union + 1e-6)
            class_ious.append(iou)

        ious_per_image.append(np.mean(class_ious))

    per_image_miou = np.mean(ious_per_image)
    global_class_ious = global_intersection / (global_union + 1e-6)
    global_miou = np.mean(global_class_ious)

    print(f'Per-image mIoU: {per_image_miou:.4f}')
    print(f'Global mIoU:    {global_miou:.4f}')
    return global_miou

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

        self.model.to(env.device)

    def forward(self, x):
        return self.model(x)


class RetinaSegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets.unsqueeze(1).float()
        targets = (targets > 0.5).float()
        return self.loss(logits, targets)


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


def train_one_epoch(start_time, model, loader, optimizer, loss_function, scaler, scheduler, metrics):
    model.train()
    metrics.reset()
    running_loss = 0.0

    num_batches = _num_batches(loader)
    for batch_number, (x, y) in enumerate(loader):
        print(f't={time.time() - start_time:.2f}: Loading training batch {batch_number + 1}/{num_batches}')

        x = x.to(env.device, non_blocking=True)
        y = y.to(env.device, non_blocking=True)

        if config.verbose or batch_number == 0:
            allocated = torch.cuda.memory_allocated(env.device) / 1024**3
            reserved = torch.cuda.memory_reserved(env.device) / 1024**3
            print(f'Memory allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB')
            print(f'First image with overlayed ground-truth mask:')
            visualize_mask_overlayed_over_image(x[0], (y[0] > 0).long())

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=config.use_amp):
            logits = model(x)
            if config.verbose or batch_number == 0:
                logits_0_mask = (logits[0, 0] > 0).long()
                print(f'First image with overlayed prediction mask:')
                visualize_mask_overlayed_over_image(x[0], logits_0_mask)
            loss = loss_function(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        metrics.update(logits.detach(), y)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, metrics.compute()

@torch.no_grad()
def validate_one_epoch(start_time, model, loader, loss_function, metrics):
    model.eval()
    metrics.reset()
    running_loss = 0.0
    pred_masks = []

    num_batches = _num_batches(loader)
    for batch_number, (x, y) in enumerate(loader):
        print(f'v={time.time() - start_time:.2f}: Loading validation batch {batch_number + 1}/{num_batches}')

        x = x.to(env.device, non_blocking=True)
        y = y.to(env.device, non_blocking=True)

        if config.verbose or batch_number == 0:
            print(f'First image with overlayed ground-truth mask:')
            visualize_mask_overlayed_over_image(x[0], (y[0] > 0).long())

        with torch.amp.autocast('cuda', enabled=config.use_amp):
            logits = model(x)
            if config.verbose or batch_number == 0:
                logits_0_mask = (logits[0, 0] > 0).long()
                print(f'First image with overlayed prediction mask:')
                visualize_mask_overlayed_over_image(x[0], logits_0_mask)
            loss = loss_function(logits, y)

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        metrics.update(logits, y)
        batch_masks = (logits[:, 0] > 0).long().cpu()
        pred_masks.extend(batch_masks)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, metrics.compute(), pred_masks


def fullsize_validate(pred_masks, fullsize_gt_dataset):
    global_correct = 0
    global_pixels = 0
    global_vessel_inter = 0
    global_vessel_union = 0
    global_bg_inter = 0
    global_bg_union = 0

    per_image_accuracies = []
    per_image_vessel_ious = []
    per_image_bg_ious = []
    per_image_mean_ious = []

    for i in range(len(fullsize_gt_dataset)):
        _, gt_mask = fullsize_gt_dataset[i]
        gt_mask = gt_mask.numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask

        pred_mask = pred_masks[i].cpu().numpy() if isinstance(pred_masks[i], torch.Tensor) else np.array(pred_masks[i])

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)

        # Accuracy
        correct = (pred_binary == gt_binary).sum()
        pixels = gt_binary.size
        global_correct += correct
        global_pixels += pixels
        per_image_accuracies.append(correct / (pixels + 1e-6))

        # Vessel IoU (class=1)
        gt_v = (gt_binary == 1)
        pred_v = (pred_binary == 1)
        v_inter = np.logical_and(gt_v, pred_v).sum()
        v_union = np.logical_or(gt_v, pred_v).sum()
        global_vessel_inter += v_inter
        global_vessel_union += v_union
        img_vessel_iou = v_inter / (v_union + 1e-6)
        per_image_vessel_ious.append(img_vessel_iou)

        # Background IoU (class=0)
        gt_bg = (gt_binary == 0)
        pred_bg = (pred_binary == 0)
        bg_inter = np.logical_and(gt_bg, pred_bg).sum()
        bg_union = np.logical_or(gt_bg, pred_bg).sum()
        global_bg_inter += bg_inter
        global_bg_union += bg_union
        img_bg_iou = bg_inter / (bg_union + 1e-6)
        per_image_bg_ious.append(img_bg_iou)

        per_image_mean_ious.append((img_vessel_iou + img_bg_iou) / 2.0)

    global_accuracy = global_correct / (global_pixels + 1e-6)
    global_vessel_iou = global_vessel_inter / (global_vessel_union + 1e-6)
    global_bg_iou = global_bg_inter / (global_bg_union + 1e-6)
    global_mean_iou = (global_vessel_iou + global_bg_iou) / 2.0

    return {
        'global_accuracy': global_accuracy,
        'per_image_accuracy': float(np.mean(per_image_accuracies)),
        'global_vessel_iou': global_vessel_iou,
        'global_bg_iou': global_bg_iou,
        'global_mean_iou': global_mean_iou,
        'per_image_vessel_iou': float(np.mean(per_image_vessel_ious)),
        'per_image_bg_iou': float(np.mean(per_image_bg_ious)),
        'per_image_mean_iou': float(np.mean(per_image_mean_ious)),
    }

def _plot_training_metrics(history):
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


def train(
        start_epoch = 1,
        saved_model_weights = None,
):
    start_time = time.time()
    print('t=0: Starting data prep and model loading')

    run = wandb.init(
        project='retina-segmentation',
        name=f'run={int(start_time)}',
        config={
            'batch_size': config.batch_size,
            'starting_learning_rate': config.starting_learning_rate,
            'max_epochs': config.max_epochs,
            'patience': config.patience,
            'seed': config.seed,
            'model': 'UNet++',
            'encoder': config.encoder_name,
        },
    )

    train_filenames = env.fetch_train_filenames()
    val_filenames = env.fetch_val_filenames()

    model = RetinaSegModel(saved_model_weights=saved_model_weights)

    wandb.watch(model, log='gradients', log_freq=100)

    train_dataset = SegmentationDataset(
        images_root_folder=env.train_images_folder,
        masks_root_folder=env.train_labels_folder,
        image_transforms=config.train_transforms,
        image_filenames=train_filenames,
        has_masks=True,
    )
    val_dataset = SegmentationDataset(
        images_root_folder=env.val_images_folder,
        masks_root_folder=env.val_labels_folder,
        image_transforms=config.val_transforms,
        image_filenames=val_filenames,
        has_masks=True,
    )
    fullsize_gt_dataset = SegmentationDataset(
        images_root_folder=env.val_images_folder,
        masks_root_folder=env.val_labels_folder,
        image_transforms=config.test_transforms,
        image_filenames=val_filenames,
        has_masks=True,
    )

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset, shuffle=False)

    loss_function = RetinaSegLoss()
    optimizer = TorchOptimizers.AdamW(model.parameters(), lr=config.starting_learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    scheduler = None

    train_metrics = MetricsAccumulator()
    val_metrics = MetricsAccumulator()

    best_fullsize_global_mean_iou = float('-inf')
    best_epoch = -1

    history = {
        'train_loss': [], 'train_accuracy': [], 'train_vessel_iou': [], 'train_bg_iou': [], 'train_mean_iou': [],
        'val_loss': [], 'val_accuracy': [], 'val_vessel_iou': [], 'val_bg_iou': [], 'val_mean_iou': [],
        'fullsize_global_accuracy': [], 'fullsize_per_image_accuracy': [],
        'fullsize_global_vessel_iou': [], 'fullsize_global_bg_iou': [], 'fullsize_global_mean_iou': [],
        'fullsize_per_image_vessel_iou': [], 'fullsize_per_image_bg_iou': [], 'fullsize_per_image_mean_iou': [],
        'best': {},
    }

    training_start_time = time.time()
    print(f't={training_start_time - start_time:.2f}: Started training')
    print(f'Config: {config}')

    torch.manual_seed(config.seed)

    epochs_since_best = 0

    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            epoch_start_time = time.time()
            print(f't={epoch_start_time - start_time:.2f}: Starting epoch {epoch}/{config.max_epochs}. Early stopping in {config.patience - epochs_since_best} epochs.')

            train_loss, train_m = train_one_epoch(epoch_start_time, model, train_loader, optimizer, loss_function, scaler, scheduler, train_metrics)

            if env.device == 'cuda':
                torch.cuda.empty_cache()

            val_loss, val_m, pred_masks = validate_one_epoch(epoch_start_time, model, val_loader, loss_function, val_metrics)

            fullsize_m = fullsize_validate(pred_masks, fullsize_gt_dataset)

            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_m['accuracy'])
            history['train_vessel_iou'].append(train_m['vessel_iou'])
            history['train_bg_iou'].append(train_m['bg_iou'])
            history['train_mean_iou'].append(train_m['mean_iou'])

            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_m['accuracy'])
            history['val_vessel_iou'].append(val_m['vessel_iou'])
            history['val_bg_iou'].append(val_m['bg_iou'])
            history['val_mean_iou'].append(val_m['mean_iou'])

            for key in fullsize_m:
                history[f'fullsize_{key}'].append(fullsize_m[key])

            print(f'================ Epoch {epoch:03d} stats ==================')
            print(f'train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}')
            print(f'train_acc: {train_m["accuracy"]:.4f}  val_acc: {val_m["accuracy"]:.4f}')
            print(f'train_vessel_iou: {train_m["vessel_iou"]:.4f}  val_vessel_iou: {val_m["vessel_iou"]:.4f}')
            print(f'train_bg_iou: {train_m["bg_iou"]:.4f}  val_bg_iou: {val_m["bg_iou"]:.4f}')
            print(f'train_mean_iou: {train_m["mean_iou"]:.4f}  val_mean_iou: {val_m["mean_iou"]:.4f}')
            print(f'fullsize_global_mean_iou: {fullsize_m["global_mean_iou"]:.4f}  fullsize_per_image_mean_iou: {fullsize_m["per_image_mean_iou"]:.4f}')
            print('===================================================')

            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_m['accuracy'],
                'train_vessel_iou': train_m['vessel_iou'],
                'train_bg_iou': train_m['bg_iou'],
                'train_mean_iou': train_m['mean_iou'],
                'val_loss': val_loss,
                'val_accuracy': val_m['accuracy'],
                'val_vessel_iou': val_m['vessel_iou'],
                'val_bg_iou': val_m['bg_iou'],
                'val_mean_iou': val_m['mean_iou'],
                'fullsize_global_accuracy': fullsize_m['global_accuracy'],
                'fullsize_per_image_accuracy': fullsize_m['per_image_accuracy'],
                'fullsize_global_vessel_iou': fullsize_m['global_vessel_iou'],
                'fullsize_global_bg_iou': fullsize_m['global_bg_iou'],
                'fullsize_global_mean_iou': fullsize_m['global_mean_iou'],
                'fullsize_per_image_vessel_iou': fullsize_m['per_image_vessel_iou'],
                'fullsize_per_image_bg_iou': fullsize_m['per_image_bg_iou'],
                'fullsize_per_image_mean_iou': fullsize_m['per_image_mean_iou'],
            })

            if fullsize_m['global_mean_iou'] > best_fullsize_global_mean_iou:
                best_fullsize_global_mean_iou = fullsize_m['global_mean_iou']
                best_epoch = epoch
                epochs_since_best = 0
                torch.save(model.model.state_dict(), env.saved_weights_filepath)
            else:
                epochs_since_best += 1
                if epochs_since_best >= config.patience:
                    wandb.run.summary['early_stopping_triggered'] = True
                    break

    except KeyboardInterrupt:
        print(f't={time.time() - start_time:.2f}: Training manually interrupted')
        wandb.run.summary['training_manually_interrupted'] = True

    finally:
        history['best']['fullsize_global_mean_iou'] = best_fullsize_global_mean_iou
        history['best']['epoch'] = best_epoch

        print()
        print('==================== Results ======================')
        print(f'Best fullsize global mean IoU: {best_fullsize_global_mean_iou:.4f}')
        print(f'Best epoch: {best_epoch}')
        print('===================================================')
        print()

        wandb.run.summary['best_fullsize_global_mean_iou'] = best_fullsize_global_mean_iou
        wandb.run.summary['best_epoch'] = best_epoch

        _plot_training_metrics(history)

        with open(env.training_output_folder + 'history.json', 'w') as json_file:
            json.dump(history, json_file, indent=4)

        wandb.save(env.saved_weights_filepath)
        wandb.save(env.training_output_folder + 'history.json')

        wandb.finish()


@torch.no_grad()
def run_val_set_inference():
    start_time = time.time()
    print('t=0: Starting validation set inference')

    val_filenames = env.fetch_val_filenames()
    val_dataset = SegmentationDataset(
        images_root_folder=env.val_images_folder,
        masks_root_folder=env.val_labels_folder,
        image_transforms=config.val_transforms,
        image_filenames=val_filenames,
        has_masks=True,
    )
    val_loader = create_dataloader(val_dataset, shuffle=False)

    model = RetinaSegModel(saved_model_weights=env.saved_weights_filepath)
    model.eval()

    pred_masks = []

    num_batches = _num_batches(val_loader)
    for batch_number, (x, y) in enumerate(val_loader):
        print(f't={time.time() - start_time:.2f}: Loading validation batch {batch_number + 1}/{num_batches}')

        x = x.to(env.device, non_blocking=True)
        y = y.to(env.device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=config.use_amp):
            logits = model(x)
            # logits shape: (batch, 1, H, W) → squeeze channel → (batch, H, W)
            batch_masks = (logits[:, 0] > 0).long()
            pred_masks.extend(batch_masks)

    print(f't={time.time() - start_time:.2f}: Inference complete, returning pred_masks')
    return pred_masks

def visualize_pred_masks(pred_masks, num_to_visualize=3):
    val_filenames = env.fetch_val_filenames()
    val_dataset = SegmentationDataset(
        images_root_folder=env.val_images_folder,
        masks_root_folder=env.val_labels_folder,
        image_transforms=config.test_transforms,
        image_filenames=val_filenames,
        has_masks=True,
    )

    indices = random.sample(range(len(val_dataset)), num_to_visualize)

    fig, axes = plt.subplots(num_to_visualize, 3, figsize=(15, 5 * num_to_visualize))
    axes[0, 0].set_title('Image')
    axes[0, 1].set_title('Predicted Mask')
    axes[0, 2].set_title('Overlay')

    for row, i in enumerate(indices):
        image_tensor, _ = val_dataset[i]
        image = gpu_image_tensor_to_numpy_array(image_tensor)

        pred_mask = pred_masks[i].cpu().numpy() if isinstance(pred_masks[i], torch.Tensor) else np.array(pred_masks[i])
        if pred_mask.shape != (image.shape[0], image.shape[1]):
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        colored_mask = CLASS_COLORS[(pred_mask > 0).astype(np.int32)]
        overlay = (0.5 * colored_mask + 0.5 * image).astype(np.uint8)

        axes[row, 0].imshow(image)
        axes[row, 0].axis('off')
        axes[row, 1].imshow(colored_mask)
        axes[row, 1].axis('off')
        axes[row, 2].imshow(overlay)
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    global env, config
    env = local_env
    config = Config(training=False)
    plot_val_images_and_masks()
    pred_masks = run_val_set_inference()
    global_mIoU = val_mean_iou(pred_masks)
    visualize_pred_masks(pred_masks)

if __name__ == "__main__":
    main()