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

@dataclass
class Environment:
    train_images_folder: str
    train_labels_folder: str
    val_images_folder: str
    val_labels_folder: str
    saved_weights_filepath: str
    training_output_folder: str
    device: str

    def fetch_train_ids(self):
        return np.array([f.stem for f in Path(self.train_images_folder).iterdir()])

    def fetch_val_ids(self):
        return np.array([f.stem for f in Path(self.val_images_folder).iterdir()])


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

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.05,
                rotate_limit=10,
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

        self.train_transforms_2 = A.Compose([
            A.HorizontalFlip(p=0.5),

            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-10, 10),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7,
            ),

            # Mild crop/resize jitter around full resolution
            A.OneOf([
                A.RandomResizedCrop(
                    size=(576, 768),
                    scale=(0.90, 1.00),
                    ratio=(1.30, 1.37),  # around 768/576 = 1.333
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.Resize(576, 768, interpolation=cv2.INTER_LINEAR, p=1.0),
            ], p=0.5),

            A.OneOf([
                A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=1.0,
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=1.0),
            ], p=0.6),

            A.GaussNoise(std_range=(0.01, 0.04), mean_range=(0.0, 0.0), p=0.2),

            A.Normalize(mean=imagenet_mean_tuple, std=imagenet_std_tuple),
            A.ToTensorV2(),
        ])

        self.train_transforms_3 = A.Compose([
            # Spatial / geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, border_mode=cv2.BORDER_CONSTANT, p=0.2),

            # Color / intensity (image-only, masks unaffected)
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),

            # Normalize + to tensor
            A.Normalize(mean=imagenet_mean_tuple, std=imagenet_std_tuple),
            A.ToTensorV2(),
        ])

        self.val_transforms = A.Compose([
            A.Normalize(mean=imagenet_mean_tuple, std=imagenet_std_tuple),
            A.ToTensorV2(),
        ])


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


class RetinaSegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.loss(logits, targets)


def train_one_epoch(start_time, model, loader, optimizer, loss_function, scaler, scheduler):
    model.train()
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
            visualize_mask_overlayed_over_image(x[0], y[0])

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=config.use_amp):
            logits = model(x)
            if config.verbose or batch_number == 0:
                logits_0_mask = logits[0].argmax(dim=0)
                print(f'First image with overlayed prediction mask:')
                visualize_mask_overlayed_over_image(x[0], logits_0_mask)
            loss = loss_function(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()

    epoch_loss = running_loss / num_batches
    return epoch_loss


@torch.no_grad()
def validate_one_epoch(start_time, model, loader, loss_function):
    model.eval()
    running_loss = 0.0

    num_batches = _num_batches(loader)
    for batch_number, (x, y) in enumerate(loader):
        print(f'v={time.time() - start_time:.2f}: Loading validation batch {batch_number + 1}/{num_batches}')

        x = x.to(env.device, non_blocking=True)
        y = y.to(env.device, non_blocking=True)

        if config.verbose or batch_number == 0:
            print(f'First image with overlayed ground-truth mask:')
            visualize_mask_overlayed_over_image(x[0], y[0])

        with torch.amp.autocast('cuda', enabled=config.use_amp):
            logits = model(x)
            if config.verbose or batch_number == 0:
                logits_0_mask = logits[0].argmax(dim=0)
                print(f'First image with overlayed prediction mask:')
                visualize_mask_overlayed_over_image(x[0], logits_0_mask)
            loss = loss_function(logits, y)

        running_loss += loss.item()

    epoch_loss = running_loss / num_batches
    return epoch_loss


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

    train_ids = env.fetch_train_ids()
    val_ids = env.fetch_val_ids()

    model = RetinaSegModel(saved_model_weights=saved_model_weights)

    wandb.watch(model, log='gradients', log_freq=100)

    train_dataset = SegmentationDataset(
        images_root_folder=env.train_images_folder,
        masks_root_folder=env.train_labels_folder,
        image_suffix='.tif',
        mask_suffix='.tif',
        image_transforms=config.train_transforms,
        image_ids=train_ids,
        has_masks=True,
    )
    val_dataset = SegmentationDataset(
        images_root_folder=env.val_images_folder,
        masks_root_folder=env.val_labels_folder,
        image_suffix='.tif',
        mask_suffix='.tif',
        image_transforms=config.val_transforms,
        image_ids=val_ids,
        has_masks=True,
    )

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset, shuffle=False)

    loss_function = RetinaSegLoss()
    optimizer = TorchOptimizers.AdamW(model.parameters(), lr=config.starting_learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    scheduler = None

    best_val_iou = float('-inf')
    best_val_iou_epoch = -1

    history = dict(
        train_loss=[], val_loss=[], train_iou=[], val_iou=[],
        best_val_iou=dict(),
    )

    training_start_time = time.time()
    print(f't={training_start_time - start_time:.2f}: Started training')
    print(f'Config: {config}')

    torch.manual_seed(config.seed)

    best_weights_path = env.training_output_folder + 'best.pt'

    epochs_since_best = 0

    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            epoch_start_time = time.time()
            print(f't={epoch_start_time - start_time:.2f}: Starting epoch {epoch}/{config.max_epochs}. Early stopping in {config.patience - epochs_since_best} epochs.')

            train_loss, train_iou = train_one_epoch(epoch_start_time, model, train_loader, optimizer, loss_function, scaler, scheduler)

            if env.device == 'cuda':
                torch.cuda.empty_cache()

            val_loss, val_iou = validate_one_epoch(epoch_start_time, model, val_loader, loss_function)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_iou'].append(train_iou)
            history['val_iou'].append(val_iou)

            print(f'================ Epoch {epoch:03d} stats ==================')
            print(f'train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}')
            print(f'train_iou: {train_iou:.4f}  val_iou: {val_iou:.4f}')
            print('===================================================')

            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_iou': train_iou,
                'val_iou': val_iou,
            })

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_val_iou_epoch = epoch
                epochs_since_best = 0
                torch.save(model.state_dict(), best_weights_path)
            else:
                epochs_since_best += 1
                if epochs_since_best >= config.patience:
                    wandb.run.summary['early_stopping_triggered'] = True
                    break

    except KeyboardInterrupt:
        print(f't={time.time() - start_time:.2f}: Training manually interrupted')
        wandb.run.summary['training_manually_interrupted'] = True

    finally:
        history['best_val_iou']['val_iou'] = best_val_iou
        history['best_val_iou']['epoch'] = best_val_iou_epoch

        print()
        print('==================== Results ======================')
        print(f'Best val iou: {best_val_iou:.2f}')
        print(f'Best val iou epoch: {best_val_iou_epoch}')
        print('===================================================')
        print()

        wandb.run.summary['best_val_iou'] = best_val_iou
        wandb.run.summary['best_val_iou_epoch'] = best_val_iou_epoch

        train_ious = history['train_iou']
        val_ious = history['val_iou']

        epochs = list(range(1, len(train_ious) + 1))

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_ious, label='Train mean IoU', marker='o')
        plt.plot(epochs, val_ious, label='Val mean IoU', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('Training and Validation mean IoU per Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        wandb.log({'iou_plot': wandb.Image(plt.gcf())})

        plt.show()
        plt.close()

        with open(env.training_output_folder + 'history.json', 'w') as json_file:
            json.dump(history, json_file, indent=4)

        wandb.save(best_weights_path)
        wandb.save(env.training_output_folder + 'history.json')

        wandb.finish()


def main():
    global env, config
    env = local_env
    config = Config(training=True)
    train()

if __name__ == '__main__':
    main()