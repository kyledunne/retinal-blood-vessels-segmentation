from torch.utils.data import Dataset as TorchDataset
import cv2
import numpy as np
from pathlib import Path
import torch

from data import env

class ValDataset(TorchDataset):
    def __init__(self):
        self.val_filenames = env.fetch_val_filenames()

    def __len__(self):
        return len(self.val_filenames)

    def __getitem__(self, idx):
        image_filename = self.val_filenames[idx]
        image = cv2.cvtColor(cv2.imread(f'{env.val_images_folder}{image_filename}'), cv2.COLOR_BGR2RGB)
        mask_stem = Path(image_filename).stem
        mask_path = next(Path(env.val_labels_folder).glob(f'{mask_stem}.*'))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return image, mask


def val_iou(pred_masks):
    val_filenames = env.fetch_val_filenames()

    # Global accumulators for 2-class IoU (background=0, vessel=1)
    intersection = np.zeros(2, dtype=np.int64)
    union = np.zeros(2, dtype=np.int64)

    for i, image_filename in enumerate(val_filenames):
        mask_stem = Path(image_filename).stem
        mask_path = next(Path(env.val_labels_folder).glob(f'{mask_stem}.*'))
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_masks[i] > 0).astype(np.uint8)

        # Vessel class (1)
        intersection[1] += np.sum((pred_binary == 1) & (gt_binary == 1))
        union[1] += np.sum((pred_binary == 1) | (gt_binary == 1))

        # Background class (0)
        intersection[0] += np.sum((pred_binary == 0) & (gt_binary == 0))
        union[0] += np.sum((pred_binary == 0) | (gt_binary == 0))

    iou_per_class = intersection / (union + 1e-6)
    miou = float(np.mean(iou_per_class))
    return miou
