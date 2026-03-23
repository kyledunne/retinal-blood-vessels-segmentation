from torch.utils.data import Dataset as TorchDataset
import cv2
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


def calculate_val_iou(pred_masks):

