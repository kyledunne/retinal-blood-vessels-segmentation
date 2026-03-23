from dataclasses import dataclass
import numpy as np
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

    def fetch_train_filenames(self):
        return np.array([f.name for f in Path(self.train_images_folder).iterdir()])

    def fetch_val_filenames(self):
        return np.array([f.name for f in Path(self.val_images_folder).iterdir()])


env: Environment = None
""" Set to appropriate environment before training/inference """


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