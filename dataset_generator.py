import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from skimage.color import rgba2rgb
from PIL import Image
# code adopted from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/Basics/custom_dataset
# author: Aladdin Persson
class CustomData(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label, img_path
