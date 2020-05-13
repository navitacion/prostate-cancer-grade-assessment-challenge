import os
import glob
import cv2
import pandas as pd
from PIL import Image

from utils.image_transform import ImageTransform
from utils.dataset import PANDADataset

data_dir = '../data/grid_224'
img_path = glob.glob(os.path.join(data_dir, '*.jpg'))
score_df = pd.read_csv(os.path.join(data_dir, 'res.csv'))

print(type(img_path))

transform = ImageTransform()
dataset = PANDADataset(img_path, score_df, transform, phase='train')

img, label = dataset.__getitem__(5)

print(img.size())
print(img.dtype)
print(img)
print(label.size())
print(label.dtype)
print(label)


