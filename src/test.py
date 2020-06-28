import numpy as np
import matplotlib.pyplot as plt
import glob, cv2
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

z = np.random.randint(0, 256, (224, 224, 3))

print(z.shape)
print(z)


transform = albu.Compose([ToTensorV2()])

z_out = transform(image=z)['image']
print(z_out)
z_out = z_out / 255.
print(z_out)


img_path = glob.glob('../data/grid_256_level_1/img/*.jpg')

print(img_path[0])

_img = cv2.imread(img_path[0])
_img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

_img = 255 - _img

_img = _img / 255


