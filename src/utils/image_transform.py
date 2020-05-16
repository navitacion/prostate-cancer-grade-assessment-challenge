from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2

class ImageTransform:

    def __init__(self, img_size):
        self.transform = {
            'train': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.HorizontalFlip(),
                albu.Normalize(),
                ToTensorV2(),
            ]),
            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(),
                ToTensorV2(),
            ])}

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        _img = augmented['image']

        return _img
