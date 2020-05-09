from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2

class ImageTransform:

    def __init__(self):
        self.transform = {
            'train': albu.Compose([
                albu.HorizontalFlip(),
                albu.Normalize(),
                ToTensorV2(),
            ]),
            'val': albu.Compose([
                albu.Normalize(),
                ToTensorV2(),
            ])}

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        _img = augmented['image']

        return _img
