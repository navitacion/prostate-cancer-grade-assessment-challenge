from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2

class ImageTransform:

    def __init__(self, img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = {
            'train': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90(),
                albu.Normalize(mean, std),
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
