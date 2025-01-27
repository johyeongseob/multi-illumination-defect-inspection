import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
import imgaug.augmenters as iaa
import torchvision.transforms.functional as TF


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)['image']
        return Image.fromarray(augmented)


class ImgaugTransform:
    def __init__(self, augmentation):
        self.augmentation = augmentation

    def __call__(self, img):
        img_np = np.array(img)
        augmented_img_np = self.augmentation(image=img_np)
        return Image.fromarray(augmented_img_np)

horizon = transforms.RandomHorizontalFlip(p=1)
vertical = transforms.RandomVerticalFlip(p=1)
rotation = transforms.RandomRotation(degrees=(45, 45))
contrast = transforms.ColorJitter(contrast=(1.5, 1.5))
bright = transforms.ColorJitter(brightness=(1.5, 1.5))
elastic = AlbumentationsTransform(A.ElasticTransform(alpha=5, sigma=20, p=1))
shifting = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
gaussnoise = AlbumentationsTransform(A.GaussNoise(var_limit=(200.0, 200.0), p=1.0))
gaussianblur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 0.5))
cropresize = transforms.Compose([transforms.CenterCrop(size=(80, 80)), transforms.Resize(size=(100, 100))])
paddingresize = transforms.Compose([transforms.Pad(padding=10), transforms.Resize(size=(100, 100))])
optical = AlbumentationsTransform(A.OpticalDistortion(distort_limit=0.5, shift_limit=0.00, p=1))
dropout = ImgaugTransform(iaa.CoarseDropout(p=0.1, size_percent=0.3))
# shift = transforms.Lambda(lambda img: TF.affine(img, angle=0, translate=(2, 2), scale=1, shear=0))


# Custom Transform과 ToTensor 변환 적용
custom_transform = transforms.Compose([
    rotation,  # Albumentations 변환을 Custom Transform으로 적용
    transforms.ToTensor()  # Tensor로 변환
])

# MNIST 데이터셋 다운로드 및 불러오기
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=custom_transform)

# DataLoader 설정
train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=64, shuffle=False)

# 첫 번째 배치의 이미지와 라벨 가져오기
images, labels = next(iter(train_loader))

# 첫 번째 이미지를 가져오기
first_image = images[32].squeeze()
first_label = labels[32]

# 이미지 시각화
plt.imshow(first_image, cmap='gray')
plt.title(f'Label: {first_label}')
plt.show()
