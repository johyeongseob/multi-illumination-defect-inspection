"""

1st ~ 14th view dataset loader

Use 7 views(1st, 2nd, 3rd, 4th, 5th, 8th, 11th)
Class: NG1(inner dent), NG2(inner scratch), NG3(outer scratch),
       NG1_like_OK(normal), NG2_like_OK(normal), NG3_like_OK(normal), OK(normal)

└─data_set
        └─train       └─val       └─test
            ├─NG1        ├─NG1       ├─NG1
            │  ├─1       ├─NG2       ├─NG2
            │  ├─2       ├─NG3       ├─NG3
            │  ├─…       └─OK        └─OK
            │  ├─14
            ├─NG2
            ├─NG3
            └─OK

"""


from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import random


class MVDataset(Dataset):
    """
    Multi-View Dataset for classification.

    Args:
        base_dir (str): Base directory containing dataset folders.
        augmentation (bool): Whether to apply data augmentation.
        target (str): Specifies the target model and class mapping.
    """

    def __init__(self, base_dir, target=None, augmentation=None):
        self.base_dir = base_dir
        self.augmentation = augmentation
        self.data = []
        self.indices = {}
        self.views = ['1', '2', '3', '4', '8', '11']
        self.class_map, (mean, std) = MVDataset.target_value(target)
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        for cls, class_idx in self.class_map.items():
            single_view_dir = os.path.join(self.base_dir, cls, self.views[0])
            file_names = os.listdir(single_view_dir)
            for file_name in file_names:
                views_paths = []
                for view in self.views:
                    file_path = os.path.join(self.base_dir, cls, view, file_name)
                    views_paths.append(file_path)
                if len(views_paths) == len(self.views):
                    self.data.append((views_paths, class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = []
        image_paths, class_id = self.data[idx]
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = self.stochastic_augmentation(image)
            image = self.transform(image)
            images.append(image)

        return images, class_id

    @staticmethod
    def target_value(task):
        """
        Initialize class mapping, normalization stats, and number of classes based on the target.

        Args:
            target (str): Target model specifying the dataset class map and normalization values.

        Returns:
            tuple: (class_map (dict), (mean (list), std (list)))
        """
        target_settings = {
            'model1': (
                {'NG1': 0, 'NG1_like_OK': 0, 'NG2': 1, 'NG2_like_OK': 1, 'NG3': 2, 'NG3_like_OK': 2, 'OK1': 3,
                 'OK2': 3},
                ([0.2303, 0.2303, 0.2303], [0.3001, 0.3001, 0.3001]),
            ),
            'model1_1': (
                {'NG1': 0, 'NG1_like_OK': 0, 'NG2': 1, 'NG2_like_OK': 1, 'NG3': 2, 'NG3_like_OK': 2},
                ([0.2303, 0.2303, 0.2303], [0.3001, 0.3001, 0.3001]),
            ),
            'model2BC': (
                {'NG1': 0, 'NG2': 0, 'NG1_like_OK': 1, 'NG2_like_OK': 1},
                ([0.0032, 0.0032, 0.0032], [0.0368, 0.0368, 0.0368]),
            ),
            'model2MC': (
                {'NG1': 0, 'NG2': 1, 'NG1_like_OK': 2, 'NG2_like_OK': 2},
                ([0.0032, 0.0032, 0.0032], [0.0368, 0.0368, 0.0368]),
            ),
            'model3': (
                {'NG3': 0, 'NG3_like_OK': 1},
                ([0.2211, 0.2211, 0.2211], [0.3857, 0.3857, 0.3857]),
            ),
            'total': (
                {'NG1': 0, 'NG2': 1, 'NG3': 2, 'NG1_like_OK': 3, 'NG2_like_OK': 3, 'NG3_like_OK': 3, 'OK1': 3,
                 'OK2': 3},
                ([0.2303, 0.2303, 0.2303], [0.3001, 0.3001, 0.3001]),
            ),
            'test': (
                {'NG1': 0, 'NG2': 1, 'NG3': 2, 'OK': 3},
                ([0.2303, 0.2303, 0.2303], [0.3001, 0.3001, 0.3001]),
            ),
        }
        if task not in target_settings:
            raise ValueError(f"Unknown target: {task}")
        return target_settings[task]

    def stochastic_augmentation(self, image, probability=0.3):
        if not self.augmentation or random.random() > probability:
            return image
        random_int = random.choice([90, 180, 270])
        augmentation_list = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation(degrees=(random_int, random_int))
        ]
        aug = random.choice(augmentation_list)
        return aug(image)

if __name__ == '__main__':
    path = 'data_set/train'
    dataset = MVDataset(base_dir=path, target='test')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # DataLoader를 순회하면서 데이터 확인
    for batch in dataloader:
        break
