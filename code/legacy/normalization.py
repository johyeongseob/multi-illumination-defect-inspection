"""

1st ~ 14th view dataset loader

Use 7 views(1st, 2nd, 3rd, 4th, 5th, 8th, 11th)
Class: NG1(inner dent), NG2(inner scratch), NG3(outer scratch), OK(normal)

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


import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

root_dir = '../data_set'
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])
dataset = ImageFolder(root=root_dir, transform=transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
data = next(iter(loader))
images = data[0]

mean = torch.mean(images, dim=(0,2,3))
std = torch.std(images, dim=(0,2,3))

print("mean:", mean)
print("std:", std)