import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor





def get_img(text, nums=1, shape=(32,32,3)):
    img = np.random.randint(0, int(255 * 0.92), shape, dtype=np.uint8)
    # img = np.zeros(shape, dtype=np.uint8)
    img = cv2.putText(img, f"{text}", (3, shape[1] - 3), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return img

class FontNumberDataset(Dataset):
    def __init__(self, length=10000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = random.randint(0, 9)
        img = get_img(label)
        img = torch.Tensor(img)
        img = img.permute(2, 0, 1)
        return img, label

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = torch.Tensor(gray)
        # gray = gray.gray = gray.unsqueeze(0)
        return gray, label



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)


if __name__ == '__main__':

    print(net)
