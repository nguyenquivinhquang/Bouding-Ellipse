import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from math import pi
def handle_input(label_path):
    f = open(label_path, 'r')
    labels = []
    lines = f.readlines()
    for line in lines:
        line = line.split(" ")
    
    return np.array(labels)
def transform_label(label):
    label[0] = (label[0]-112)/224
    label[1] = (label[1]-112)/224
    label[2] /= 224
    label[3] /=224
    label[4] / pi
    return label
    
class CustomImageDataset(Dataset):
    def __init__(self, img_dir,label_dir,  transform=None):
        self.img_labels = pd.read_csv(label_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        row = self.img_labels.iloc[idx]
        label = [row[1], row[2],row[3],row[4],row[5]]
        label = transform_label(label)
        label = torch.FloatTensor(label)
        if self.transform:
            image = self.transform(image)
        return image, label
if __name__ == "__main__":
    label_path = 'dataset/train_label.csv'
    img_path = '/mnt/Code/python/Bouding-Ellipse/dataset/img/'

    transform_train_list = [
        transforms.ToPILImage(),
        transforms.Resize((224,224), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    data_transforms = transforms.Compose(transform_train_list)
    
    training_data  = CustomImageDataset(img_dir=img_path, label_dir=label_path, transform=data_transforms)
    image, label = training_data[1]
    print(image.shape, label)
    
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True, num_workers=2)
    for i, batch in enumerate(train_dataloader):
        print(i, batch[0].shape, batch[1])
        break
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")

    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[1]
    # print(label.shape)
    # for t in  label: print(t)
