import torchvision.models as models
import torchvision
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torch.optim import Adam

from tqdm import tqdm

import pandas as pd
import numpy as np 
from PIL import Image
import os

import config

class ResNet_MTL(nn.Module):
    def __init__(self, num_class_1, num_class_2):
        super().__init__()

        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        in_ftrs = self.backbone.fc.in_features

        # remove classification layer
        self.backbone.fc = nn.Identity

        # classification layer for dataset 1
        self.head_bce = nn.Linear(in_ftrs, num_class_1)
        # classification layer for dataset 2
        self.head_ce = nn.Linear(in_ftrs, num_class_2)
    
    def forward(self, x, task):
        features = self.backbone(x)

        if task == 'bce':
            out = self.head_bce(features)
        elif task == 'ce':
            out = self.head_ce(features)
        else:
            raise ValueError("Unknown task Function")
        
        return out

# getter for the train/val loaders for the cats and dogs dataset
def catsVsDogs_Dataset(tranformations):
    
    data_dir = "/scratch/gssodhi/catNdogs/PetImages"  # path to folder containing cats and dogs

    train_transform = tranformations['train']
    val_transform = tranformations['val']

    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_transform  # temporary
    )

    class_names = full_dataset.classes
    print(class_names)  # ['cats', 'dogs']

    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader


class ButterFly(Dataset):
    def __init__(self, csv_path, root_dir, split, transform=None):
        self.df = pd.read_csv(csv_path)

        self.df = self.df[self.df['data set'] == split].reset_index(drop=True)

        self.root_dir = root_dir
        self.transformation = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir, row['filepaths'])
        image = Image.open(img_path).convert("RGB")

        label = int(row['class id'])

        if self.transformation:
            image = self.transformation(image)

        return image, label



def train(model, loader, criterion, optimizer, device, task):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(loader, desc='train'):
        x, y = x.to(device), y.to(device)

        if task == 'bce':
            y = y.float().unsqueeze(1)

        optimizer.zero_grad()
        out = model(x, task)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def test(model, loader, criterion, device, task):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc='val'):
            x, y = x.to(device), y.to(device)

            if task == 'bce':
                y = y.float().unsqueeze(1)

            out = model(x, task)
            loss = criterion(out, y)
            running_loss += loss.item()

    return running_loss / len(loader)



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tranformations = {
            'train': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ]),

            'val': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
        }

    # Get dataset 1
    train_loader1, val_loader1 = catsVsDogs_Dataset(tranformations)

    # get dataset 2 csv_path, root_dir, split, transform=None
    csv_path = '/scratch/gssodhi/butterfly/splits.csv'
    root_dir = '/scratch/gssodhi/butterfly'

    train_dataset_2 = ButterFly(csv_path, root_dir, split='train', transform=tranformations['train'])
    val_dataset_2 = ButterFly(csv_path, root_dir, split='valid', transform=tranformations['val'])

    train_loader_2 = DataLoader(
        train_dataset_2,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader_2 = DataLoader(
        val_dataset_2,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )


    # load model
    myModel = ResNet_MTL(
                        num_class_1 = 1, 
                        num_class_2 = 100
                    )

    myModel.to(device)

    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.CrossEntropyLoss()

    optimizer = Adam(myModel.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_loss_1, val_loss_1 = [], []
    train_loss_2, val_loss_2 = [], []
    for epoch in range(config.epochs):
        # Task 1: Cats vs Dogs
        train_loss_1 = train(
            myModel, train_loader1, criterion1, optimizer, device, task='bce'
        )
        val_loss_1 = test(
            myModel, val_loader1, criterion1, device, task='bce'
        )

        # Task 2: Butterflies
        train_loss_2 = train(
            myModel, train_loader_2, criterion2, optimizer, device, task='ce'
        )
        val_loss_2 = test(
            myModel, val_loader_2, criterion2, device, task='ce'
        )

        print(f"Epoch {epoch}: "
            f"BCE train {train_loss_1:.4f} val {val_loss_1:.4f} | "
            f"CE train {train_loss_2:.4f} val {val_loss_2:.4f}")
