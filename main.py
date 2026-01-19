import torchvision.models as models
import torchvision
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from tqdm import tqdm

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
def CatsVsDogs_Dataset(tranformations):
    
    data_dir = "/scratch/gssodhi/catNdogs/PetImages"  # path to folder containing cats/ and dogs/

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

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform


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

def train(model, loader1, loader2, criterion1, criterion2, optimizer, device):
    model.train()
    running_loss_1 = 0.0
    running_loss_2 = 0.0


    for (x1, y1), (x2, y2) in tqdm(zip(loader1, loader2), desc='train'):
        x1, y1, x2, y2 = x1.to(device), y1.to(device), x2.to(device), y2.to(device)

        optimizer.zero_grad()

        # forward prop for dataset 1
        out_1 = model(x1, 'bce')
        loss_1 = criterion1(out_1, y1)
        # forward prop for dataset 2
        out_2 = model(x2, 'ce')
        loss_2 = criterion2(out_2, y2)

        total_loss = loss_1 + loss_2

        # backprop and take a step
        avg_loss.backward()
        optimizer.step()

        running_loss_1 += loss_1.item()
        running_loss_2 += loss_2.item()
    
    avg_loss_1 = running_loss_1 / len(loader1)
    avg_loss_2 = running_loss_2 / len(loader2)

    return avg_loss_1, avg_loss_2

def test(model, loader1, loader2, criterion1, criterion2, device):
    model.eval()
    running_loss_1 = 0.0
    running_loss_2 = 0.0

    with torch.no_grad():
        for (x1, y1), (x2, y2) in tqdm(zip(loader1, loader2), desc='val'):
                x1, y1, x2, y2 = x1.to(device), y1.to(device), x2.to(device), y2.to(device)

                # forward prop for dataset 1
                out_1 = model(x1, 'bce')
                loss_1 = criterion1(out_1, y1)
                # forward prop for dataset 2
                out_2 = model(x2, 'ce')
                loss_2 = criterion2(out_2, y2)

                total_loss = loss_1 + loss_2

                running_loss_1 += loss_1.item()
                running_loss_2 += loss_2.item()
        
    avg_loss_1 = running_loss_1 / len(loader1)
    avg_loss_2 = running_loss_2 / len(loader2)

    return avg_loss_1, avg_loss_2


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

    train_loader1, val_loader1 = CatsVsDogs_Dataset(tranformations)

    # load model
    myModel = ResNet_MTL(
                        num_classes_1 = 1, 
                        num_class_2 = 4
                    )

    myModel.to(device)

    criterion1 = nn.BCEWithLogitLoss()
    criterion1 = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters, lr=config.lr, weight_decay=config.weight_decay)

    train_loss_1, val_loss_1 = [], []
    train_loss_2, val_loss_2 = [], []
    for epoch in config.epochs:

        train_loss_1, train_loss_2 = train(myModel, 
                                        train_loader1,
                                        train_loader1,
                                        criterion1,
                                        criterion2,
                                        optimizer,
                                        device)

        val_loss_1, val_loss_2 = test(myModel, 
                                        val_loader1,
                                        val_loader1,
                                        criterion1,
                                        criterion2,
                                        device)

        print(f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f} | "
            f"F1: {f1:.4f} | AUC: {auc:.4f}")




