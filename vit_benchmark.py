"""Train RPQViT model on Imagenet dataset using transformers library"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from vit_pytorch import ViT
from rpq.models.rpqvit import RPQViT
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
parser.add_argument('--opt', default='adamw', type=str, help='optimizer')
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--epochs', default=90, type=int, help='number of epochs')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--data-dir', default='data', type=str, help='data directory')
parser.add_argument('--save-dir', default='checkpoints', type=str, help='save directory')
parser.add_argument('--model', default='vit', type=str, help='model to train')
parser.add_argument('--no-pretrained', action='store_true', help='disable pretrained weights')
parser.add_argument('--no-resume', action='store_true', help='disable resume training')
parser.add_argument('--no-save', action='store_true', help='disable saving checkpoints')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seed
torch.manual_seed(args.seed)

# Data loading code
print('==> Preparing data..')
traindir = os.path.join(args.data_dir, 'train')
valdir = os.path.join(args.data_dir, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]),
)

test_dataset = ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
)

# Model
print('==> Building model..')
if args.model == 'vit':
        model = ViT(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0.1,
        )
elif args.model == 'rpqvit':
    model = RPQViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
    )

model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
if args.opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
elif args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.opt == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# Resume training
if not args.no_resume:
    if os.path.isfile('checkpoint.pth'):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('checkpoint.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        print('==> No checkpoint found..')
        start_epoch = 0
else:
    start_epoch = 0

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(total=len(train_loader), unit='batch') as t:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_postfix(loss=(train_loss / (batch_idx + 1)), acc=(100. * correct / total))
            t.update()

# Testing
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with tqdm(total=len(test_loader), unit='batch') as t:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                t.set_postfix(loss=(test_loss / (batch_idx + 1)), acc=(100. * correct / total))
                t.update()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if not args.no_save:
            torch.save(state, 'checkpoint.pth')
        best_acc = acc

best_acc = 0
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
