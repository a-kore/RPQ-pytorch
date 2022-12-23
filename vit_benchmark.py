"""Train ViT and RPQViT model variants on multiple datasets and logging tensorboard results."""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from vit_pytorch import ViT
from rpq.models.rpqvit import RPQViT
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
parser.add_argument('--opt', default='adamw', type=str, help='optimizer')
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--epochs', default=90, type=int, help='number of epochs')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--data-dir', default='data', type=str, help='data directory')
parser.add_argument('--save-dir', default='checkpoints', type=str, help='save directory')
parser.add_argument('--model', default='vit', type=str, help='model to train')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to train on')
parser.add_argument('--resume', default=0, type=int, help='resume training')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seed
torch.manual_seed(args.seed)

# Data loading code
print('==> Preparing data..')
if args.dataset == 'imagenet':
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')

    train_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    )

    test_dataset = ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
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
elif args.dataset == 'cifar10':
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True,
        transform= transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

elif args.dataset == 'cifar100':
    train_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True,
        transform= transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]),
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

elif args.dataset == 'mnist':
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, download=True,
        transform= transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()])
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()])
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
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
if args.resume == 1:
    if os.path.isfile('./checkpoint/ckpt.pth'):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/ckpt.pth')
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
    """Return train loss and train accuracy."""
    print('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    train_losses = []
    accs = []
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

            train_loss = loss.item()
            _, predicted = outputs.max(1)
            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()
            accs.append(correct/total)
            train_losses.append(train_loss)

            t.set_postfix(loss=np.mean(train_losses), acc=np.mean(accs))
            t.update()

    return np.mean(train_losses), np.mean(accs)

# Testing
def test(epoch):
    """Return test loss and test accuracy and save checkpoint if best accuracy."""
    global best_acc
    model.eval()
    test_loss = 0
    test_losses = []
    accs = []
    correct = 0
    total = 0
    with tqdm(total=len(test_loader), unit='batch') as t:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss = loss.item()
                test_losses.append(test_loss)
                _, predicted = outputs.max(1)
                total = targets.size(0)
                correct = predicted.eq(targets).sum().item()
                accs.append(correct/total)
                t.set_postfix(loss=np.mean(test_losses), acc=np.mean(accs))
                t.update()

    # Save checkpoint.
    acc = np.mean(accs)
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return np.mean(test_losses),  acc

best_acc = 0
writer = SummaryWriter(log_dir="logs")

for epoch in range(start_epoch, start_epoch+args.epochs):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    writer.add_scalars(args.model, {'train_loss': train_loss}, epoch)
    writer.add_scalars(args.model, {'train_acc': train_acc}, epoch)
    writer.add_scalars(args.model, {'test_loss': test_loss}, epoch)
    writer.add_scalars(args.model, {'test_acc': test_acc}, epoch)
    
writer.close()
