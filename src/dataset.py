import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import os

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, Blur, RandomRotate90
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

def parse_carbon(inchi_text):
    return inchi_text.split('/c')[1].strip().split('/h')[0].strip()

def parse_hydrogen(inchi_text):

    details = inchi_text.split('/c')[1].strip().split('/h')
    if len(details)==2:
        return details[1].strip()
    else:
        return ''

class TrainDatasetWithMultiLabel(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.labels = df['InChI_text'].values
        self.transform = transform
        self.molecular_labels = df['InChI_text'].map(lambda x: x.split('/c')[0].strip()).values
        self.carbon_labels = df['InChI_text'].map(parse_carbon).values
        self.hydrogen_labels = df['InChI_text'].map(parse_hydrogen).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        one_c = image[:,:,0]
        kernel = np.ones((3,3),np.uint8)
        one_c = cv2.erode(one_c,kernel,iterations=3)
        image[:,:,1] = one_c
        one_c = cv2.dilate(one_c,kernel,iterations=1)
        image[:,:,2] = one_c
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label = torch.LongTensor(label)
        label_length = torch.LongTensor([len(label)])

        molecular_label = self.molecular_labels[idx]
        molecular_label = self.tokenizer.text_to_sequence(molecular_label)
        molecular_label = torch.LongTensor(molecular_label)
        molecular_label_length = torch.LongTensor([len(molecular_label)])

        carbon_label = self.carbon_labels[idx]
        carbon_label = self.tokenizer.text_to_sequence(carbon_label)
        carbon_label = torch.LongTensor(carbon_label)
        carbon_label_length = torch.LongTensor([len(carbon_label)])

        hydrogen_label = self.hydrogen_labels[idx]
        hydrogen_label = self.tokenizer.text_to_sequence(hydrogen_label)
        hydrogen_label = torch.LongTensor(hydrogen_label)
        hydrogen_label_length = torch.LongTensor([len(hydrogen_label)])

        return image, label, label_length, molecular_label, molecular_label_length, carbon_label, carbon_label_length, hydrogen_label, hydrogen_label_length

class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, multi_label=False):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.labels = df['InChI_text'].values
        self.multi_label = multi_label
        self.transform = transform
        if self.multi_label:
            self.molecular_label = df['InChI_text'].map(lambda x: x.split('/c')[0].strip()).values
            self.carbon_label = df['InChI_text'].map(parse_carbon).values
            self.hydrogen_label = df['InChI_text'].map(parse_hydrogen).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return image, torch.LongTensor(label), label_length


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = transform
        self.fix_transform = Compose([Transpose(p=1), VerticalFlip(p=1)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, _ = image.shape
        if h > w:
            image = self.fix_transform(image=image)['image']
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image