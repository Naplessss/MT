import os
import gc
import re
import math
import time
import random
import shutil
import pickle
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import Levenshtein
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from functools import partial
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, Blur, RandomRotate90
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import timm
import warnings
warnings.filterwarnings('ignore')

from utils import Tokenizer
from utils import get_train_file_path, get_score, init_logger, seed_torch
from dataset import TrainDataset, TestDataset, TrainDatasetWithMultiLabel
from attention import Attention, DecoderWithAttention
from utils import AverageMeter, asMinutes, timeSince
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import argparse
parser = argparse.ArgumentParser(description='CNN-LSTM')
parser.add_argument('--model_name', default='resnet34')
parser.add_argument('--meta_info', default='v0')
parser.add_argument('--size', default=224, type=int)
parser.add_argument('--batch_size_per_node', default=32, type=int)
parser.add_argument('--encoder_lr', default=1e-4, type=float)
parser.add_argument('--decoder_lr', default=4e-4, type=float)
parser.add_argument('--min_lr', default=1e-6, type=float)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--nodes', default=1, type=int)
parser.add_argument('--tmax', default=4, type=int)
parser.add_argument('--debug', default=0, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--local', default=0, type=int)
args = parser.parse_args()

if args.nodes > 1:
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

# 1) DDP init
torch.distributed.init_process_group(backend="nccl")
global_rank = torch.distributed.get_rank()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print('cur rank',local_rank,'global_rank', global_rank)


class CFG:
    debug=False if args.debug==0 else True
    max_len=275
    max_molecular_len=20
    max_carbon_len=150
    max_hydrogen_len=200
    print_freq=1000
    num_workers=6
    model_name=args.model_name
    meta_info=args.meta_info
    size=args.size
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs=args.epochs
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    T_max=args.tmax # CosineAnnealingLR
    #T_0=4 # CosineAnnealingWarmRestarts
    encoder_lr=args.encoder_lr
    decoder_lr=args.decoder_lr
    min_lr=args.min_lr
    batch_size=args.batch_size_per_node
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5
    seed=42
    n_fold=5
    trn_fold=[0]
    train=True

CFG = CFG()
train = pd.read_pickle('../train.pkl')
train['file_path'] = train['image_id'].apply(get_train_file_path)
if global_rank == 0:
    print(f'train.shape: {train.shape}')
tokenizer = torch.load('../tokenizer.pth')

if CFG.debug:
    CFG.epochs = 10
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)

if args.local:
    os.makedirs(f'../logs/{CFG.model_name}_{CFG.meta_info}', exist_ok=True)
    LOGGER = init_logger(log_file=f'../logs/{CFG.model_name}_{CFG.meta_info}/{CFG.model_name}_{CFG.meta_info}.log')
else:
    os.makedirs(f'/mnt/epblob/zhgao/MT/logs/{CFG.model_name}_{CFG.meta_info}', exist_ok=True)
    LOGGER = init_logger(log_file=f'/mnt/epblob/zhgao/MT/logs/{CFG.model_name}_{CFG.meta_info}/{CFG.model_name}_{CFG.meta_info}.log')
seed_torch(seed=CFG.seed)

folds = train.copy()
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['InChI_length'])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
if global_rank == 0:
    print(folds.groupby(['fold']).size())

def get_transforms(*, data):

    if data == 'train':
        return Compose([
            Resize(CFG.size, CFG.size),
            RandomRotate90(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            RandomRotate90(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

def bms_collate(batch):
    imgs, labels, label_lengths, molecular_labels, molecular_label_lengths, \
        carbon_labels, carbon_label_lengths, hydrogen_labels, hydrogen_label_lengths = [], [], [], [], [], [], [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
        molecular_labels.append(data_point[3])
        molecular_label_lengths.append(data_point[4])
        carbon_labels.append(data_point[5])
        carbon_label_lengths.append(data_point[6])
        hydrogen_labels.append(data_point[7])
        hydrogen_label_lengths.append(data_point[8])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    molecular_labels = pad_sequence(molecular_labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    carbon_labels = pad_sequence(carbon_labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    hydrogen_labels = pad_sequence(hydrogen_labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])

    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1),\
                            molecular_labels, torch.stack(molecular_label_lengths).reshape(-1, 1),\
                            carbon_labels, torch.stack(carbon_label_lengths).reshape(-1, 1),\
                            hydrogen_labels, torch.stack(hydrogen_label_lengths).reshape(-1, 1)

def bms_collate_valid(batch):
    imgs, labels, molecular_labels, carbon_labels,  hydrogen_labels = [], [], [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        molecular_labels.append(data_point[3])
        carbon_labels.append(data_point[5])
        hydrogen_labels.append(data_point[7])

    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    molecular_labels = pad_sequence(molecular_labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    carbon_labels = pad_sequence(carbon_labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    hydrogen_labels = pad_sequence(hydrogen_labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])

    return torch.stack(imgs), np.array(labels), np.array(molecular_labels), np.array(carbon_labels), np.array(hydrogen_labels)

def decoder_fn(decoder, features, labels, label_lengths, criterion):
    predictions, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, labels, label_lengths)
    targets = caps_sorted[:, 1:]
    predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True, enforce_sorted=False).data
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False).data
    loss = criterion(predictions, targets)
    return loss

def predict_fn(decoder, features, max_len, tokenizer):
    with torch.no_grad():
        if isinstance(decoder, DistributedDataParallel):
            predictions = decoder.module.predict(features, max_len, tokenizer)
        else:
            predictions = decoder.predict(features, max_len, tokenizer)
    _seq_preds = torch.argmax(predictions.detach().cpu(), -1).numpy()
    _text_preds = tokenizer.predict_captions(_seq_preds)

    return _text_preds

def merge_text_preds(molecular, carbon, hydrogen, add_prefix=False):
    text = ""
    if add_prefix:
        text += f"InChI=1S/"
    text += f"{molecular}/c{carbon}"
    if len(hydrogen)!=0:
        text += f"/h{hydrogen}"
    return text

def train_fn(train_loader,
             encoder,
             decoder_molecular_envs,
             decoder_carbon_envs,
             decoder_hydrogen_envs,
             criterion, epoch,
             encoder_scheduler,
             encoder_optimizer,
             device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    decoder_molecular, decoder_molecular_optimizer, decoder_molecular_scheduler = decoder_molecular_envs
    decoder_carbon, decoder_carbon_optimizer, decoder_carbon_scheduler = decoder_carbon_envs
    decoder_hydrogen, decoder_hydrogen_optimizer, decoder_hydrogen_scheduler = decoder_hydrogen_envs

    # switch to train mode
    encoder.train()
    decoder_molecular.train()
    decoder_carbon.train()
    decoder_hydrogen.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels, label_lengths, molecular_labels, molecular_label_lengths,
                carbon_labels, carbon_label_lengths, hydrogen_labels, hydrogen_label_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        batch_size = images.size(0)
        features = encoder(images)

        loss_molecular = decoder_fn(decoder_molecular, features, molecular_labels, molecular_label_lengths, criterion)
        loss_carbon = decoder_fn(decoder_carbon, features, carbon_labels, carbon_label_lengths, criterion)
        loss_hydrogen = decoder_fn(decoder_hydrogen, features, hydrogen_labels, hydrogen_label_lengths, criterion)
        loss = loss_molecular + loss_carbon + loss_hydrogen
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.max_grad_norm)
        for decoder in [decoder_molecular, decoder_carbon, decoder_hydrogen]:
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            encoder_optimizer.step()
            decoder_molecular_optimizer.step()
            decoder_carbon_optimizer.step()
            decoder_hydrogen_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_molecular_optimizer.zero_grad()
            decoder_carbon_optimizer.zero_grad()
            decoder_hydrogen_optimizer.zero_grad()

            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if global_rank == 0:
            if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    # 'Encoder Grad: {encoder_grad_norm:.4f}  '
                    # 'Decoder Grad: {decoder_grad_norm:.4f}  '
                    'Encoder LR: {encoder_lr:.6f}  '
                    # 'Decoder LR: {decoder_molecular_lr:.6f}-{decoder_carbon_lr:.6f}-{decoder_hydrogen_lr:.6f}  '
                    'Decoder LR: {decoder_molecular_lr:.6f}  '
                    .format(
                    epoch+1, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    remain=timeSince(start, float(step+1)/len(train_loader)),
                    # encoder_grad_norm=encoder_grad_norm,
                    # decoder_grad_norm=decoder_grad_norm,
                    encoder_lr=encoder_scheduler.get_lr()[0],
                    decoder_molecular_lr=decoder_molecular_scheduler.get_lr()[0],
                    # decoder_carbon_lr=decoder_carbon_scheduler.get_lr()[0],
                    # decoder_hydrogen_lr=decoder_hydrogen_scheduler.get_lr()[0],
                    ))
    return losses.avg

def dist_all_gather(res, ori, func):
    dist.all_gather(res, ori)
    res = torch.cat(res)
    if func == 'cat':
        return res

def valid_fn(valid_loader, encoder, decoder_molecular, decoder_carbon, decoder_hydrogen,
             tokenizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluation mode
    encoder.eval()
    decoder_molecular.eval()
    decoder_carbon.eval()
    decoder_hydrogen.eval()
    text_preds = []
    text_labels = []
    seq_preds = []
    start = end = time.time()
    valid_tqdm_loader = enumerate(tqdm(valid_loader, 'eval valid set'))
    for step, (images, labels, molecular_labels, carbon_labels, hydrogen_labels) in valid_tqdm_loader:
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            features = encoder(images)
        _text_molecular_preds = predict_fn(decoder_molecular, features, CFG.max_molecular_len, tokenizer)
        _text_carbon_preds = predict_fn(decoder_carbon, features, CFG.max_carbon_len, tokenizer)
        _text_hydrogen_preds = predict_fn(decoder_hydrogen, features, CFG.max_hydrogen_len, tokenizer)
        text_labels.append(tokenizer.predict_captions(labels))
        _text_preds = [merge_text_preds(molecular, carbon, hydrogen) for (molecular, carbon, hydrogen) in zip(_text_molecular_preds,
                                                                                                       _text_carbon_preds,
                                                                                                       _text_hydrogen_preds)]
        text_preds.append(_text_preds)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if global_rank == 0:
            if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Elapsed {remain:s} '
                    .format(
                    step, len(valid_loader), batch_time=batch_time,
                    data_time=data_time,
                    remain=timeSince(start, float(step+1)/len(valid_loader)),
                    ))
    text_preds = np.concatenate(text_preds)
    text_labels = np.concatenate(text_labels)
    text_preds = [f"InChI=1S/"+text for text in text_preds]
    text_labels = [f"InChI=1S/"+text.replace('<sos>','') for text in text_labels]
    score = torch.Tensor([get_score(text_labels, text_preds)]).to(device)
    LOGGER.info(f'rank {local_rank}, score {score.cpu().numpy()[0]}')
    dist.all_reduce(score, op=dist.ReduceOp.SUM)
    score = score.cpu().numpy()[0] / dist.get_world_size()
    if global_rank == 0:
        LOGGER.info(f"labels: {text_labels[:5]}")
        LOGGER.info(f"preds: {text_preds[:5]}")
    return score


class Encoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        if model_name.startswith('swin'): # swintransformer
            self.n_features = self.cnn.head.in_features
        elif hasattr(self.cnn, 'classifier'): # effb
            self.n_features = self.cnn.classifier.in_features
        elif hasattr(self.cnn, 'head'): #nfnet
            self.n_features = self.cnn.head.fc.in_features
        elif hasattr(self.cnn, 'fc'):   # resnet
            self.n_features = self.cnn.fc.in_features
        else:
            self.n_features = self.cnn.head.in_features

    def forward(self, x):
        features = self.cnn.forward_features(x)
        if len(features.size())==2:
            bs = features.size(0)
            features = features.view(bs,self.n_features,1,1)
        features = features.permute(0, 2, 3, 1)
        return features

# ====================================================
# Train loop
# ====================================================
def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values

    train_dataset = TrainDatasetWithMultiLabel(train_folds, tokenizer, transform=get_transforms(data='train'))
    valid_dataset = TrainDatasetWithMultiLabel(valid_folds, tokenizer, transform=get_transforms(data='valid'))

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset,
                              sampler = train_sampler,
                              batch_size=CFG.batch_size,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=bms_collate)
    valid_loader = DataLoader(valid_dataset,
                              sampler = valid_sampler,
                              batch_size=CFG.batch_size,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=bms_collate_valid)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    encoder = Encoder(CFG.model_name, pretrained=True)
    encoder_dim = encoder.n_features
    # DDP
    encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder).cuda().to(local_rank)
    encoder = torch.nn.parallel.DistributedDataParallel(encoder,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    if global_rank == 0:
        print('encoder DDP finish')
    encoder_optimizer = Adam(encoder.parameters(), lr=CFG.encoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    encoder_scheduler = get_scheduler(encoder_optimizer)

    def build_decoder():
        decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                    embed_dim=CFG.embed_dim,
                                    decoder_dim=CFG.decoder_dim,
                                    vocab_size=len(tokenizer),
                                    dropout=CFG.dropout,
                                    device=local_rank,
                                    encoder_dim=encoder_dim)
        decoder = nn.SyncBatchNorm.convert_sync_batchnorm(decoder).cuda().to(local_rank)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
        decoder_optimizer = Adam(decoder.parameters(), lr=CFG.decoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
        decoder_scheduler = get_scheduler(decoder_optimizer)

        return decoder, decoder_optimizer, decoder_scheduler

    decoder_molecular_envs = build_decoder()
    decoder_carbon_envs = build_decoder()
    decoder_hydrogen_envs = build_decoder()

    if global_rank == 0:
        print('decoder DDP finish')
    # ====================================================
    # loop
    # ===================================================
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])

    best_score = np.inf
    best_loss = np.inf

    for epoch in range(CFG.epochs):
        train_sampler.set_epoch(epoch)
        print(f'START EPOCH{epoch}')
        start_time = time.time()
        # train
        avg_loss = train_fn(train_loader,
                            encoder,
                            decoder_molecular_envs,
                            decoder_carbon_envs,
                            decoder_hydrogen_envs,
                            criterion, epoch,
                            encoder_scheduler,
                            encoder_optimizer,
                            local_rank)
        score = valid_fn(valid_loader, encoder, decoder_molecular_envs[0], decoder_carbon_envs[0], decoder_hydrogen_envs[0],
             tokenizer, criterion, local_rank)

        if isinstance(encoder_scheduler, ReduceLROnPlateau):
            encoder_scheduler.step(score)
        elif isinstance(encoder_scheduler, CosineAnnealingLR):
            encoder_scheduler.step()
        elif isinstance(encoder_scheduler, CosineAnnealingWarmRestarts):
            encoder_scheduler.step()

        for decoder_envs in [decoder_molecular_envs, decoder_carbon_envs, decoder_hydrogen_envs]:
            decoder_scheduler = decoder_envs[2]
            if isinstance(decoder_scheduler, ReduceLROnPlateau):
                decoder_scheduler.step(score)
            elif isinstance(decoder_scheduler, CosineAnnealingLR):
                decoder_scheduler.step()
            elif isinstance(decoder_scheduler, CosineAnnealingWarmRestarts):
                decoder_scheduler.step()

        elapsed = time.time() - start_time
        if global_rank == 0:
            LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
            LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
            if args.local:
                save_path = f'../weights/{CFG.model_name}_{CFG.meta_info}'
            else:
                save_path = f'/mnt/epblob/zhgao/MT/weights/{CFG.model_name}_{CFG.meta_info}'
            os.makedirs(f'{save_path}/{CFG.model_name}_{CFG.meta_info}', exist_ok=True)
            if score < best_score:
                best_score = score
                LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'encoder': encoder.state_dict(),
                            # 'encoder_optimizer': encoder_optimizer.state_dict(),
                            # 'encoder_scheduler': encoder_scheduler.state_dict(),
                            'decoder_molecular': decoder_molecular_envs[0].state_dict(),
                            # 'decoder_optimizer': decoder_optimizer.state_dict(),
                            # 'decoder_scheduler': decoder_scheduler.state_dict(),
                            'decoder_carbon': decoder_carbon_envs[0].state_dict(),
                            # 'decoder_optimizer': decoder_optimizer.state_dict(),
                            # 'decoder_scheduler': decoder_scheduler.state_dict(),
                            'decoder_hydrogen': decoder_hydrogen_envs[0].state_dict(),
                            # 'decoder_optimizer': decoder_optimizer.state_dict(),
                            # 'decoder_scheduler': decoder_scheduler.state_dict(),
                            },
                            f'{save_path}/{CFG.model_name}_{CFG.meta_info}_epoch_{epoch}_fold_{fold}_cv_{best_score}.pth')
        dist.barrier()

def main():

    """
    Prepare: 1.train  2.folds
    """

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                train_loop(folds, fold)


if __name__ == '__main__':
    main()
