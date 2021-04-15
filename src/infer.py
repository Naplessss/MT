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
    IAAAdditiveGaussianNoise, Transpose, Blur
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from torch.nn import DataParallel
import timm
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

from utils import Tokenizer
from utils import get_train_file_path, get_score, init_logger, seed_torch
from dataset import TrainDataset, TestDataset
from beam import TopKDecoder
from attention import Attention, DecoderWithAttention
from utils import AverageMeter, asMinutes, timeSince

import argparse
parser = argparse.ArgumentParser(description='CNN-LSTM')
parser.add_argument('--model_name', default='efficientnet_b3_pruned')
parser.add_argument('--weight_name', default='efficientnet_b3_pruned_bs32x8_size320_epoch_12_fold_0_cv_2.9199962615966797.pth')
parser.add_argument('--beam_size', default=2, type=int)
parser.add_argument('--size', default=224, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--debug', default=1, type=int)
args = parser.parse_args()

class CFG:
    debug=False if args.debug==0 else True
    weight_name=args.weight_name
    max_len=275
    print_freq=1000
    num_workers=6
    model_name=args.model_name
    size=args.size
    batch_size=args.batch_size
    k=args.beam_size
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5

CFG = CFG()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test = pd.read_csv('../sample_submission.csv')

def get_test_file_path(image_id):
    return "../test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id
    )

test['file_path'] = test['image_id'].apply(get_test_file_path)

print(f'test.shape: {test.shape}')

tokenizer = torch.load('../tokenizer.pth')

if CFG.debug:
    test = test.head(1000)

def get_transforms(*, data):

    if data == 'train':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

class Encoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        if hasattr(self.cnn, 'classifier'): # effb
            self.n_features = self.cnn.classifier.in_features
        elif hasattr(self.cnn, 'head'): #nfnet
            self.n_features = self.cnn.head.fc.in_features
        elif hasattr(self.cnn, 'fc'):   # resnet
            self.n_features = self.cnn.fc.in_features
        else:
            self.n_features = 3072

    def forward(self, x):
        features = self.cnn.forward_features(x)
        features = features.permute(0, 2, 3, 1)
        return features

def inference(test_loader, encoder, decoder, tokenizer, device):
    encoder.eval()
    decoder.eval()
    text_preds = []

    # k = 2
    topk_decoder = TopKDecoder(decoder, CFG.k, CFG.decoder_dim, CFG.max_len, tokenizer)

    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        images = images.to(device)
        predictions = []
        with torch.no_grad():
            encoder_out = encoder(images)
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
            if hasattr(decoder, 'module'):
                h, c = decoder.module.init_hidden_state(encoder_out)
            else:
                h, c = decoder.init_hidden_state(encoder_out)
            hidden = (h.unsqueeze(0), c.unsqueeze(0))

            decoder_outputs, decoder_hidden, other = topk_decoder(None, hidden, encoder_out)

            for b in range(batch_size):
                length = other['topk_length'][b][0]
                tgt_id_seq = [other['topk_sequence'][di][b, 0, 0].item() for di in range(length)]
                predictions.append(tgt_id_seq)
            assert len(predictions) == batch_size

        predictions = tokenizer.predict_captions(predictions)
        predictions = ['InChI=1S/' + p.replace('<sos>', '') for p in predictions]
        text_preds.append(predictions)
    text_preds = np.concatenate(text_preds)
    return text_preds

def load_weight():
    net_dict = torch.load(f'../weights/{CFG.weight_name}', map_location=lambda storage, loc: storage)
    for key in ['encoder', 'decoder']:
        new_state_dict = OrderedDict()
        for k,v in net_dict[key].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
            net_dict[key] = new_state_dict
    return net_dict

if __name__ == '__main__':
    net_dict = load_weight()
    encoder = Encoder(CFG.model_name, pretrained=False)
    encoder.load_state_dict(net_dict['encoder'])
    encoder.to(device)
    encoder_dim = encoder.n_features

    decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                   embed_dim=CFG.embed_dim,
                                   decoder_dim=CFG.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   dropout=CFG.dropout,
                                   device=device,
                                   encoder_dim=encoder_dim)
    decoder.load_state_dict(net_dict['decoder'])
    decoder = DataParallel(decoder).to(device)


    test_dataset = TestDataset(test, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    predictions = inference(test_loader, encoder, decoder, tokenizer, device)

    # submission
    test['InChI'] = [f"{text}" for text in predictions]
    # test[['image_id', 'InChI']].to_csv(f'/mnt/epblob/zhgao/MT/weights/{CFG.model_name}_{CFG.meta_info}/{CGF.weight_path}_sub.csv', index=False)
    test[['image_id', 'InChI']].to_csv(f'../output/{CFG.weight_name}_sub.csv', index=False)
    print(test[['image_id', 'InChI']].head())
