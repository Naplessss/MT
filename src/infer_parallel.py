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
from torch.nn import DataParallel
import timm
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn

class Attention(nn.Module):
    """
    Attention network for calculate attention value
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: input size of encoder network
        :param decoder_dim: input size of decoder network
        :param attention_dim: input size of attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        # self.layer_norm_1 = nn.LayerNorm(encoder_dim)
        # self.layer_norm_2 = nn.LayerNorm(decoder_dim)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    """
    Decoder network with attention network used for training
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, device, encoder_dim=512, dropout=0.5, pool_size=32):
        """
        :param attention_dim: input size of attention network
        :param embed_dim: input size of embedding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of characters used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.pool_size = pool_size
        self.pool = nn.AdaptiveMaxPool2d(pool_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, max_len=275):
        """
        :param encoder_out: output of encoder network
        :param encoded_captions: transformed sequence from character to integer
        :param caption_lengths: length of transformed sequence
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # embedding transformed sequence for vector
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        # set decode length by caption length - 1 because of omitting start token
        decode_lengths = (caption_lengths - 1).tolist()
        # decode_lengths = [min(max_len, decode_length) for decode_length in decode_lengths]
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)
        # predict sequence
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        # decode_lengths = torch.tensor(decode_lengths, dtype=torch.int32).to(self.device)
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def predict(self, encoder_out, decode_lengths, tokenizer):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        # embed start tocken for LSTM input
        start_tockens = torch.ones(batch_size, dtype=torch.long).to(self.device) * tokenizer.stoi["<sos>"]
        embeddings = self.embedding(start_tockens)
        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        predictions = torch.zeros(batch_size, decode_lengths, vocab_size).to(self.device)
        # predict sequence
        for t in range(decode_lengths):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            if np.argmax(preds.detach().cpu().numpy()) == tokenizer.stoi["<eos>"]:
                break
            embeddings = self.embedding(torch.argmax(preds, -1))
        return predictions

    def forward_step(self, prev_tokens, hidden, encoder_out, function):
        assert len(hidden) == 2
        h, c = hidden
        h, c = h.squeeze(0), c.squeeze(0)

        embeddings = self.embedding(prev_tokens)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        attention_weighted_encoding, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding
        h, c = self.decode_step(
            torch.cat([embeddings, attention_weighted_encoding], dim=1),
            (h, c))  # (batch_size_t, decoder_dim)
        preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

        hidden = (h.unsqueeze(0), c.unsqueeze(0))
        predicted_softmax = function(preds, dim=1)
        return predicted_softmax, hidden, None

from utils import Tokenizer
from utils import get_train_file_path, get_score, init_logger, seed_torch
from dataset import TrainDataset, TestDataset
from beam import TopKDecoder
# from attention import Attention, DecoderWithAttention
from utils import AverageMeter, asMinutes, timeSince
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


import argparse
parser = argparse.ArgumentParser(description='CNN-LSTM')
parser.add_argument('--model_name', default='efficientnet_b3_pruned')
parser.add_argument('--weight_path', default='', type=str)
parser.add_argument('--weight_name', default='efficientnet_b3_pruned_bs32x8_size320_v5_epoch_19_fold_0_cv_2.010636568069458.pth')
parser.add_argument('--beam_size', default=5, type=int)
parser.add_argument('--size', default=224, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--nodes', default=1, type=int)
parser.add_argument('--debug', default=1, type=int)
parser.add_argument('--max_len', default=256, type=int)
parser.add_argument('--local', default=0, type=int)
parser.add_argument('--fix', default=1, type=int)
args = parser.parse_args()

class CFG:
    debug=False if args.debug==0 else True
    weight_path=args.weight_path
    weight_name=args.weight_name
    max_len=args.max_len
    local=args.local
    fix=args.fix
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

if args.nodes > 1:
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

# 1) DDP init
torch.distributed.init_process_group(backend="nccl")
global_rank = torch.distributed.get_rank()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print('cur rank',local_rank,'global_rank', global_rank)

test = pd.read_csv('../sample_submission.csv')
world_size = torch.distributed.get_world_size()
part = global_rank
test = test.iloc[global_rank::world_size].reset_index(drop=True)

def get_test_file_path(image_id):
    return "../test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id
    )

test['file_path'] = test['image_id'].apply(get_test_file_path)

print(f'test.shape: {test.shape}; global_rank: {global_rank}, local_rank: {local_rank}, world_size:{world_size}, part:{part}')

tokenizer = torch.load('../tokenizer2.pth')

if CFG.debug:
    test = test.head(1000)

def get_transforms(*, data):
    if data == 'test':
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
    if CFG.local:
        net_dict = torch.load(f'../weights/{CFG.weight_path}/{CFG.weight_name}', map_location=lambda storage, loc: storage)
    else:
        net_dict = torch.load(f'/mnt/epblob/zhgao/MT/weights/{CFG.weight_path}/{CFG.weight_name}', map_location=lambda storage, loc: storage)

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
    encoder.cuda().to(local_rank)
    encoder_dim = encoder.n_features
    encoder = torch.nn.parallel.DistributedDataParallel(encoder,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                   embed_dim=CFG.embed_dim,
                                   decoder_dim=CFG.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   dropout=CFG.dropout,
                                   device=device,
                                   encoder_dim=encoder_dim)
    decoder.load_state_dict(net_dict['decoder'])
    decoder.cuda().to(local_rank)
    decoder = torch.nn.parallel.DistributedDataParallel(decoder,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    test_dataset = TestDataset(test, transform=get_transforms(data='test'), fix=CFG.fix)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    predictions = inference(test_loader, encoder, decoder, tokenizer, local_rank)

    # submission
    test['InChI'] = [f"{text}" for text in predictions]
    if CFG.local:
        test[['image_id', 'InChI']].to_csv(f'../weights/{CFG.weight_path}/{CFG.weight_name}_sub_fix_{CFG.fix}_part_{part}.csv', index=False)
    else:
        test[['image_id', 'InChI']].to_csv(f'/mnt/epblob/zhgao/MT/weights/{CFG.weight_path}/{CFG.weight_name}_sub_fix_{CFG.fix}_part_{part}.csv', index=False)
    print(test[['image_id', 'InChI']].head())
