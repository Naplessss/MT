import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import torch

def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')

def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'/d+', j))
            assert len(num_list) == 1, '_'.join(num_list) + form
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')

def split_form3(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = '/'+elem if elem == 'c' else ' /'+elem
        regx = re.compile("[A-Z]?[0-9]+")
        char_list = regx.split(num)
        num_list = regx.findall(num)
        for i in range(len(num_list)):
            _char = ' '.join(char_list[i])
            _num = num_list[i]
            if _char == '':
                num_string += f" {_num}"
            else:
                num_string += f" {_char} {_num}"
        last_char = ' '.join(char_list[-1])
        if len(num_list) == 0 or len(char_list[-1])==1:
            num_string += f" {last_char}"
        else:
            num_string += f"{last_char}"
        string += num_string
    return string
# ====================================================
# Tokenizer
# ====================================================
class Tokenizer(object):

    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions

def main():
    train = pd.read_csv('../train_labels.csv')
    print(f'train.shape: {train.shape}')
    # ====================================================
    # preprocess train.csv
    # ====================================================
    train['InChI_1'] = train['InChI'].progress_apply(lambda x: x.split('/')[1])
    train['InChI_text'] = train['InChI_1'].progress_apply(split_form) + ' ' + \
                            train['InChI'].apply(lambda x: '/'.join(x.split('/')[2:])).progress_apply(split_form3).values
    # ====================================================
    # create tokenizer
    # ====================================================
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train['InChI_text'].values)
    torch.save(tokenizer, '../tokenizer.pth')
    print('Saved tokenizer')
    # ====================================================
    # preprocess train.csv
    # ====================================================
    lengths = []
    tk0 = tqdm(train['InChI_text'].values, total=len(train))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2
        lengths.append(length)
    train['InChI_length'] = lengths
    train.to_pickle('../train.pkl')
    print('Saved preprocessed train.pkl')

if __name__ == '__main__':
    main()