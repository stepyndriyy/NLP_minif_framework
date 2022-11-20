#!/usr/bin/env python3
# coding=utf-8

import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Dataset

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from nltk.tokenize import WordPunctTokenizer

from gensim.models import Word2Vec


# no batches 
class RNNclassificationProblem(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNclassificationProblem, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size) # simplyfied lstm
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input):
        hidden = self.initHidden()
        output, hidden = self.gru(input, hidden)
        output = self.linear(hidden)
        output = self.softmax(output)
        return output, hidden

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, training_batch, batch_idx):
        X, Y = training_batch
        x_hat, hidden = self.forward(X)

        criterion = nn.NLLLoss()
        loss = criterion(x_hat, Y)
        self.log('train_loss', loss, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, Y = val_batch
        
        x_hat, hidden = self.forward(X)

        criterion = nn.NLLLoss()
        loss = criterion(x_hat, Y)
        self.log('val_loss', loss, on_epoch=True, batch_size=1)

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class TRECDataset(Dataset):
    def __init__(self, path):
        self.data = []
        self.path = path
        Y, X = [], []
        for line in list(open(path, encoding='utf-8')):
            splitted = line.split(':')
            Y.append(splitted[0])
            X.append(splitted[1])
            
        label_set = {}
        label_num = 0
        for y in Y:
            if y not in label_set:
                label_set[y] = label_num
                label_num += 1
        
        self.n_iters = len(X)
        for i in range(self.n_iters):
            self.data.append([X[i], label_set[Y[i]]])
        
    def __len__(self):
        return self.n_iters

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

class EmbeddedDataset(Dataset):
    def __init__(self, dataset, embedding):
        self.embedding = embedding
        self.data = dataset
        self.n_iters = len(dataset)

        for dt in self.data:
            dt[0] = embedding.get_phrase_embeddings(dt[0])
            dt[1] = torch.tensor([dt[1]])
        
    def __len__(self):
        return self.n_iters

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]
        

class Embedding:
    def __init__(self, dataset):
        self.tokenizer = WordPunctTokenizer()
        data_tok = list(self.tokenizer.tokenize(line[0].lower()) for line in trec_dataset.data)

        self.embedding = Word2Vec(data_tok,  
                            vector_size=32,      # embedding vector size
                            min_count=1,  # consider words that occured at least 1 times
                        window=5).wv  # define context as a 5-word window around the target word

    def get_phrase_embeddings(self, phrase):
        ans = []
        for token in self.tokenizer.tokenize(phrase.lower()):
            ans.append(list(self.embedding.get_vector(token)))
        return torch.tensor(ans)

if __name__ == "__main__":

    # TODO load from arguments
    #save_path = './checkpoints/'

    trec_dataset = TRECDataset('./TREC.txt')

    embedding = Embedding(trec_dataset)
                    
    #print(embedding.get_phrase_embeddings("who are you?"))    
    
    trec_embedded_dataset = EmbeddedDataset(trec_dataset, embedding)

    #print(trec_embedded_dataset[0])

    dataloader = DataLoader(trec_embedded_dataset, batch_size=None)

    model = RNNclassificationProblem(32, 64, 6)

    trainer = pl.Trainer(max_epochs=10, enable_checkpointing=True)
    trainer.fit(model, dataloader)
