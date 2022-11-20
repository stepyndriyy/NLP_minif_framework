import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), # <bos>
                    torch.tensor(token_ids), 
                    torch.tensor([EOS_IDX]))) # <eos>

class dataTransformer:
    def __init__(self, data, tokenizer='spacy', language='en'):
        self.tokenizer = get_tokenizer(tokenizer, language=language)
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

        # helper function to yield list of tokens
        def yield_tokens(data_iter: Iterable) -> List[str]:
            for data_sample in data_iter:
                yield self.tokenizer(data_sample)

        self.vocab_transform = build_vocab_from_iterator(yield_tokens(data),
                                                        min_freq=1,
                                                        specials=self.special_symbols,
                                                        special_first=True)
        
        self.vocab_transform.set_default_index(UNK_IDX)

        # src and tgt language text transforms to convert raw strings into tensors indices
        self.text_transform = sequential_transforms(self.tokenizer, #Tokenization
                                                    self.vocab_transform, #Numericalization
                                                    tensor_transform) # Add BOS/EOS and create tensor
    
    def __call__(self, line):
        return self.text_transform(line)

class ClassificationDataSet(Dataset):
    def __init__(self, data, classification, tokenizer='spacy', language='en'):
        self.classification = classification
        self.classification_map = {}
        self.classification_size = 0
        for c in classification:
            if c not in classification_map:
                classification_map[c] = self.classification_size
                self.classification_size += 1
        
        self.data = data
        self.data_size = len(data)
        self.text_transform = dataTransformer(data, tokenizer=tokenizer, language=language)
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        src_batch = []
        for i in idx:
            src_sample = self.data[i]            
            src_batch.append(self.text_transform(src_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX) #, batch_first=True)
        return src_batch, tgt_batch

class Seq2SeqDataSet(Dataset):
    def __init__(self, data_lang_first, data_lang_second, tokenizer='spacy', lang_first='en', lang_second='en'):
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.data_lang_first = data_lang_first
        self.data_lang_second = data_lang_second
        self.data_size = len(data_lang_first)
        self.text_transform = {}
        self.text_transform[0] = dataTransformer(data_lang_first, tokenizer=tokenizer, language=lang_first)
        self.text_transform[1] = dataTransformer(data_lang_second, tokenizer=tokenizer, language=lang_second)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        src_batch, tgt_batch = [], []
        # idx = torch.tensor([idx]).squeeze()
        for i in idx:
            #print("i: ", i)
            src_sample = self.data_lang_first[i]
            tgt_sample = self.data_lang_second[i]
            src_batch.append(self.text_transform[0](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[1](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX) #, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX) #, batch_first=True)
        return src_batch, tgt_batch
    