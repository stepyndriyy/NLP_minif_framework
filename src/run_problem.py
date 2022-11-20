from seq2seq_model import Seq2SeqTransformer
from datasetloader import Seq2SeqDataSet

from torchtext.datasets import multi30k, Multi30k
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler
import torch
import pytorch_lightning as pl

if __name__ == "__main__":
    DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'

    # We need to modify the URLs for the dataset since the links to the original dataset are broken
    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    SRC_LANGUAGE = 'de'
    TGT_LANGUAGE = 'en'
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    
    data_from = []
    data_to = []
    for t in train_iter:
        x, y = t
        data_from.append(x)
        data_to.append(y)

    dataset = Seq2SeqDataSet(data_from, data_to, lang_first='de_core_news_sm', lang_second='en_core_web_sm')

    SRC_VOCAB_SIZE = len(dataset)
    TGT_VOCAB_SIZE = len(dataset)
    EMB_SIZE = 512//2
    NHEAD = 8//4
    FFN_HID_DIM = 512//2
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)


    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        sampler=BatchSampler(
            SequentialSampler(dataset), batch_size=BATCH_SIZE, drop_last=True
        ),
    )

    trainer = pl.Trainer(max_epochs=10, enable_checkpointing=True, accelerator=DEVICE)
    trainer.fit(transformer, dataloader)
    
