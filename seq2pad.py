from Bio import SeqIO
import sys
import argparse
import random
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
#from keras.utils import pad_sequences
from keras_preprocessing.sequence import pad_sequences

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
args=sys.argv
amino_acids=list("ARNDCEQGHILKMFPSTWYV")
#fa_file='step9_all.fa' # ARFG  03.fa protein.faa step9_all.fa AMIE_PSEAE.fasta
fa_file=args[1]# 1k_test.fa cdhit99.ros.fa.fa2
texts = []
#SeqIO.parse(fa_file, 'fasta')
for index, record in enumerate(SeqIO.parse(fa_file, 'fasta')):
    temp_str = ""
    for item in (record.seq):
                #        print(item)
        if item in amino_acids:
            temp_str = temp_str + " " + item
    texts.append(temp_str)
###NPL
MAX_NB_CHARS=21
MAX_SEQUENCE_LENGTH= 6269 
train_tokenizer = Tokenizer(num_words=MAX_NB_CHARS)
train_tokenizer.fit_on_texts(texts)
char_index = train_tokenizer.word_index
print('Found %s unique tokens.' % len(char_index))
print(char_index)
query_sequences = train_tokenizer.texts_to_sequences(texts)
query_x = pad_sequences(query_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
query_x_tensor = torch.from_numpy(query_x)
print (len(query_x_tensor))
torch.save(query_x_tensor,args[2])
