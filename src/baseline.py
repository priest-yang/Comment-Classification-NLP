
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm.auto import tqdm, trange
from collections import Counter
import random
from torch import optim

import pandas as pd
import pickle

import wandb

import numpy as np
from sklearn.metrics import f1_score

# Sort of smart tokenization
from nltk.tokenize import RegexpTokenizer
from gensim import corpora

# Attention plotting
import pickle

import os
import sys
cur_dir = os.path.dirname(os.path.abspath("__file__"))  # Gets the current notebook directory
src_dir = os.path.join(cur_dir, '../')  # Constructs the path to the 'src' directory
# Add the 'src' directory to sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils import *



batch_size = 1
embedding_size = 50
learning_rate = 5e-5
epochs = 200
max_steps = 160000

# !wandb login  # Make sure to run this cell before training
wandb.init(project="630proj baseline")


dictionary = corpora.Dictionary.load('../data/comments.dict')
index_to_word = dictionary.id2token
word_to_index = dictionary.token2id
tokenizer = RegexpTokenizer(r'\w+')


class DocumentAttentionClassifier(nn.Module):
    
    def __init__(self, vocab_size, num_heads, embeddings_fname, num_classes=24):
        '''
        Creates the new classifier model. embeddings_fname is a string containing the
        filename with the saved pytorch parameters (the state dict) for the Embedding
        object that should be used to initialize this class's word Embedding parameters
        '''
        super(DocumentAttentionClassifier, self).__init__()
        
        torch.set_default_dtype(torch.float32)  # Set default to float64
        
        # Save the input arguments to the state
        self.vocab_size = vocab_size
        self.word_embeddings = torch.load(embeddings_fname)
        self.embedding_size = self.word_embeddings.embedding_dim
        self.num_heads = num_heads
        self.embeddings_fname = embeddings_fname

        self.attention = torch.empty((num_heads, self.embedding_size),dtype=torch.float32, requires_grad=True)
        torch.nn.init.uniform_(self.attention, a=-0.5, b=0.5)

        self.linear = nn.Linear(self.embedding_size * num_heads, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

        init.normal_(self.linear.weight, mean=0.0, std=1.0)
        init.constant_(self.linear.bias, 0.0)
        pass
    

    def forward(self, word_ids):

        word_embeddings = self.word_embeddings(word_ids).squeeze(0)

        r = torch.einsum('ij,kj->ik', word_embeddings, self.attention)

        a = torch.softmax(r, dim=0)      

        d = torch.einsum('ij,ik->jk', word_embeddings, a).transpose(0, 1)

        d = d.flatten()

        output = self.linear(d)
        # output = self.sigmoid(output)

        return output


sent_train_df = pd.read_csv('../data/train.csv', index_col=0)
sent_dev_df = pd.read_csv('../data/test.csv', index_col=0)


sent_train_df['label'] = sent_train_df.apply(lambda x: np.array(x[2:].values), axis=1)
sent_dev_df['label'] = sent_dev_df.apply(lambda x: np.array(x[2:].values), axis=1)


train_list = []
dev_list = []

key = "Review"

for df in [sent_train_df, sent_dev_df]:
    df['text'] = df[key].str.lower()
    # if 'label' in df.columns:
    #     df['label'] = df['label'].astype(float)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        word_list = tokenizer.tokenize(row['text'])
        index_list = np.array([word_to_index[word] for word in word_list if word in word_to_index])

        if df is sent_train_df:
            label = row['label'].astype(float)
            train_list.append((index_list, label))
        elif df is sent_dev_df:
            label = row['label'].astype(float)
            dev_list.append((index_list, label))
    

model = DocumentAttentionClassifier(vocab_size=len(word_to_index), num_heads=4, embeddings_fname='../data/embedding_layer.pth', num_classes=train_list[0][1].shape[0])

loss_function = nn.CrossEntropyLoss() # Example for a classification task
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # Adjust learning rate as needed

train_dataloader = DataLoader(train_list, batch_size=batch_size, shuffle=True)



def run_eval(model, eval_data):
    '''
    Scores the model on the evaluation data and returns the F1
    '''
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for inputs, label in eval_data:
            inputs = torch.tensor(inputs).long()
            label = torch.tensor(label).float()
            try:
                pred= model(inputs)
                all_preds.append(pred)
                all_labels.append(label)
            except:
                continue
        
        te_label = np.array([label.numpy() for label in all_labels])
        te_pred = np.array([pred.numpy() for pred in all_preds])
        test_metrics = get_modified_evaluation(te_label, te_pred, list_metrics=["hamming_loss"], threshold=0.25) 

        return test_metrics



for epoch in tqdm(range(epochs)):
    model.train()

    loss_sum = 0
    # TODO: use your DataLoader to iterate over the data
    for step, data in enumerate(train_dataloader):

        # NOTE: since you created the data np.array instances,
        # these have now been converted to Tensor objects for us
        word_ids, label = data    
        
        # TODO: Fill in all the training details here
        try:
            outputs = model(word_ids.squeeze(0))
        except:
            continue
        
        optimizer.zero_grad()
        loss = loss_function(outputs, label[0])
        loss_sum += loss.item()

        loss.backward()
        optimizer.step()

        # print(f'epochs: {epoch}, step: {step}, loss: {loss.item()}')
        
        if step > max_steps:
            print('Max steps reached')
            break

    # Evaluate the model after each epoch
    if epoch % 1 == 0:
        model.eval()
        test_metrics = run_eval(model, dev_list)
        print(f'Epoch {epoch} test metrics: {test_metrics}')
        wandb.log(test_metrics)
        












