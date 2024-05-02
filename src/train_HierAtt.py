
import numpy as np
import pandas as pd
import csv
from nltk.tokenize import word_tokenize, sent_tokenize
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matrix_mul, element_wise_mul
from tqdm import tqdm

from gensim import corpora

import os
import sys
# cur_dir = os.path.dirname(os.path.abspath("__file__"))  # Gets the current notebook directory
# src_dir = os.path.join(cur_dir, '../src')  # Constructs the path to the 'src' directory
# # Add the 'src' directory to sys.path
# if src_dir not in sys.path:
#     sys.path.append(src_dir)

from utils import *

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_max_lengths, get_evaluation

import argparse
import shutil
import numpy as np

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)

import warnings
warnings.filterwarnings("ignore")

import wandb
wandb.init(project='si630proj')


class MyDataset(Dataset):
    
    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            next(reader, None)  # Skip the header
            for idx, line in enumerate(reader):
                # Assuming the first column is station_id, and the second column is the review text.
                text = line[2].lower()
                # Read the rest of the columns as labels (multi-label for each row)
                label = [float(label) for label in line[3:]]  # Adjusted to read multiple labels
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = np.array(labels)  # Convert labels to a numpy array for easier handling

        self.dict = corpora.Dictionary.load(dict_path)
        # self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
        #                         usecols=[0]).values
        # self.dict = list(self.dict.token2id.keys())
        # self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = self.labels.shape[1]  # Adjusted to get the number of classes from the label shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]  # Now label is a vector
        text = self.texts[index]
        document_encode = [
            [self.dict.token2id[word] if word in self.dict.token2id else -1 for word in word_tokenize(text=sentences)] for sentences
            in sent_tokenize(text=text)]

            # [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            # in sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label.astype(np.float32)  # Ensure correct data type for labels



class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttNet, self).__init__()

        # dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        # dict_len, embed_size = dict.shape
        # dict_len += 1
        
        # unknown_word = np.zeros((1, embed_size))
        # dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = torch.load(word2vec_path)
        # self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        
        embed_size = self.lookup.embedding_dim
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output)
        output = element_wise_mul(f_output,output.permute(1,0))

        return output, h_output


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils import matrix_mul, element_wise_mul

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output



class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):

        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)

        return output






train_set = '../data/train.csv'
test_set = '../data/test.csv'
word2vec_path = '../data/embedding_layer.pth'
dict_path = '../data/comments.dict'
word_hidden_size = 50
sent_hidden_size = 50
batch_size = 2
lr = 1e-4
momentum = 0.9
num_epoches = 200
test_interval = 1 #test every epoch
es_min_delta = 0.0
es_patience = 5

training_params = {"batch_size": batch_size,
                   "shuffle": True,
                   "drop_last": True}

test_params = {"batch_size": batch_size,
                   "shuffle": False,
                   "drop_last": False}


max_word_length, max_sent_length = get_max_lengths(train_set)
training_set = MyDataset(train_set, dict_path, max_sent_length, max_word_length)
training_generator = DataLoader(training_set, **training_params)
test_set = MyDataset(test_set, dict_path, max_sent_length, max_word_length)
test_generator = DataLoader(test_set, **test_params)
model = HierAttNet(word_hidden_size, sent_hidden_size, batch_size, training_set.num_classes,
                   word2vec_path, max_sent_length, max_word_length)



if torch.cuda.is_available():
        model.cuda()

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
best_loss = 1e-1
best_epoch = 0
model.train()
num_iter_per_epoch = len(training_generator)


wandb.run.name = f"lr{lr}_batch{batch_size}_{optimizer.__class__.__name__}"
wandb.run.save()

for epoch in range(num_epoches):
    for iter, (feature, label) in tqdm(enumerate(training_generator), total=num_iter_per_epoch):
        if torch.cuda.is_available():
            feature = feature.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        model._init_hidden_state()
        predictions = model(feature)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()
        training_metrics = get_modified_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["hamming_loss"], threshold=0.25)
        wandb.log({'train_loss':training_metrics['hamming_loss']})

        # print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Hamming Loss: {}".format(
        #     epoch + 1,
        #     num_epoches,
        #     iter + 1,
        #     num_iter_per_epoch,
        #     optimizer.param_groups[0]['lr'],
        #     loss, training_metrics["hamming_loss"]))
        # writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
        # writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
    if epoch % test_interval == 0:
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for te_feature, te_label in tqdm(test_generator):
            num_sample = len(te_label)
            if num_sample != model.batch_size:
                continue

            if torch.cuda.is_available():
                te_feature = te_feature.cuda()
                te_label = te_label.cuda()
            with torch.no_grad():
                model._init_hidden_state(num_sample)
                te_predictions = model(te_feature)
            te_loss = criterion(te_predictions, te_label)
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())
        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array([te_label_ten.numpy() for te_label_ten in te_label_ls])
        test_metrics = get_modified_evaluation(te_label, te_pred.numpy(), list_metrics=["hamming_loss"], threshold=0.25) 
        # test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["loss"]) #list_metrics=["accuracy", "confusion_matrix"]
        # output_file.write(
        #     "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
        #         epoch + 1, num_epoches,
        #         te_loss,
        #         test_metrics["accuracy"],
        #         test_metrics["confusion_matrix"]))
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Hamming Loss: {}".format(
            epoch + 1,
            num_epoches,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["hamming_loss"]))

        wandb.log({'test_loss':test_metrics['hamming_loss']})
        # writer.add_scalar('Test/Loss', te_loss, epoch)
        # writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        model.train()
        # if te_loss + es_min_delta < best_loss:
        #     best_loss = te_loss
        #     best_epoch = epoch
        #     torch.save(model, saved_path + os.sep + "whole_model_han")
            
        # # Early stopping
        # if epoch - best_epoch > es_patience > 0:
        #     print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
        #     break


