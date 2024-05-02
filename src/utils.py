import torch
import sys
import csv
csv.field_size_limit(sys.maxsize)
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def get_modified_evaluation(y_true, y_prob, list_metrics, threshold=0.25):
    output = {}
    
    # Normalize predictions and labels
    y_prob = np.where(y_prob < 0, 0, y_prob)

    y_pred_normalized = y_prob / np.max(y_prob, axis=1, keepdims=True)
    y_true_normalized = y_true / np.max(y_true, axis=1, keepdims=True)
    
    # Apply threshold to normalized predictions to get binary predictions
    y_pred_binary = np.where(y_pred_normalized > threshold, 1, 0)
    y_true_normalized = np.where(y_true_normalized > threshold, 1, 0)
    
    # Example metrics
    if 'hamming_loss' in list_metrics:
        output['hamming_loss'] = metrics.hamming_loss(y_true_normalized, y_pred_binary)
    
    # Add more custom or adapted metrics as needed
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print (word)
    print (sent)






