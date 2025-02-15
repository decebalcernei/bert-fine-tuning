from collections import Counter
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
import os
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import re
import nltk
from nltk import word_tokenize
import sys
import string


# GLOBAL VARIABLES
device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0


def get_dataloaders(batch_size):
    
    tmp_train_raw_reviews, tmp_train_raw_aspects = load_data(os.path.join( 'dataset', 'laptop14_train.txt'))
    test_raw_reviews, test_raw_aspects = load_data(os.path.join( 'dataset', 'laptop14_test.txt'))
    
    # We do not have a dev set -> let's create it specifying the portion of the train set to use
    train_raw_reviews, train_raw_aspects, dev_raw_reviews, dev_raw_aspects = create_dev_set(tmp_train_raw_reviews, tmp_train_raw_aspects, portion=0.1)

    # Create the datasets
    train_dataset = Aspects(train_raw_reviews, train_raw_aspects)
    dev_dataset = Aspects(dev_raw_reviews, dev_raw_aspects)
    test_dataset = Aspects(test_raw_reviews, test_raw_aspects)
    
    
    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader


def load_data(path):

    mapper = {'O': 1, 'T': 2}

    # we'll store all reviews and aspects.
    reviews_dataset = []
    aspects_dataset = []

    with open(path, 'r') as f:
        entries = f.readlines()

        for entry in entries:
            _, review_and_tags = entry.split("####")
            # review and tags is a string of tokens=tags
            # we want a list to split them tokens to one side and tags to the other
            review_and_tags = review_and_tags.split()
            review = []
            tags = []
            for element in review_and_tags:
                token, tag = element.rsplit('=', 1) # element.split('=') will give an error when we have == in element
                review.append(token)
                # we use tag[0] because we're only interested in aspect identification, we don't need the polarity
                # furthermore we use the mapper for convenience.
                tags.append(mapper[tag[0]])
            reviews_dataset.append(review)
            aspects_dataset.append(tags)
    return reviews_dataset, aspects_dataset
       


def create_dev_set(tmp_train_raw_reviews, tmp_train_raw_aspects, portion=0.1):
    # we do not have the intents to stratify on anymore -> we'll randomically select a portion

    dev_raw_reviews = []
    dev_raw_aspects = []

    n_elements = round(len(tmp_train_raw_reviews)*portion)

    elements = random.sample(range(len(tmp_train_raw_reviews)), n_elements)

    elements.sort(reverse = True)

    for element in elements:
        # we remove the review&aspect from train and add it to dev
        review = tmp_train_raw_reviews.pop(element)
        aspect = tmp_train_raw_aspects.pop(element)
        dev_raw_reviews.append(review)
        dev_raw_aspects.append(aspect)


    return tmp_train_raw_reviews, tmp_train_raw_aspects, dev_raw_reviews, dev_raw_aspects


class Aspects(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset_reviews, dataset_spans, unk='unk'):
        self.reviews = []
        self.spans = []
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for i in range(len(dataset_reviews)): # len(dataset_reviews) == len(dataset_spans) 
            # to have similar situation like previous exercise => reuse same code :)
            review = ' '.join(dataset_reviews[i])
            self.spans.append(dataset_spans[i])
            self.reviews.append(review)
        
        self.review_ids = self.tokenizer(self.reviews, padding=True)
        review_max_length = len(self.review_ids['attention_mask'][0]) # they will have all the same length
        self.aligned_spans = self.align_spans(self.reviews, self.spans, review_max_length)
        


    def __len__(self):
        return len(self.reviews)


    def __getitem__(self, idx):
        # We get the info about the utterance, the slots(which are aligned), and intent
        input_ids = torch.tensor(self.review_ids['input_ids'][idx], dtype=torch.long)
        attention_mask = torch.tensor(self.review_ids['attention_mask'][idx], dtype=torch.long)
        token_type_ids = torch.tensor(self.review_ids['token_type_ids'][idx], dtype=torch.long)
        spans = torch.tensor(self.aligned_spans[idx], dtype=torch.long)
        #starts = torch.tensor(self.start_spans[idx], dtype=torch.long)
        #ends = torch.tensor(self.end_spans[idx], dtype=torch.long)

        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'spans': spans
        }
        return sample


    # Auxiliary methods
    def align_spans(self, reviews, spans, max_length, pad=0):
        # assuming len(reviews) == len(spans)
        aligned_spans_list = []
        counter = 0

        for review in reviews:

            aligned_spans = []
            aligned_spans.append(pad) # CLS
            #tokenized_review = self.tokenizer(review)
            review = review.split()
            spans_current = spans[counter]

            for i in range(len(review)):
                token = self.tokenizer(review[i])
                token_length = len(token['input_ids'][1:-1])
                aligned_spans.append(spans_current[i])
                if token_length > 1:
                    """
                    print(f'qui abbiamo avuto una subtokenization: {review}, token={self.tokenizer.convert_ids_to_tokens(token["input_ids"])}')
                    print(f'il primo span è {spans_current[i]}, poi metto {token_length-1} di [PAD]')
                    exit()
                    """
                    # We experienced subtokenization, we put the same exact slot for every subtoken
                    aligned_spans.extend([pad] * (token_length-1))

            counter += 1

            aligned_spans.append(pad) # SEP
            #if len(tokenized_review["input_ids"]) != len(aligned_spans):
            #    print(f'review: {" ".join(review)}, number {counter-1}, created problems of misalignement')

            if len(aligned_spans) < max_length:
                difference = max_length - len(aligned_spans)
                aligned_spans.extend([pad] * difference) # padding

            aligned_spans_list.append(aligned_spans)
        return aligned_spans_list # list of lists



# Since the way we designed the dataset class we do not need to pad anything
# because bert's tokenizer already did (specifying padding=True)

"""
Its main objective is to create your batch without spending much time
implementing it manually. Try to see it as a glue that you specify the way
examples stick together in a batch. If you don’t use it, PyTorch only put
batch_size examples together as you would using torch.stack (not exactly it,
but it is simple like that).
"""

def collate_fn(data):

    new_item = {}
    input_ids = torch.stack([d['input_ids'] for d in data]).to(device)
    attention_mask = torch.stack([d['attention_mask'] for d in data]).to(device)
    token_type_ids = torch.stack([d['token_type_ids'] for d in data]).to(device)
    y_spans = torch.stack([d['spans'] for d in data]).to(device)


    new_item["input_ids"] = input_ids
    new_item["attention_mask"] = attention_mask
    new_item["token_type_ids"] = token_type_ids
    new_item["aspect_labels"] = y_spans

    return new_item


def init_weights(mat):
    """
    m: module (like "nn.Linear", "nn.LSTM")
    m_name : name of the module (like "slot_out")
    """
    for m_name, m in mat.named_modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                if 'aspects_out' in m_name:
                    torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias != None:
                        m.bias.data.fill_(0.01)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# needed for the evaluation
# slightly modified to fit our implementation.
# original source: https://github.com/lixin4ever/E2E-TBSA/blob/master/utils.py

def tag2ot(ote_tag_sequence):
    """
    transform ote tag sequence to a sequence of opinion target
    :param ote_tag_sequence: tag sequence for ote task
    :return:
    """
    n_tags = len(ote_tag_sequence) # number of tags
    ot_sequence = [] # will store (start, end) for each aspect
    beg, end = -1, -1
    inside_span = False
    for i in range(n_tags):
        tag = ote_tag_sequence[i]
        if tag == 2: # if is 0 or 1 nothing.
            if inside_span: #we already encountered at least a 2 and this is another one
                if (i+1) >= n_tags or ote_tag_sequence[i+1] != 2: #here or the tags are finished or the next tag is not a 2 -> add end
                    end = i
                    ot_sequence.append((beg, end))
                    beg, end = -1, -1 # we added an aspect, reset indexes
                    inside_span = False # reset the insisde_span flag
            else: # this is the first 2 we encountered
                if i+1 < n_tags and ote_tag_sequence[i+1] == 2: # not S
                    beg = i
                    inside_span = True # we are inside a span
                else: # it's a S or was the last tag of the sequence
                    beg, end = i, i
                    ot_sequence.append((beg, end))
                    beg, end = -1, -1 # we added an aspect, reset indexes
    return ot_sequence

