from collections import Counter
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
import os
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# GLOBAL VARIABLES
device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0


def get_dataloaders(batch_size):
    
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

    # We do not have a dev set -> let's create it specifying the portion of the train set to use
    train_raw, dev_raw = create_dev_set(tmp_train_raw, portion=0.1)
    

    # We create the lang class that contains the mapping word2id, lab2id and viceversa
    lang = create_lang(train_raw, dev_raw, test_raw, cutoff=0)
    
    # Create the datasets
    train_dataset, dev_dataset, test_dataset = create_datasets(train_raw, dev_raw, test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=batch_size*2, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, lang


def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def create_dev_set(tmp_train_raw, portion=0.1):

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw


def create_lang(train_raw, dev_raw, test_raw, cutoff=0):
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff)

    return lang


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab


def create_datasets(train_raw, dev_raw, test_raw, lang):
    train_dataset = IntentsAndSlots_Bert(train_raw, lang)
    dev_dataset = IntentsAndSlots_Bert(dev_raw, lang)
    test_dataset = IntentsAndSlots_Bert(test_raw, lang)
    
    return train_dataset, dev_dataset, test_dataset


"""
Prepare the dataset to be handled by the model, i.e. converts from text to numbers, len method, get method
"""

class IntentsAndSlots_Bert(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.tokenizer(self.utterances, padding=True)
        utterance_max_length = len(self.utt_ids['attention_mask'][0]) # they will have all the same length
        # First we align the slots, then we use the same lang.slot2id
        self.aligned_slots = self.align_slots(self.utterances, self.slots, utterance_max_length, 'pad')
        self.slot_ids = self.mapping_seq(self.aligned_slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)


    def __len__(self):
        return len(self.utterances)


    def __getitem__(self, idx):
        # We get the info about the utterance, the slots(which are aligned), and intent
        input_ids = torch.tensor(self.utt_ids['input_ids'][idx], dtype=torch.long)
        attention_mask = torch.tensor(self.utt_ids['attention_mask'][idx], dtype=torch.long)
        token_type_ids = torch.tensor(self.utt_ids['token_type_ids'][idx], dtype=torch.long)
        slots = torch.tensor(self.slot_ids[idx], dtype=torch.long)
        intent = self.intent_ids[idx]

        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'slots': slots,
            'intent': intent
        }
        return sample


    # Auxiliary methods
    def align_slots(self, utterances, slots, max_length, pad):
        # assuming len(utterances) == len(slots)
        aligned_slots_list = []
        counter = 0

        for utterance in utterances:

            aligned_slots = []
            aligned_slots.append(pad) # CLS
            tokenized_utterance = self.tokenizer(utterance)
            utterance = utterance.split()
            slots_current = slots[counter].split()

            for i in range(len(utterance)):
                token = self.tokenizer(utterance[i])
                token_length = len(token['input_ids'][1:-1])
                #print(f"token is {token['input_ids']}")
                aligned_slots.append(slots_current[i])
                if token_length > 1:
                    # We experienced subtokenization, we put the same exact slot for every subtoken
                    #print(f'the utterance is {self.tokenizer.convert_ids_to_tokens(tokenized_utterance["input_ids"])}')
                    aligned_slots.extend([pad] * (token_length-1))

            counter += 1

            aligned_slots.append(pad) # SEP


            if len(tokenized_utterance["input_ids"]) != len(aligned_slots):
                print(f'utterance: {" ".join(utterance)}, number {counter-1}, created problems of misalignement')
                exit()

            if len(aligned_slots) < max_length:
                difference = max_length - len(aligned_slots)
                aligned_slots.extend([pad] * difference) # padding

            aligned_slots_string = " ".join(aligned_slots)
            # since we want to use the same mapping_seq functions, we have to give
            # it the input as a list of strings
            aligned_slots_list.append(aligned_slots_string)

        return aligned_slots_list # list of strings


    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]


    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    print(f'il problema è {x}')
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res # returns a list of list of numbers...


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
    #for key in data[0].keys():
    #    new_item[key] = [d[key] for d in data]
    
    #new_item['input_ids'] = [element.detach() for element in new_item['input_ids']]
    #new_item['attention_mask'] = [element.detach() for element in new_item['attention_mask']]
    #new_item['token_type_ids'] = [element.detach() for element in new_item['token_type_ids']]
    #new_item['slots'] = [element.detach() for element in new_item['slots']]
    input_ids = torch.stack([d['input_ids'] for d in data]).to(device)
    attention_mask = torch.stack([d['attention_mask'] for d in data]).to(device)
    token_type_ids = torch.stack([d['token_type_ids'] for d in data]).to(device)
    y_slots = torch.stack([d['slots'] for d in data]).to(device)
    intents = torch.LongTensor([d['intent'] for d in data]).to(device)


    new_item["input_ids"] = input_ids
    new_item["attention_mask"] = attention_mask
    new_item["token_type_ids"] = token_type_ids
    new_item["slot_labels"] = y_slots
    new_item["intent_labels"] = intents

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
                if 'slot_out' in m_name or 'intent_out' in m_name:
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
