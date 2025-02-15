import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from transformers import BertTokenizer, BertModel
from pprint import pprint

"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model
"""



class JointBertABSA(nn.Module):

    def __init__(self, args):
        super(JointBertABSA, self).__init__()

        # we need to predict start and end of aspects
        # no polarity for now

        self.args = args
        # Bert as backbone
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        # Idea, like the paper, try to implement something like
        # target_star: 3,11 target_end: 4,11 
        #### easier way -> predict start, end or none.

        self.aspects_out = nn.Linear(hidden_size, 3)
        self.dropout = nn.Dropout(self.args.drop)

 

    def forward(self, input_ids, attention_mask, token_types_ids):

        outputs = self.bert(input_ids, attention_mask, token_types_ids)

        last_hidden_state = outputs.last_hidden_state
        
        # Compute aspect logits
        last_hidden_state = self.dropout(last_hidden_state)
        aspects_out = self.aspects_out(last_hidden_state)

        aspects_out = aspects_out.permute(0,2,1)
        
        return aspects_out