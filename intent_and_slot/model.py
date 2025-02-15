import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from transformers import BertTokenizer, BertModel
from pprint import pprint

"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model
"""



class JointBert(nn.Module):

    def __init__(self, args, out_slot, out_int):
        super(JointBert, self).__init__()

        # out_slot = number of slots (output size for slot filling) ## all the possible slots
        # out_int = number of intents (output size for intent class) ## all the possible intents

        self.args = args
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        hidden_size = self.bert.config.hidden_size
        self.slot_out = nn.Linear(hidden_size, out_slot)
        self.intent_out = nn.Linear(hidden_size, out_int)
        self.dropout = nn.Dropout()
        
 
        
    def forward(self, input_ids, attention_mask, token_types_ids):

        outputs = self.bert(input_ids, attention_mask, token_types_ids)

        last_hidden_state = outputs.last_hidden_state
        pooled = outputs.pooler_output

        last_hidden_state = self.dropout(last_hidden_state)
        pooled = self.dropout(pooled)
        
        # Compute slot logits
        slots = self.slot_out(last_hidden_state) ## torch.Size([5, 15, 768])
        # Compute intent logits
        intent = self.intent_out(pooled) ## torch.Size([5, 768])
        
        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        
        return slots, intent