from conll import evaluate
from sklearn.metrics import classification_report
import torch
from transformers import BertTokenizer, BertModel

import numpy as np


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, alpha, clip=5):
    model.train()
    loss_array = []
    for sample in data:

        optimizer.zero_grad() # Zeroing the gradient
        slots, intents = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])

        loss_intent = criterion_intents(intents, sample['intent_labels'])
        loss_slot = criterion_slots(slots, sample['slot_labels'])
        loss = (1-alpha) * loss_intent + alpha * loss_slot # In joint training we sum the losses.
                                       # Is there another way to do that?
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

        loss_array.append(loss.item())

    return loss_array
  

def eval_loop(data, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            
            slots, intents = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])


            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intent_labels'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            # Slot inference
            output_slots = torch.argmax(slots, dim=1)

            for id_seq, seq in enumerate(output_slots):
                length = sample['attention_mask'][id_seq].sum().item()
                utterance = tokenizer.convert_ids_to_tokens(sample['input_ids'][id_seq][:length])
                
                gt_ids = sample['slot_labels'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                    
                utterance = utterance[1:-1] # Remove [CLS] and [SEP]
                gt_slots = gt_slots[1:-1] # we put pad at the beggining and end = [CLS] and [SEP]
                to_decode = seq[:length].tolist()
                to_decode = to_decode[1:-1]

                indexes_padding = set()

                gt_seq = []
                for id_el, elem in enumerate(gt_slots):
                    if elem != 'pad':
                        gt_seq.append((utterance[id_el], elem))
                    else:
                        indexes_padding.add(id_el)
                ref_slots.append(gt_seq)


                #ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    if id_el not in indexes_padding:
                        tmp_seq.append((utterance[id_el], lang.id2slot[elem]))

                hyp_slots.append(tmp_seq)


    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array
