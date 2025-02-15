
from sklearn.metrics import classification_report
import torch
from transformers import BertTokenizer, BertModel
from evals import evaluate_ote
import numpy as np

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        aspects = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
    
        loss_aspects = criterion(aspects, sample['aspect_labels'])
        
        loss_aspects.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss_array.append(loss_aspects.item())
    return loss_array


def eval_loop(data, model):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()
    precision = []
    recall = []
    f1_score = []
    with torch.no_grad():
        for sample in data:
            ref_aspects = []
            hyp_aspects = []
            aspects = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])

            output_aspects = torch.argmax(aspects, dim=1)

            for id_seq, seq in enumerate(output_aspects):

                length = sample['attention_mask'][id_seq].sum().item()
                #review = tokenizer.convert_ids_to_tokens(sample['input_ids'][id_seq])
                
                gt_ids = sample['aspect_labels'][id_seq][:length].tolist()
                
                #gt_ids = gt_ids[1:-1] # we put pad at the beggining and end = [CLS] and [SEP]
                
                to_decode = seq[:length].tolist()
                #to_decode = to_decode[1:-1]
                
                indexes_padding = set()

                gt_seq = []
                for id_el, elem in enumerate(gt_ids):
                    if elem != 0: # 0 indicates [PAD] -> we inserted it to deal with sub-tokens
                        gt_seq.append(elem)
                    else:
                        indexes_padding.add(id_el)
                ref_aspects.append(gt_seq)
                #print(f'gt_seq {gt_seq}')
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    if id_el not in indexes_padding:
                        tmp_seq.append(elem)
                hyp_aspects.append(tmp_seq)
                #print(f'tmp_seq {tmp_seq}\n')
            precision_batch, recall_batch, f1_score_batch = evaluate_ote(ref_aspects, hyp_aspects)
            precision.append(precision_batch)
            recall.append(recall_batch)
            f1_score.append(f1_score_batch)
        return (precision, recall, f1_score)

