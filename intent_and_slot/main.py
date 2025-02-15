import argparse
from functions import *
from utils import *
from model import *
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from collections import Counter
from transformers import BertTokenizer
import torch.nn as nn
import logging
"""
# Configure the logging
logging.basicConfig(
    filename='intent_and_slot_bert.log',  # Log file name
    level=logging.INFO,         # Logging level
    format='%(message)s'  # Log format
)
"""



# GLOBAL VARIABLES
device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0 # Bert also maps the '[PAD]' to 0, tokenizer.pad_token_id == 0

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def main(args):

    print(f'main launched with {args}')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    n_epochs = args.epochs
    patience = args.patience
    learning_rate = args.learning_rate
    clip = args.clip
    alpha = args.alpha
    runs = args.runs
    batch_size = args.batch_size

    train_loader, dev_loader, test_loader, lang = get_dataloaders(batch_size)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)


    slot_f1s, intent_acc = [], []

    for x in tqdm(range(0, runs)):
        model = JointBert(args, out_slot, out_int).to(device)
        model.apply(init_weights) # changed the init to be applied only on our nn.Linear

        optimizers = {'Adam': optim.Adam, 'AdamW': optim.AdamW, 'sgd': optim.SGD}

        optimizer = optimizers[args.optimizer](model.parameters(), lr=learning_rate)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)#tokenizer.pad_token_id)
        criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        last_epoch = -1
        best_model = None
        


        for epoch in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                              criterion_intents, model, alpha, clip=clip)
            if epoch % 3 == 0: # We check the performance every 3 epochs
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, model, lang)
                #losses_dev.append(np.asarray(loss_dev).mean())

                f1 = results_dev['total']['f']
                print(f'\nf1 is {f1}')
                # For decreasing the patience you can also use the average between slot f1 and intent accuracy
                if f1 > best_f1:
                    best_f1 = f1
                    # Here you should save the model
                    best_model = copy.deepcopy(model).to('cpu')
                    last_epoch = epoch
                    patience = args.patience
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean
        model_name = 'model'
        dict_args = vars(args)
        for element in dict_args:
            model_name += '_'
            model_name += element
            model_name += '='
            model_name += str(dict_args[element])
        model_name += '.pt'
        PATH = os.path.join("bin", model_name)
        saving_object = {"epoch": last_epoch,
                         "model": best_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         }
        torch.save(saving_object, PATH)

        best_model.to(device)
        results_test, intent_test, _ = eval_loop(test_loader, best_model, lang)  
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))
    #logging.info(f"{args} --> f1:  {round(slot_f1s.mean(),3)} +- {round(slot_f1s.std(),3)}   ---   acc:  {round(intent_acc.mean(), 3)} +- {round(slot_f1s.std(), 3)}")

    return 0



if __name__ == "__main__":
     # Create the parser
    parser = argparse.ArgumentParser(description='PyTorch BERT Slot filling & Intent classification')
    
    # Add arguments
    parser.add_argument('--learning_rate', default="0.0001", type=float, help='learning rate of the model')
    parser.add_argument('--batch_size', default="64", type=int, help='batch size of training')
    parser.add_argument('--emb_size', default="512", type=int, help='size of the embedded layer')
    parser.add_argument('--hid_size', default="512", type=int, help='size of the hidden layer')
    parser.add_argument('--clip', default="1", type=int, help='clip the gradient')
    parser.add_argument('--patience', default="3", type=int, help='patience of the model')
    parser.add_argument('--epochs', default="200", type=int, help='number of epochs')
    parser.add_argument('--optimizer', default="Adam", help="select 'Adam' or 'sgd' or 'AdamW'")
    parser.add_argument('--bidir', default=False, type=str2bool, help="LSTM with bidirectionality or not")
    parser.add_argument('--drop', default=0, type=float, help="apply dropout rate, 0 = no dropout")
    parser.add_argument('--alpha', default="0.5", type=float, help='alpha parameter for the combined loss')
    parser.add_argument('--runs', default="3", type=int, help='number of runs')
    
    # Parse the arguments
    args = parser.parse_args()
    main(args)