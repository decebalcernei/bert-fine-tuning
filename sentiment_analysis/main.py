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


# GLOBAL VARIABLES
device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0 # Bert also maps the '[PAD]' to 0

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def main(args):

    print(f'main launched with {args}')

    n_epochs = args.epochs
    patience = args.patience
    learning_rate = args.learning_rate
    clip = args.clip
    runs = args.runs
    batch_size = args.batch_size

    best_f1 = 0
    last_epoch = -1
    best_model = None

    train_loader, dev_loader, test_loader = get_dataloaders(batch_size)


    model = JointBertABSA(args).to(device)
    model.apply(init_weights) # changed the init to be applied only on our nn.Linear
    optimizers = {'Adam': optim.Adam, 'AdamW': optim.AdamW, 'sgd': optim.SGD}
    optimizer = optimizers[args.optimizer](model.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)#tokenizer.pad_token_id)
    #precision, recall, f1_score = eval_loop(test_loader, model)
    for epoch in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion, model, clip=clip)
        #print(f'loss {epoch} is {np.asarray(loss).mean()}')
        if epoch % 3 == 0: # We check the performance every 3 epochs
            precision, recall, f1_score = eval_loop(dev_loader, model)
            f1 = np.mean(f1_score)
            print(f"precision: {np.mean(precision):.2f}\t recall: {np.mean(recall):.2f}\t f1_score: {np.mean(f1_score):.2f}")
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
    precision, recall, f1_score = eval_loop(test_loader, best_model)
    
    print(f"precision: {np.mean(precision):.2f}\t recall: {np.mean(recall):.2f}\t f1_score: {np.mean(f1_score):.2f}")


    return 0



if __name__ == "__main__":
     # Create the parser
    parser = argparse.ArgumentParser(description='PyTorch Joint Bert ABSA')
    
    # Add arguments
    parser.add_argument('--learning_rate', default="5e-5", type=float, help='learning rate of the model')
    parser.add_argument('--batch_size', default="64", type=int, help='batch size of training')
    parser.add_argument('--clip', default="1", type=int, help='clip the gradient')
    parser.add_argument('--patience', default="3", type=int, help='patience of the model')
    parser.add_argument('--epochs', default="20", type=int, help='number of epochs')
    parser.add_argument('--optimizer', default="Adam", help="select 'Adam' or 'sgd' or 'AdamW'")
    parser.add_argument('--drop', default=0, type=float, help="apply dropout rate, 0 = no dropout")
    parser.add_argument('--runs', default="1", type=int, help='number of runs')
    
    # Parse the arguments
    args = parser.parse_args()
    main(args)