import argparse
import numpy as np
import random
import torch
from sklearn.metrics import f1_score
BATCH_SIZE = 20
EPOCHS = 15
LR = 2e-5
SEED = 17
MAX_SEQ_LEN = 512

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='train path')
    parser.add_argument('--dev_path', type=str, required=True, help='dev path')
    parser.add_argument('--test_path', type=str, required=True, help='test path')
    parser.add_argument('--best_model_path', type=str, help='best model path')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--max_seq_len', type=int, default=512, help='max sequence length')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda')
    parser.add_argument('--temp', type=float, default=0.8, help='temp')
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    return parser.parse_args()

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def colorprint(text, color):
    color_dict = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'grey': '\033[98m',
        'black': '\033[99m',
        'reset': '\033[0m',
    }
    print(color_dict[color] + text + color_dict['reset'])

def compute_metric(label, pred, average='macro', f1_average=None):
    f1_average = average if f1_average is None else f1_average
    f1 = round(f1_score(label, pred, average=f1_average, zero_division=1), 4)
    return f1