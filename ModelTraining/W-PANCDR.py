import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import pandas as pd
import itertools
israndom=False
from itertools import cycle
from sklearn.model_selection import StratifiedKFold,KFold

from ModelTraining.model import Encoder, GCN, ADV
device = torch.device('cuda')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False