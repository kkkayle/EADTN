import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve

def compute_metrics(Y, S, P):
    tn, fp, fn, tp = confusion_matrix(Y, P).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc_value = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    auprc_value = auc(fpr, tpr)
    return {
        'AUC': auc_value,
        'AUPRC': auprc_value,
        'Accuracy': accuracy,
        'Recall': recall,
        'Specificity': specificity,
        'Precision': precision
    }

def get_kfold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


class DTIDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

def cluster_collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [], []
    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    clusters_new = torch.zeros(N, dtype=torch.long)
    for i, pair in enumerate(batch_data):
        compoundstr, proteinstr, label,cluster =  pair[0], pair[1], pair[2],pair[-1]
        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        labels_new[i] = np.int32(label)
        clusters_new[i] = np.int32(cluster)
    return (compound_new, protein_new, labels_new,clusters_new)

def cluster_collate_fn_normal_train(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [], []
    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, pair in enumerate(batch_data):
        compoundstr, proteinstr, label,cluster =  pair[0], pair[1], pair[2],pair[-1]
        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        labels_new[i] = np.int32(label)
    return (compound_new, protein_new, labels_new)

def collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [], []
    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()
        compoundstr, proteinstr, label =  pair[0], pair[1], pair[2]
        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        labels_new[i] = np.int32(label)
    return (compound_new, protein_new, labels_new)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, num_n_fold=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath

    def __call__(self, score, model):

        if self.best_score == -np.inf:
            self.save_checkpoint(score, model)
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Have a new best checkpoint: ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.savepath +
                   '/valid_best_checkpoint.pth')
class PolyLoss(nn.Module):
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        one_hot = torch.zeros((16, 2), device=self.DEVICE).scatter_(
            1, torch.unsqueeze(labels, dim=-1), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)