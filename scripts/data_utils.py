from pathlib import Path 
import numpy as np
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def senteval_load_preprocessed(fpath):
    """
    Assume the encodings are cached by preprocess_data.py
    E.g., at ../data/senteval/bigram_shift.bert
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError(f'{fpath} not exists.')
    data = torch.load(fpath)
    nclasses = np.unique(data['y']).shape[0]    
    return data, nclasses


def train_val_test_split(data_x, data_y, nclasses, seed, train_size_per_class, val_size_per_class):
    assert train_size_per_class * nclasses + val_size_per_class * nclasses < len(data_y), "train and val sizes should add up to be no more than the total num in the data!"

    train_x, other_x, train_y, other_y = train_test_split(
        data_x, data_y,
        random_state=seed, 
        train_size=int(train_size_per_class * nclasses), 
        shuffle=True,
        stratify=data_y
    )
    val_x, remain_x, val_y, remain_y = train_test_split(
        other_x, other_y,
        random_state=seed,
        train_size=int(val_size_per_class * nclasses),
        shuffle=True,
        stratify=other_y
    )
    test_x, _, test_y, _ = train_test_split(
        remain_x, remain_y,
        random_state=seed,
        train_size=int(val_size_per_class * nclasses),
        shuffle=True,
        stratify=remain_y
    )
    return np.array(train_x), np.array(train_y), \
        np.array(val_x), np.array(val_y), \
        np.array(test_x), np.array(test_y) 

    
def senteval_load_file(filepath):
    """
    Input:
        filepath. e.g., "<repo_dir>/data/senteval/bigram_shift.txt"
    Return: 
        task_data: list of {'X': str, 'y': int}
        nclasses: int
    """
    # Just load all portions, and then do train/dev/test splitting myself
    tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
    task_data=[]
    
    for linestr in Path(filepath).open().readlines():
        line = linestr.rstrip().split("\t")
        task_data.append({
            'X': line[-1], 'y': line[1]
        })

    # Convert labels str to int
    all_labels = [item['y'] for item in task_data]
    labels = sorted(np.unique(all_labels))
    tok2label = dict(zip(labels, range(len(labels))))
    nclasses = len(tok2label) 
    for i, item in enumerate(task_data):
        item['y'] = tok2label[item['y']]
    
    return task_data, nclasses

def stratify_sample(all_data, all_labels, n_per_class, rs=0):
    df = pd.DataFrame({
      "data": all_data,
      "label": all_labels
    })
    df_sampled = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n_per_class, random_state=rs))
    sampled_data = [row['data'] for i, row in df_sampled.iterrows()]
    return sampled_data, df_sampled
