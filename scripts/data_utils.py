from pathlib import Path 
import numpy as np
import pandas as pd


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