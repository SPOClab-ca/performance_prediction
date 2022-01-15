import argparse
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from data_utils import senteval_load_preprocessed, train_val_test_split
from utils import timed_func 


class Trainer:
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = \
            X_train, X_val, X_test, Y_train, Y_val, Y_test
            
    def probe(self):
        models = {
            "LogReg": LogisticRegression(),
            "MLP-10": MLPClassifier([10]),
            "MLP-20": MLPClassifier([20]),
            "RF-100": RandomForestClassifier(),
            "RF-10": RandomForestClassifier(10),
            "DecisionTree": DecisionTreeClassifier(),
            "SVM": SVC(probability=True)
        }
        report = {}
        
        for model_name in models:
            classifier = models[model_name]
            
            classifier.fit(self.X_train, self.Y_train)
            tr_pred = classifier.predict(self.X_train)
            tr_prob = classifier.predict_proba(self.X_train)
            tr_acc = accuracy_score(self.Y_train, tr_pred)
            tr_loss = log_loss(self.Y_train, tr_prob)
            
            val_pred = classifier.predict(self.X_val)
            val_prob = classifier.predict_proba(self.X_val)
            val_acc = accuracy_score(self.Y_val, val_pred)
            val_loss = log_loss(self.Y_val, val_prob)
            
            test_pred = classifier.predict(self.X_test)
            test_prob = classifier.predict_proba(self.X_test)
            test_acc = accuracy_score(self.Y_test, test_pred)
            test_loss = log_loss(self.Y_test, test_prob)

            result = {
                "train_acc": tr_acc, "train_loss": tr_loss, 
                "val_acc": val_acc, "val_loss": val_loss,
                "test_acc": test_acc, "test_loss": test_loss,
                "model": model_name
            }
            for k in result:
                if k not in report:
                    report[k] = [result[k]]
                else:
                    report[k].append(result[k])
        return pd.DataFrame(report)


def compare_three_conditions(all_data, nclasses, rs_list=[0,1,2,31,32767], train_size_per_class=1200):
    """
    Compare three settings: (A) use full neurons; (B) only those with MI=0; (C) only those with MI>0
    """
    agg_report = []
    for rs in tqdm(rs_list):

        all_labels = all_data['y'].astype(int)
        train_neurons, train_labels, val_neurons, val_labels, test_neurons, test_labels = train_val_test_split(
            all_data['X'], all_labels, nclasses, seed=rs, train_size_per_class=train_size_per_class, val_size_per_class=train_size_per_class/4)

        mis = np.array([mutual_info_classif(train_neurons[:, j:j+1], train_labels) for j in range(train_neurons.shape[1])]).reshape(-1)  # (D,)
        train_neurons_nonzero = train_neurons[:, mis>0]  # (N, D_nonzero)
        train_neurons_zeromi = train_neurons[:, mis==0]  # (N, D_zero)
        val_neurons_nonzero = val_neurons[:, mis>0]
        val_neurons_zeromi = val_neurons[:, mis==0]
        test_neurons_nonzero = test_neurons[:, mis>0]
        test_neurons_zeromi = test_neurons[:, mis==0]

        # Three classifiers
        configs = {
            "Full": Trainer(train_neurons, train_labels,
                val_neurons, val_labels,
                test_neurons, test_labels),
            "Nonzero": Trainer(train_neurons_nonzero, train_labels,
                val_neurons_nonzero, val_labels,
                test_neurons_nonzero, test_labels),
            "ZeroMI": Trainer(train_neurons_zeromi, train_labels, 
                val_neurons_zeromi, val_labels,
                test_neurons_zeromi, test_labels)
        }
        for c in configs:
            trainer = configs[c]
            report = trainer.probe()
            report["rs"] = [rs] * len(report)
            report["config"] = [c] * len(report)
            report["train_size_per_class"] = [train_size_per_class] * len(report)
            
            agg_report.append(report)
    return pd.concat(agg_report).reset_index(drop=True)


@timed_func
def main(args):
    all_report = []
    for layer in range(13):
        taskstr = f"{args.task}_layer_{layer}"
        all_data, nclasses = senteval_load_preprocessed("../data/senteval/embeddings_{}/{}".format(args.model_str, taskstr))
        print(f"Loaded preprocess data for task {taskstr}")
        report = compare_three_conditions(all_data, nclasses)
        report["task"] = [taskstr] * len(report)
        report["nclasses"] = [nclasses] * len(report)
        all_report.append(report)
    all_report_df = pd.concat(all_report).reset_index(drop=True)
    return all_report_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bigram_shift")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--corruption_step", type=int, default=0)
    args = parser.parse_args()
    
    model_str = "{}".format(args.model.replace("-", "_").replace("/", "_"))
    if args.corruption_step > 0:
        model_str += "_corr_{}".format(args.corruption_step)
    args.model_str = model_str
    print(args)

    report_folder = Path(f"../reports/embeddings_{args.model_str}")
    if not report_folder.exists():
        report_folder.mkdir()
    report_path = Path(report_folder, f"report_{args.task}.csv")
    if report_path.exists():
        print("Detected previous probing results. Skip it.")
    else:
        all_report_df = main(args)
        all_report_df.to_csv(report_path, index=False)
        print("Saved to {}".format(report_path))