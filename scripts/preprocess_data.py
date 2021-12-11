import os 
import glob
import sys
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "../libs/InferSent")
from data_utils import senteval_load_file

MAXLEN=512
MAX_LAYER=13

class Preprocessor(object):
    def __init__(self, skip_existing=False) -> None:
        super().__init__()
        self.skip_existing = skip_existing
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    
class SentEvalPreprocessor(Preprocessor):
    data_path = "../data/senteval/"
    file_path = data_path + "{}.txt"

    def __init__(self, skip_existing, model) -> None:
        super().__init__(skip_existing=skip_existing)

        tasks = []
        for dpath in glob.glob(self.file_path.format("*")):
            task_name = os.path.basename(dpath).split(".")[0]
            tasks.append(task_name)
        self.tasks = tasks
        
        self.modelname = {
            "bert": "bert-base-multilingual-cased",
            "roberta": "xlm-roberta-base"
            }[model]
        self.modelname_short = model
        self.bert, self.tokenizer = self.prepare_model()
        print("Prepared model: {}".format(self.modelname))

    def preprocess_all_tasks_w_bert(self):
        for task in self.tasks:
            print (f"Task: {task}")
            self.preprocess_w_bert(task)

    def prepare_model(self):
        model = AutoModel.from_pretrained(self.modelname, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(self.modelname) 
        model.to(self.device)
        model.eval()
        return model, tokenizer 
    
    def preprocess_w_bert(self, task, bsize=10):
        save_path = os.path.join(self.data_path, "{}.{}".format(task, self.modelname_short))
        if self.skip_existing and os.path.exists(save_path+"_pooler"):
            print(f"Preprocessed file found at {save_path}. Skip it.")
            return 

        x_list_by_layer = [[] for layer in range(MAX_LAYER)]
        x_pooler_list = []
        y_list = []
        data, n_classes = senteval_load_file(filepath=self.file_path.format(task))
        for i in tqdm(range(0, len(data), bsize)):
            batch_text = [d['X'] for d in data[i:i+bsize]]
            batch_inputs = self.tokenizer(batch_text,
                                        truncation=True,
                                        max_length=MAXLEN,
                                        padding='max_length',
                                        return_tensors="pt")
            batch_inputs = {k:v.to(self.device) for k,v in batch_inputs.items()}
            with torch.no_grad():
                all_output = self.bert(**batch_inputs)
                all_output = all_output
                pooler_output = all_output.pooler_output.cpu()  # (bsz, D)
                hidden_output = [t[:, 0, :].cpu() for t in all_output.hidden_states]  # List (len=n_layer) of (bsz, D)
                for layer in range(MAX_LAYER):
                    x_list_by_layer[layer].append(hidden_output[layer])  # (bsz, D)
                x_pooler_list.append(pooler_output)
                y_list += [d['y'] for d in data[i:i+bsize]]

        y_list = np.array(y_list, dtype=np.int16)
        for layer in range(MAX_LAYER):
            x_tensor = torch.cat(x_list_by_layer[layer], dim=0).numpy()  # (N, D)
            torch.save({'X': x_tensor, 'y': y_list}, save_path+f"_layer_{layer}")
        x_pooler_tensor = torch.cat(x_pooler_list, dim=0).numpy()  # (N, D)
        torch.save({'X': x_pooler_tensor, 'y': y_list}, save_path+f"_pooler")
        print(f"Saved models to {save_path}")


if __name__ == "__main__":
    print("Starting preprocessing")
    preprocessor = SentEvalPreprocessor(skip_existing=True, model="roberta") 
    preprocessor.preprocess_all_tasks_w_bert()
