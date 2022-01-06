import argparse
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

    def __init__(self, skip_existing, modelname, corruption_step) -> None:
        super().__init__(skip_existing=skip_existing)

        # Prepare texts
        tasks = []
        for dpath in glob.glob(self.file_path.format("*")):
            task_name = os.path.basename(dpath).split(".")[0]
            tasks.append(task_name)
        self.tasks = tasks
        
        # Prepare model
        self.model, self.tokenizer = self.prepare_model(modelname, corruption_step)

        # Prepare directory to save embedding
        save_path = Path(self.data_path, "embeddings_{}".format(self.modelname.replace("-", "_").replace("/", "_")))
        if save_path.exists():
            print(f"Preprocessed directory found at {save_path}.")
        else:
            save_path.mkdir(parents=True)
        self.save_path = save_path

    def preprocess_all_tasks_w_transformer(self):
        for task in self.tasks:
            print (f"Task: {task}")
            self.preprocess_w_transformer(task)

    def prepare_model(self, modelname, corruption_step):
        if corruption_step == 0:
            modelpath = modelname  # For loading model 
            self.modelname = modelname  # For saving the SentEval embeddings
        else:
            modelpath = "../data/corrupted_{}_checkpoints/checkpoint-{}".format(modelname.replace("-", "_").replace("/", "_"), corruption_step)
            self.modelname = "{}_corr_{}".format(modelname, corruption_step)
        model = AutoModel.from_pretrained(modelpath, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(modelpath)
        model.to(self.device)
        model.eval()
        print("Loaded model and tokenizer from {}".format(modelpath))
        return model, tokenizer 
    
    def preprocess_w_transformer(self, task, bsize=10):
        if self.skip_existing and Path(self.save_path, f"{task}_layer_1").exists():
            print(f"Found existing embedding for task {task}. Skipping...")
        x_list_by_layer = [[] for layer in range(MAX_LAYER)]
        #x_pooler_list = []
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
                all_output = self.model(**batch_inputs)
                all_output = all_output
                #pooler_output = all_output.pooler_output.cpu()  # (bsz, D)
                hidden_output = [t[:, 0, :].cpu() for t in all_output.hidden_states]  # List (len=n_layer) of (bsz, D)
                for layer in range(MAX_LAYER):
                    x_list_by_layer[layer].append(hidden_output[layer])  # (bsz, D)
                #x_pooler_list.append(pooler_output)
                y_list += [d['y'] for d in data[i:i+bsize]]

        y_list = np.array(y_list, dtype=np.int16)
        for layer in range(MAX_LAYER):
            x_tensor = torch.cat(x_list_by_layer[layer], dim=0).numpy()  # (N, D)
            torch.save(
                {'X': x_tensor, 'y': y_list}, 
                Path(self.save_path, f"{task}_layer_{layer}")
            )
        #x_pooler_tensor = torch.cat(x_pooler_list, dim=0).numpy()  # (N, D)
        #torch.save(
        #    {'X': x_pooler_tensor, 'y': y_list}, 
        #    Path(self.save_path, f"{task}_pooler")
        #)
        print(f"Saved task {task} embeddings to {self.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="roberta-base")
    parser.add_argument("--corruption_step", type=int, default=0) 
    args = parser.parse_args() 
    print(args) 

    preprocessor = SentEvalPreprocessor(True, args.modelname, args.corruption_step) 
    preprocessor.preprocess_all_tasks_w_transformer()
