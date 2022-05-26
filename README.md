# Predicting the Fine-Tuning Performance with Probing

Repository structure:  
    `data`: where the data are placed. Here we place the SentEval txt files at `data/senteval/*.txt`  
    `scripts`: where the codes for the experiments are located. `script_run_*.sh` contain the scripts to run the experiments used in our paper. The most important three scripts are: (1) `glue_classify.py` (this can be launched by `script_run_classify.sh`), which is the fine-tuning classification experiment. (2) `preprocess_data.py` (this can be launched by `script_run_preprocess.sh`), which prepares the embeddings for probing. Note that this cacheing step is necessary, so that we do not need to rerun through the Transformer embeddings steps when using other probing classifiers. (3) `probing.py` (launched by `script_run_probing.sh`), the probing scripts.    
    `notebooks`: some subsequent analyses 