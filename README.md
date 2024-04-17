# How to run DittoLITE experiments (instuctions): 

1. Ensure you are on branch "modify-model"
2. Install the required packages into a conda environment
3. DeepBlocker is included as a submodule inside this repo. Ensure the submodule is initialized.
4. Inside DeepBlocker directory, create the following folders: embedding, data.
5. Inside embedding, download FastText model embeddings and choose the model 'wiki.en.bin'. [Link](https://fasttext.cc/docs/en/pretrained-vectors.html)
6. Inside data, download the datasets from [here](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) using the 'Browse' option for each dataset.
7. Use create_matches_csv.py script to create the matches.csv file for each dataset. Conceptually, this is done by concatenating train, val, test csvs, retaining only the rows with label 1 and removing the label column. 


## Requirements

* Python 3.7.7
* PyTorch 1.9
* HuggingFace Transformers 4.9.2
* Spacy with the ``en_core_web_lg`` models
* NVIDIA Apex (fp16 training)

Install required packages
```
conda install -c conda-forge nvidia-apex
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Training DeepBlocker models
In DeepBlocker_v1/main.py file, change folder_root and cols_to_block and run to train the blocker model. The blocking metrics: Recall, CSSR and runtimes (time elapsed) will be printed out at the end of training. 

## Training the Ditto Distill-BERT Blocker
```
cd {PROJECT_ROOT}/ditto/blocker
```
In train_blocker.py file, adjust train_fn and valid_fn as needed.
python train_blocker.py

## Measure DITTO Distill-BERT Blocker performance
```
cd {PROJECT_ROOT}/ditto/blocking
```
Below is a sample script run. Change the path to datasets accordingly. This will print out metrics: Recall, CSSR and runtime (time elapsed).
```
python blocker.py --model_fn distilbert-base-uncased --left_fn /home/sridhar/ditto/DeepBlocker_v1/data/Dirty/iTunes-Amazon/tableA.csv --right_fn /home/sridhar/ditto/DeepBlocker_v1/data/Dirty/iTunes-Amazon/tableB.csv --golden_fn /home/sridhar/ditto/DeepBlocker_v1/data/Dirty/iTunes-Amazon/matches.csv
```

## Training with Ditto
To train the matching model with Ditto, adjust task parameter according to the dataset used. Adjust lm parameter and include one among these values: ['roberta-base','prajjwal1
/
bert-mini','prajjwal1
/
bert-tiny']
```
python train_ditto.py \
  --task Structured/Beer \
  --n_epochs 50 \
  --lm roberta-base \
```

The meaning of the flags:
* ``--task``: the name of the tasks (see ``configs.json``)
* ``--lm``: the language model.
* ``--save_model``: if this flag is on, then save the checkpoint to ``{logdir}/{task}/model.pt``