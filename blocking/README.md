# The optional SentenceBERT fine-tuning for advanced blocking

We leverage the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) library to fine-tune the LMs for entity record representation. 

## Train the advanced blocking model

The following command fine-tunes the BERT model on an entity pair dataset to generate vector representations of entity data entries:
```
CUDA_VISIBLE_DEVICES=0 python train_blocker.py \
  --train_fn ../data/er_magellan/Structured/Beer/train.txt \
  --valid_fn ../data/er_magellan/Structured/Beer/valid.txt \
  --model_fn model.pth \
  --batch_size 64 \
  --n_epochs 40 \
  --lm bert \
  --fp16
```

Parameters:
* ``--train_fn``: the training dataset (serialized)
* ``--valid_fn``: the validation dataset (serialized)
* ``--model_fn``: the path to the output model (see sentence-transformers)
* ``--batch_size``, ``--n_epochs``, ``--lm``: batch size, number of epochs, the language model
* ``--fp16``: whether to train with fp16 accelaration

## Run the blocking model

To run the trained blocking model:
```
CUDA_VISIBLE_DEVICES=0 python blocker.py \
  --input_path input/ \
  --left_fn table_a.txt \
  --right_fn table_b.txt \
  --output_fn candidates.jsonl \
  --model_fn model.pth \
  --k 10
```
where
* ``--input_path``, ``left_fn``, ``right_fn`` are the path to the data directory containing two files, ``left_fn`` and ``right_fn``. The two files are serialized and contain one entry per line
* ``--output_fn``: the output file in jsonline format
* ``--model_fn``: the trained model
* ``--k`` (optional): if this parameter is set, then the candidates will be the top-k most similar entries for each row in ``right_fn``
* ``--threshold`` (optional): if this parameter is set, then the candidates will be all entry pairs of similarity above the threshold
