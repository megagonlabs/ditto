# Ditto for clustering and attribute scoring

To run a ditto model for clustering (command line) from the parent directory:

```
CUDA_VISIBLE_DEVICES=0 python clustering/demo.py \
  --task Textual/Abt-Buy \
  --input_path demo_data_abt_buy.jsonl \
  --output_path demo_data_abt_buy.jsonl.output \
  --lm roberta \
  --use_gpu \
  --fp16 \
  --checkpoint_path checkpoints/
```

Parameters:
* ``--task``: the name of the tasks
* ``--input_path``: the input jsonlines file containing the list of records to be clustered. See ``demo_data_abt_buy.jsonl``.
* ``--output_path``: the output file name
* ``--lm``: the language model (roberta by default)
* ``--use_gpu``: if set, then run the predictions on GPU(s)
* ``--fp16``: whether to run with fp16 optimization 
* ``--checkpoint_path``: the path to the model checkpoint (e.g., ``checkpoints/Textual/Abt-Buy.pt``)
* Other parameters like ``--batch_size``, ``--max_len``, ``--dk``, ``--summarize`` are also supported (See the section for ``matcher.py``)

The script performs clustering on the input file and creates a new attribute ``cluster_id`` indicating the cluster of each record.


## Flask API

### Clustering

We also provide a Flask API to run clustering:

To start the API:
```
export FLASK_APP=clustering/demo.py
CUDA_VISIBLE_DEVICES=0 python -m flask run
```

To query:
```
curl -d @clustering_query.json  -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/query
```

The input file ``clustering_query.json`` essentially contains the parameters of the command line scripts and the input file in the ``records`` field.

### Attribute scoring

We also provide an API for querying attribute scores (the importance of each attribute). For example,

```
curl -d @attr_score_query.json  -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/query
```
The input json file ``attr_score_query.json`` contains two records in fields ``left`` and ``right``. 
We run the model on: 
* the original (left, right) pair
* for every attribute, the (left, right) pair with the attribute dropped
* for every token appears in left or right, the (left, right) pair with the token dropped

By comparing the resulting label (which might be flipped) and confidence score (increase or decrease), we can get the importance of each feature (attribute or token).
