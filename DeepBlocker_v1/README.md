# DeepBlocker

DeepBlocker is a Python package for performing blocking for entity matching using deep learning. It provides functionalities for transforming tuples into embeddings customized for blocking. Given these tuple embeddings, DeepBlocker also provides various utilities to retrieve similar tuples and construct the candidate set efficiently. DeepBlocker is self-supervised and does not require any labeled data. DeepBlocker provides multiple instantiations for tuple embedding  and vector pairing for performing blocking. It is also modular and easily customizable. Each of the subcomponent is based on a pre-defined and intuitive API that allows altering and swapping up these components to achieve bespoke implementations.  

# Paper and Data

For details on the architecture of the models used, take a look at our paper
[Deep Learning for Blocking in Entity Matching: A Design Space Exploration (VLDB '21)](http://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf).

All public datasets used in the paper can be downloaded from the [datasets page](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md).

We used fastText for word embedding that is [pre-trained on Wikipedia](https://fasttext.cc/docs/en/pretrained-vectors.html).


# Quick Start: DeepBlocker in 30 seconds

There are four main steps in using DeepBlocker:

1. Load the relevant libraries

```python
import pandas as pd
from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
```

2. Data processing: Load the relevant datasets for blocking.

```python
left_df = pd.read_csv("left_table_csv_file_name")
right_df = pd.read_csv("right_table_csv_file_name")
```

3. Instantiate the DeepBlocker with appropriate classes for tuple embedding and vector pairing models.

```python
tuple_embedding_model = AutoEncoderTupleEmbedding()
topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
```

4. Train the models and perform blocking of the tables. Report the accuracy.

```python
candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block)
golden_df = pd.read_csv("matches_csv_file_name")
print(blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df))
```


# Installation

DeepBlocker relies on a set of external libraries that can be found in [requirements.txt](requirements.txt).
You can install them as
```
pip install -r requirements.txt
```

# Tutorials

We provide a [sample script](main.py) illustrating how DeepBlocker works for three major tuple embedding models -- AutoEncoder, CTT and Hybrid.




# Support

Please contact Saravanan Thirumuruganathan for any questions about the code.

# The Team

DeepBlocker was developed by QCRI and University of Wisconsin-Madison.
For the entire list of contributors please refer the [DeepBlocker paper](http://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf).
