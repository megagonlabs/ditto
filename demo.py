import json
import jsonlines
import argparse
import sys
import numpy as np

from tqdm import tqdm
from scipy.special import softmax

sys.path.insert(0, "Snippext_public")
from copy import copy
from ditto.dataset import DittoDataset
from ditto.summarize import Summarizer
from ditto.knowledge import *
from matcher import to_str, classify, load_model
from sklearn.feature_extraction.text import TfidfVectorizer

def basic_blocking(records):
    """Basic blocking with TF-IDF

    Args:
        records (List of dict): the entity records

    Returns:
        List of tuples: the candidate pairs (3 pairs for each record)
    """
    docs = []
    for rec in records:
        docs.append(' '.join(rec.values()).lower())

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs).todense()

    N = len(records)
    result = []
    for rid1 in range(N):
        scores = []
        for rid2 in range(N):
            if rid1 == rid2:
                continue
            similarity = np.inner(vectors[rid1], vectors[rid2])
            scores.append((similarity, rid2))
        scores.sort(reverse=True)
        for _, rid2 in scores[:2]:
            result.append((rid1, rid2))
    return result


def get_connected_components(edges):
    """Compute the connected components given the EM pair predictions.

    Args:
        edges (list of tuples): the entity pairs of the form (left, right, lable)

    Returns:
        List of tuples: a list of (node, cluster_id)
    """
    parent = {}
    def get_parent(node):
        current = node
        to_update = []
        while parent[current] != current:
            to_update.append(current)
            current = parent[current]

        for up in to_update:
            parent[up] = current
        return current

    for t1, t2, lbl in edges:
        if t1 not in parent:
            parent[t1] = t1
        if t2 not in parent:
            parent[t2] = t2

        if int(lbl) == 1:
            p1 = get_parent(t1)
            p2 = get_parent(t2)
            parent[p1] = p2

    p2idx = {}
    result = []
    for item in parent:
        p = get_parent(item)
        if p not in p2idx:
            p2idx[p] = len(p2idx)
        result.append((item, p2idx[p]))

    return result


def process_batches(all_rows, all_pairs, model, batch_size, lm, max_len):
    """Make prediction in all the pairs and store in results in rows.
    """

    def process_batch(rows, pairs):
        try:
            predictions, logits = classify(pairs, config, model, lm=lm, max_len=max_len)
        except:
            # ignore the whole batch
            return
        scores = softmax(logits, axis=1)
        for row, pred, score in zip(rows, predictions, scores):
            # if score[1] >= 0.9:
            #     pred = 1
            # else:
            #     pred = 0
            row['match'] = pred
            row['match_confidence'] = score[int(pred)]

    # make predictions
    row_batch = []
    pair_batch = []
    for pair, row in tqdm(zip(all_pairs, all_rows), total=len(all_rows)):
        row_batch.append(row)
        pair_batch.append(pair)
        if len(pair_batch) == batch_size:
            process_batch(row_batch, pair_batch)
            pair_batch.clear()
            row_batch.clear()

    if len(pair_batch) > 0:
        process_batch(row_batch, pair_batch)

    return all_rows


def cluster(records,
            config, model,
            batch_size=512,
            summarizer=None,
            lm='roberta',
            max_len=256,
            dk_injector=None):
    """Run the model over the input file containing the candidate entry pairs

    Args:
        records (List of Dict): the input records
        config (Dict): the task configuration
        model (SnippextModel): the model for prediction
        batch_size (int): the batch size
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        List of Dict: the output records with cluster_id's
    """

    N = len(records)
    pairs = []
    rows = []
    # no blocking, get all pairs
    # for rid1 in range(N):
    #     for rid2 in range(rid1 + 1, N):
    #         pairs.append((to_str(records[rid1], summarizer, max_len, dk_injector),
    #                       to_str(records[rid2], summarizer, max_len, dk_injector)))
    #         rows.append({'left_id': rid1,
    #                      'left': records[rid1],
    #                      'right_id': rid2,
    #                      'right': records[rid2]})

    # basic blocking with TF-IDF
    candidate_pairs = basic_blocking(records)
    for rid1, rid2 in candidate_pairs:
        pairs.append((to_str(records[rid1], summarizer, max_len, dk_injector),
                      to_str(records[rid2], summarizer, max_len, dk_injector)))
        rows.append({'left_id': rid1,
                     'left': records[rid1],
                     'right_id': rid2,
                     'right': records[rid2]})

    # processing
    process_batches(rows, pairs, model, batch_size, lm, max_len)

    # get connected components
    edges = []
    for row in rows:
        edges.append((row['left_id'], row['right_id'], row['match']))
    components = get_connected_components(edges)
    components.sort(key=lambda x: x[1])
    return components


def predict_pair(left, right,
            config, model,
            batch_size=512,
            summarizer=None,
            lm='roberta',
            max_len=256,
            dk_injector=None):
    """Run the model over the input file containing the candidate entry pairs

    Args:
        left (Dict): the 1st record to be matched
        right (Dict): the 2nd record to be matched
        config (Dict): the task configuration
        model (SnippextModel): the model for prediction
        batch_size (int): the batch size
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        Dict: the output with the scores and feature scores
    """
    pairs = []
    features = []

    # original
    features.append({'name': 'original'})
    pairs.append((to_str(left, summarizer, max_len, dk_injector),
                  to_str(right, summarizer, max_len, dk_injector)))

    # droping an attribute
    for attr in left.keys():
        left_copy = copy(left)
        right_copy = copy(right)
        if attr in left_copy:
            del left_copy[attr]
        if attr in right_copy:
            del right_copy[attr]
        features.append({'name': 'attr:' + attr})
        pairs.append((to_str(left_copy, summarizer, max_len, dk_injector),
                      to_str(right_copy, summarizer, max_len, dk_injector)))

    # dropping a keyword
    left_str = to_str(left, summarizer, max_len, dk_injector)
    right_str = to_str(right, summarizer, max_len, dk_injector)
    left_tokens = left_str.split(' ')
    right_tokens = right_str.split(' ')
    all_tokens = left_tokens + right_tokens
    all_tokens = list(set([t.lower() for t in all_tokens]))
    all_tokens.sort()

    for token in all_tokens:
        if token in ['col', 'val']:
            continue
        new_left = [t for t in left_tokens if t.lower() != token]
        new_right = [t for t in right_tokens if t.lower() != token]
        features.append({'name': 'token:' + token})
        pairs.append((' '.join(new_left), ' '.join(new_right)))

    # predict
    process_batches(features, pairs, model, batch_size, lm, max_len)

    # return
    result = features[0]
    result['features'] = features[1:]
    return result


from flask import Flask
from flask import jsonify, request

app = Flask(__name__)

config = None
model = None

@app.route('/query', methods=['POST'])
def query():
    query_request = request.json

    # load variables
    query_type = query_request['query_type']
    task = query_request.get('task', 'Structured/iTunes-Amazon')
    lm = query_request.get('lm', 'roberta')
    use_gpu = query_request.get('use_gpu', True)
    fp16 = query_request.get('fp16', True)
    checkpoint_path = 'checkpoints/'
    max_len = query_request.get('max_len', 256)
    dk = query_request.get('dk', None)
    summarize = query_request.get('summarize', False)

    global config
    global model

    # reload the model for the first time
    if config is None or config['name'] != task:
        config, model = load_model(task, checkpoint_path,
                                   lm, use_gpu, fp16)

    summarizer = dk_injector = None
    if summarize:
        summarizer = Summarizer(config, lm)

    if dk is not None:
        if 'product' in dk:
            dk_injector = ProductDKInjector(config, dk)
        else:
            dk_injector = GeneralDKInjector(config, dk)

    if query_type == 'cluster':
        records = query_request['records']
        result = cluster(records,
                         config, model,
                         summarizer=summarizer,
                         max_len=max_len,
                         lm=lm,
                         dk_injector=dk_injector)
    elif query_type == 'predict':
        left = query_request['left']
        right = query_request['right']
        result = predict_pair(left, right,
                         config, model,
                         summarizer=summarizer,
                         max_len=max_len,
                         lm=lm,
                         dk_injector=dk_injector)
    else:
        raise ValueError('query type not found')

    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='Structured/iTunes-Amazon')
    parser.add_argument("--input_path", type=str, default='demo_data.jsonl')
    parser.add_argument("--output_path", type=str, default='demo_data.jsonl.output')
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/')
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--max_len", type=int, default=256)
    hp = parser.parse_args()

    # load the models
    config, model = load_model(hp.task, hp.checkpoint_path,
                               hp.lm, hp.use_gpu, hp.fp16)

    summarizer = dk_injector = None
    if hp.summarize:
        summarizer = Summarizer(config, hp.lm)

    if hp.dk is not None:
        if 'product' in hp.dk:
            dk_injector = ProductDKInjector(config, hp.dk)
        else:
            dk_injector = GeneralDKInjector(config, hp.dk)

    # reading the entity records
    records = []
    with jsonlines.open(hp.input_path) as reader:
        for rec in reader:
            if 'cluster_id' in rec:
                del rec['cluster_id']
            records.append(rec)

    # run clustering
    output = cluster(records, config, model,
            summarizer=summarizer,
            max_len=hp.max_len,
            lm=hp.lm,
            dk_injector=dk_injector)

    # dump the output
    with jsonlines.open(hp.output_path, mode='w') as writer:
        for cc in output:
            rec = records[cc[0]]
            rec['cluster_id'] = cc[1]
            writer.write(rec)

    # run pair prediction
    left = records[output[0][0]]
    right = records[output[1][0]]
    output = predict_pair(left, right,
            config, model,
            summarizer=summarizer,
            max_len=hp.max_len,
            lm=hp.lm,
            dk_injector=dk_injector)

    json.dump(output, open('pair.json', 'w'))
