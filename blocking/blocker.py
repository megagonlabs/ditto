import os
import sys
import jsonlines
import pickle
import numpy as np
import argparse

from tqdm import tqdm

sys.path.append("sentence-transformers")

from sentence_transformers import SentenceTransformer

def encode_all(path, input_fn, model, overwrite=False):
    """Encode a collection of entries and output to a file

    Args:
        path (str): the input path
        input_fn (str): the file of the serialzied entries
        model (SentenceTransformer): the transformer model
        overwrite (boolean, optional): whether to overwrite out_fn

    Returns:
        List of str: the serialized entries
        List of np.ndarray: the encoded vectors
    """
    input_fn = os.path.join(path, input_fn)
    output_fn = input_fn + '.mat'

    # read from input_fn
    lines = open(input_fn).read().split('\n')

    # encode and dump
    if not os.path.exists(output_fn) or overwrite:
        vectors = model.encode(lines)
        vectors = [v / np.linalg.norm(v) for v in vectors]
        pickle.dump(vectors, open(output_fn, 'wb'))
    else:
        vectors = pickle.load(open(output_fn, 'rb'))
    return lines, vectors


def blocked_matmul(mata, matb,
                   threshold=None,
                   k=None,
                   batch_size=512):
    """Find the most similar pairs of vectors from two matrices (top-k or threshold)

    Args:
        mata (np.ndarray): the first matrix
        matb (np.ndarray): the second matrix
        threshold (float, optional): if set, return all pairs of cosine
            similarity above the threshold
        k (int, optional): if set, return for each row in matb the top-k
            most similar vectors in mata
        batch_size (int, optional): the batch size of each block

    Returns:
        list of tuples: the pairs of similar vectors' indices and the similarity
    """
    mata = np.array(mata)
    matb = np.array(matb)
    results = []
    for start in tqdm(range(0, len(matb), batch_size)):
        block = matb[start:start+batch_size]
        sim_mat = np.matmul(mata, block.transpose())
        if k is not None:
            indices = np.argpartition(-sim_mat, k, axis=0)
            for row in indices[:k]:
                for idx_b, idx_a in enumerate(row):
                    idx_b += start
                    results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))

        if threshold is not None:
            indices = np.argwhere(sim_mat >= threshold)
            total += len(indices)
            for idx_a, idx_b in indices:
                idx_b += start
                results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
    return results


def dump_pairs(out_fn, entries_a, entries_b, pairs):
    """Dump the pairs to a jsonl file
    """
    with jsonlines.open(out_fn, mode='w') as writer:
        for idx_a, idx_b, score in pairs:
            writer.write([entries_a[idx_a], entries_b[idx_b], str(score)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/er_magellan/Structured/Beer/")
    parser.add_argument("--left_fn", type=str, default=None)
    parser.add_argument("--right_fn", type=str, default=None)
    parser.add_argument("--output_fn", type=str, default='candidates.jsonl')
    parser.add_argument("--model_fn", type=str, default="model.pth/")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None) # 0.6
    hp = parser.parse_args()

    # load the model
    model = SentenceTransformer(hp.model_fn)

    # generate the vectors
    mata = matb = None
    entries_a = entries_b = None
    if hp.left_fn is not None:
        entries_a, mata = encode_all(hp.input_path, hp.left_fn, model)
    if hp.right_fn is not None:
        entries_b, matb = encode_all(hp.input_path, hp.right_fn, model)

    if mata and matb:
        pairs = blocked_matmul(mata, matb,
                   threshold=hp.threshold,
                   k=hp.k,
                   batch_size=hp.batch_size)
        dump_pairs(os.path.join(hp.input_path, hp.output_fn),
                   entries_a,
                   entries_b,
                   pairs)
