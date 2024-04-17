#GiG

import numpy as np
import pandas as pd
from pathlib import Path
import os
from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils

def do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, vector_pairing_model):
    folder_root = Path(folder_root)
    left_csv_path = folder_root / left_table_fname
    right_csv_path = folder_root / right_table_fname
    try:
        left_df = pd.read_csv(left_csv_path,encoding="ISO-8859-1")
        right_df = pd.read_csv(right_csv_path,encoding="ISO-8859-1")
    except UnicodeDecodeError as e:
        with open(left_csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines):
                try:
                    line.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    print(f"UnicodeDecodeError occurred at line {line_num + 1}: {line}")
                    break
        raise e
    
    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block)

    golden_df = pd.read_csv(Path(folder_root) /  "matches.csv")
    statistics_dict = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
    return statistics_dict


# def do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, vector_pairing_model):
#     folder_root = Path(folder_root)
#     left_df = pd.read_csv(folder_root / left_table_fname)
#     right_df = pd.read_csv(folder_root / right_table_fname)

#     db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
#     candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block)

#     golden_df = pd.read_csv(Path(folder_root) /  "matches.csv")
#     statistics_dict = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
#     return statistics_dict


if __name__ == "__main__":
    folder_root = "data/Textual/Abt-Buy"
    left_table_fname, right_table_fname = "tableA.csv", "tableB.csv"

    # df = pd.read_csv(os.path.join(folder_root,left_table_fname))
    # print(df.columns)

    #cols_to_block = ["Beer_Name", "Brew_Factory_Name", "Style"]
    #cols_to_block = ["id","content"]
    #cols_to_block = ["title","manufacturer","price"]
    #cols_to_block = ["title","authors","venue"]
    #cols_to_block = ["name","addr","city"]
    cols_to_block = ["name","description","price"]

    print("using AutoEncoder embedding")
    tuple_embedding_model = AutoEncoderTupleEmbedding()
    topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
    statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    print(statistics_dict)

    # print("using CTT embedding")
    # tuple_embedding_model = CTTTupleEmbedding()
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
    # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    # print(statistics_dict)

    # print("using Hybrid embedding")
    # tuple_embedding_model = HybridTupleEmbedding()
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
    # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    # print(statistics_dict)
