#GiG
import numpy as np
import pandas as pd
import time
from pathlib import Path
import blocking_utils

class DeepBlocker:
    def __init__(self, tuple_embedding_model, vector_pairing_model):
        self.tuple_embedding_model = tuple_embedding_model
        self.vector_pairing_model = vector_pairing_model

    def validate_columns(self):
        #Assumption: id column is named as id
        if "id" not in self.cols_to_block:
            self.cols_to_block.append("id")
        self.cols_to_block_without_id = [col for col in self.cols_to_block if col != "id"]

        #Check if all required columns are in left_df
        check = all([col in self.left_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the left dataset")

        #Check if all required columns are in right_df
        check = all([col in self.right_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the right dataset")


    def preprocess_datasets(self):
        self.left_df = self.left_df[self.cols_to_block]
        self.right_df = self.right_df[self.cols_to_block]

        self.left_df.fillna(' ', inplace=True)
        self.right_df.fillna(' ', inplace=True)

        self.left_df = self.left_df.astype(str)
        self.right_df = self.right_df.astype(str)


        self.left_df["_merged_text"] = self.left_df[self.cols_to_block_without_id].agg(' '.join, axis=1)
        self.right_df["_merged_text"] = self.right_df[self.cols_to_block_without_id].agg(' '.join, axis=1)

        #Drop the other columns
        self.left_df = self.left_df.drop(columns=self.cols_to_block_without_id)
        self.right_df = self.right_df.drop(columns=self.cols_to_block_without_id)


    def block_datasets(self, left_df, right_df, cols_to_block):
        self.left_df = left_df
        self.right_df = right_df
        self.cols_to_block = cols_to_block

        print('cols to block',cols_to_block)
        print('left df head',left_df.columns)
        print('right df head',right_df.columns)

        self.validate_columns()
        self.preprocess_datasets()

        print("Performing pre-processing for tuple embeddings ")
        all_merged_text = pd.concat([self.left_df["_merged_text"], self.right_df["_merged_text"]], ignore_index=True)
        self.tuple_embedding_model.preprocess(all_merged_text)

        print('after preprocess')

        start_time = time.time()

        print("Obtaining tuple embeddings for left table")
        self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
        print("Obtaining tuple embeddings for right table")
        self.right_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.right_df["_merged_text"])


        print("Indexing the embeddings from the right dataset")
        self.vector_pairing_model.index(self.right_tuple_embeddings)

        print("Querying the embeddings from left dataset")
        topK_neighbors = self.vector_pairing_model.query(self.left_tuple_embeddings)

        self.candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)

        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Time elapsed in seconds',time_elapsed)

        return self.candidate_set_df
