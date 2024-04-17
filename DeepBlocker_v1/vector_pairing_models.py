#GiG

import numpy as np 
from scipy.spatial import distance

#This is the Abstract Base Class for all Vector Pairing models
class ABCVectorPairing:
    def __init__(self):
        pass 

    #Input is an embedding matrix: #tuples x #dimension
    def index(self, embedding_matrix):
        pass 

    #Input is an embedding matrix: #tuples x #dimension
    #Output: is a matrix of size #tuples x K where K is application dependent
    def query(self, embedding_matrix):
        pass 

#This is a top-K based blocking strategy
# We index the tuple embeddings from one of the datasets and query the othe
#This is an expensive approach that computes all pair cosine and similarity 
# and then extracts top-K neighbors
class ExactTopKVectorPairing(ABCVectorPairing):
    def __init__(self, K):
        super().__init__()
        self.K = K
    
    #Input is an embedding matrix: #tuples x #dimension
    def index(self, embedding_matrix_for_indexing):
        self.embedding_matrix_for_indexing = embedding_matrix_for_indexing

    #Input is an embedding matrix: #tuples x #dimension
    #Output: is a matrix of size #tuples x K where K is an optional parameter
    # the j-th entry in i-th row corresponds to the top-j-th nearest neighbor for i-th row
    def query(self, embedding_matrix_for_querying, K=None):
        if K is None:
            K = self.K

        #Compute the cosine similarity between two matrices with same number of dimensions
        # E.g. N1 x D and N2 x D, this outputs a similarity matrix of size N1 x N2
        #Note: we pass embedding matrix for querying first and then indexing so that we get
        # top-K neighbors in the indexing matrix 
        all_pair_cosine_similarity_matrix = 1 - distance.cdist(embedding_matrix_for_querying, self.embedding_matrix_for_indexing, metric="cosine")
        #-all_pair_cosine_similarity_matrix is needed to get the max.. use all_pair_cosine_similarity_matrix for min
        topK_indices_each_row = np.argsort(-all_pair_cosine_similarity_matrix)[:, :K]
        #you can get the corresponding simlarities via all_pair_cosine_similarity_matrix[index, topK_indices_each_row[index]]
    
        return topK_indices_each_row
    



