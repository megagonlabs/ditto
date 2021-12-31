import numpy as np
import csv
import sys
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, OrderedDict
from nltk.corpus import stopwords

from .dataset import get_tokenizer

stopwords = set(stopwords.words('english'))

class Summarizer:
    """To summarize a data entry pair into length up to the max sequence length.

    Args:
        task_config (Dictionary): the task configuration
        lm (string): the language model (bert, albert, or distilbert)

    Attributes:
        config (Dictionary): the task configuration
        tokenizer (Tokenizer): a tokenizer from the huggingface library
    """
    def __init__(self, task_config, lm):
        self.config = task_config
        self.tokenizer = get_tokenizer(lm=lm)
        self.len_cache = {}

        # build the tfidf index
        self.build_index()

    def build_index(self):
        """Build the idf index.

        Store the index and vocabulary in self.idf and self.vocab.
        """
        fns = [self.config['trainset'],
               self.config['validset'],
               self.config['testset']]
        content = []
        for fn in fns:
            with open(fn) as fin:
                for line in fin:
                    LL = line.split('\t')
                    if len(LL) > 2:
                        for entry in LL:
                            content.append(entry)

        vectorizer = TfidfVectorizer().fit(content)
        self.vocab = vectorizer.vocabulary_
        self.idf = vectorizer.idf_

    def get_len(self, word):
        """Return the sentence_piece length of a token.
        """
        if word in self.len_cache:
            return self.len_cache[word]
        length = len(self.tokenizer.tokenize(word))
        self.len_cache[word] = length
        return length

    def transform(self, row, numeric_col_names = None, max_len=128, extract_numeric = False):
        """Summarize one single example.

        Only retain tokens of the highest tf-idf

        Args:
            row (str): a matching example of two data entries and a binary label, separated by tab
            max_len (int, optional): the maximum sequence length to be summarized to

        Returns:
            str: the summarized example
        """
        sentA, sentB, label = row.strip().split('\t')
        res = ''
        cnt = Counter()
        
        numeric_val_A = ""
        numeric_val_B = ""
        if extract_numeric:
            numeric_val_A, sentA = self.separate_numeric_values(sentA, numeric_col_names)
            numeric_val_B, sentB = self.separate_numeric_values(sentB, numeric_col_names)
        
        for sent in [sentA, sentB]:
            tokens = sent.split(' ')
            for token in tokens:
                if token not in ['COL', 'VAL'] and \
                   token not in stopwords:
                    if token in self.vocab:
                        cnt[token] += self.idf[self.vocab[token]]

        for sent in [sentA, sentB]:
            token_cnt = Counter(sent.split(' '))
            total_len = token_cnt['COL'] + token_cnt['VAL']

            subset = Counter()
            for token in set(token_cnt.keys()):
                subset[token] = cnt[token]
            subset = subset.most_common(max_len)

            topk_tokens_copy = set([])
            for word, _ in subset:
                bert_len = self.get_len(word)
                if total_len + bert_len > max_len:
                    break
                total_len += bert_len
                topk_tokens_copy.add(word)

            num_tokens = 0
            for token in sent.split(' '):
                if token in ['COL', 'VAL']:
                    res += token + ' '
                elif token in topk_tokens_copy:
                    res += token + ' '
                    topk_tokens_copy.remove(token)
            res += '\t'
        
        if extract_numeric:
            res += numeric_val_A + '\t'+ numeric_val_B + '\t'

        res += label + '\n'
        return res

    def separate_numeric_values(self, sentence, numeric_col_names = None):
        """return dictionary of the numeric values
        """
        if numeric_col_names == None:
            return None, sentence
        
        numeric = {}
        text = ''
        for col in sentence.split('COL '):
            if len(col) > 0:
                header = col.split(' VAL')[0]
                if header in numeric_col_names:
                    value = col.split('VAL')[1].replace(' ','').replace('/','').replace('-','').replace('$','')
                    if '%' in value:
                        if value.replace('%','').replace('.','').isnumeric():
                            numeric[header] = float(value.replace('%',''))/100
                    elif value.replace('.','').isnumeric():
                        numeric[header] = float(value)
                    else:
                        numeric[header] = 0
                else:
                    text += 'COL ' + col
        
        numeric_result = ""
        for header in numeric_col_names:
            if header in numeric.keys():
                numeric_result += str(numeric[header]) + " "
                
        return numeric_result, text
    
    def transform_file(self, input_fn, numeric_col_names = None, max_len = 256,
                       overwrite = True, is_num_ditto = False):
        """Summarize all lines of a tsv file.

        Run the summarizer. If the output already exists, just return the file name.

        Args:
            input_fn (str): the input file name
            max_len (int, optional): the max sequence len
            overwrite (bool, optional): if true, then overwrite any cached output

        Returns:
            str: the output file name
        """
        out_fn = input_fn + '.su'
        if not os.path.exists(out_fn) or \
           os.stat(out_fn).st_size == 0 or overwrite:
            with open(out_fn, 'w') as fout:
                for line in open(input_fn):
                    fout.write(self.transform(line, 
                                              max_len=max_len, 
                                              numeric_col_names=numeric_col_names,
                                              extract_numeric=is_num_ditto))
        return out_fn
