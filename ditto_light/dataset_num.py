# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 01:06:47 2021

@author: lydia
"""

import torch

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
def get_tokenizer(lm):
    return AutoTokenizer.from_pretrained(lm)

class NumDittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='bert-base-uncased',
                 is_num_ditto = False,
                 da=None):
        self.tokenizer = get_tokenizer(lm)
        self.textpairs = []
        self.num_pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size
        
        if isinstance(path, list):
            lines = path
        else:
            lines = open(path)
        
        for line in lines:
            s1, s2, num1, num2, label = line.strip().split('\t')
            
            num1 = self.convert_string_to_float_tensor(num1)
            num2 = self.convert_string_to_float_tensor(num2)
            
            self.num_pairs.append((num1, num2))
            self.textpairs.append((s1, s2))
            self.labels.append(int(label))

        self.textpairs = self.textpairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.textpairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.textpairs[idx][0]
        right = self.textpairs[idx][1]

        # left + right
        encoding = self.tokenizer.encode_plus(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True,
                                  return_attention_mask = True,
                                  return_token_type_ids = True)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]
        
        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split(' [SEP] ')
            encoding = self.tokenizer.encode(text=left,
                                      text_pair=right,
                                      max_length=self.max_len,
                                      truncation=True)
            
            input_ids_aug = encoding["input_ids"]
            attention_mask_aug = encoding["attention_mask"]
            token_type_ids_aug = encoding["token_type_ids"]
            

            return input_ids, input_ids_aug, self.labels[idx], attention_mask, \
                    token_type_ids, attention_mask_aug, token_type_ids_aug, \
                    self.num_pairs[idx][0], self.num_pairs[idx][1]
        else:
            return input_ids, self.labels[idx], attention_mask, token_type_ids, \
                self.num_pairs[idx][0], self.num_pairs[idx][1]
    
    def convert_string_to_float_tensor(self, num_str):
        return list(map(float,num_str.strip().split(" ")))
        

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            input_ids, input_ids_aug, labels, attention_mask, token_type_ids, attention_mask_aug, token_type_ids_aug, num_1, num_2 = zip(*batch)

            maxlen = max([len(x) for x in (input_ids + input_ids_aug)])
            input_ids = [xi + [0]*(maxlen - len(xi)) for xi in input_ids]
            attention_mask = [xi + [0]*(maxlen - len(xi)) for xi in attention_mask]
            token_type_ids = [xi + [0]*(maxlen - len(xi)) for xi in token_type_ids]
            
            input_ids_aug = [xi + [0]*(maxlen - len(xi)) for xi in input_ids_aug]
            attention_mask_aug = [xi + [0]*(maxlen - len(xi)) for xi in attention_mask_aug]
            token_type_ids_aug = [xi + [0]*(maxlen - len(xi)) for xi in token_type_ids_aug]
            
            return torch.LongTensor(input_ids), \
                   torch.LongTensor(input_ids_aug), \
                   torch.LongTensor(labels), \
                   torch.LongTensor(attention_mask), \
                   torch.LongTensor(token_type_ids), \
                   torch.LongTensor(attention_mask_aug), \
                   torch.LongTensor(token_type_ids_aug), \
                   torch.LongTensor(num_1), \
                   torch.LongTensor(num_2)
                   
        else:
            input_ids, labels, attention_mask, token_type_ids, num_1, num_2 = zip(*batch)
            
            maxlen = max([len(x) for x in input_ids])
            input_ids = [xi + [0]*(maxlen - len(xi)) for xi in input_ids]
            attention_mask = [xi + [0]*(maxlen - len(xi)) for xi in attention_mask]
            token_type_ids = [xi + [0]*(maxlen - len(xi)) for xi in token_type_ids]
            num_1 = [torch.tensor(xi, dtype=torch.float32) for xi in num_1]
            num_2 = [torch.tensor(xi, dtype=torch.float32) for xi in num_2]
            
            return torch.LongTensor(input_ids), \
                   torch.LongTensor(labels), \
                   torch.LongTensor(attention_mask), \
                   torch.LongTensor(token_type_ids), \
                   torch.tensor(num_1), \
                   torch.tensor(num_2)

