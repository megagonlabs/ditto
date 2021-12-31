# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 10:06:23 2021

@author: lydia
"""

import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset_num import NumDittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.num_ditto import train

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='bert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    #parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--dk", type=str, default="product")
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    hp.lm = lm_mp[hp.lm]

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    numeric_feature_cols = config['number_feature_columns']

    # summarize the sequences up to the max sequence length
    hp.summarize = True
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len, is_num_ditto=True, numeric_col_names=numeric_feature_cols)
        validset = summarizer.transform_file(validset, max_len=hp.max_len, is_num_ditto=True, numeric_col_names=numeric_feature_cols)
        testset = summarizer.transform_file(testset, max_len=hp.max_len, is_num_ditto=True, numeric_col_names=numeric_feature_cols)

    # if hp.dk is not None:
    #     if hp.dk == 'product':
    #         injector = ProductDKInjector(config, hp.dk)
    #     else:
    #         injector = GeneralDKInjector(config, hp.dk)

    #     trainset = injector.transform_file(trainset, is_num_ditto=True, numeric_col_names=numeric_feature_cols)
    #     validset = injector.transform_file(validset, is_num_ditto=True, numeric_col_names=numeric_feature_cols)
    #     testset = injector.transform_file(testset, is_num_ditto=True, numeric_col_names=numeric_feature_cols)

    # load train/dev/test sets
    train_dataset = NumDittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da)
    valid_dataset = NumDittoDataset(validset, lm=hp.lm)
    test_dataset = NumDittoDataset(testset, lm=hp.lm)

    # train and evaluate the model
    train(train_dataset,
          valid_dataset,
          test_dataset,
          run_tag, hp)
