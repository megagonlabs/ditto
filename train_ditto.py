import os
import argparse
import json
import sys

sys.path.insert(0, "Snippext_public")

from ditto.dataset import DittoDataset
from ditto.summarize import Summarizer
from ditto.knowledge import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--balance", dest="balance", action="store_true")
    parser.add_argument("--size", type=int, default=None)

    hp = parser.parse_args()

    # only a single task for baseline
    task = hp.task

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
    task_type = config['task_type']
    vocab = config['vocab']
    tasknames = [task]

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset, vocab, task,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   balance=hp.balance)
    valid_dataset = DittoDataset(validset, vocab, task, lm=hp.lm)
    test_dataset = DittoDataset(testset, vocab, task, lm=hp.lm)

    if hp.da is None:
        from snippext.baseline import initialize_and_train
        initialize_and_train(config,
                             train_dataset,
                             valid_dataset,
                             test_dataset,
                             hp,
                             run_tag)
    else:
        from snippext.mixda import initialize_and_train
        augment_dataset = DittoDataset(trainset, vocab, task,
                                      lm=hp.lm,
                                      max_len=hp.max_len,
                                      augment_op=hp.da,
                                      size=hp.size,
                                      balance=hp.balance)
        initialize_and_train(config,
                             train_dataset,
                             augment_dataset,
                             valid_dataset,
                             test_dataset,
                             hp,
                             run_tag)
