datasets = """Structured/Amazon-Google
Structured/DBLP-ACM
Structured/DBLP-GoogleScholar
Structured/Walmart-Amazon
Textual/Abt-Buy
Textual/Company""".split('\n')

ops = """swap
append_col
del
swap
swap
del""".split('\n')

lm = "roberta"
epochs = 20

import os
import time

for dataset, op in zip(datasets, ops):
    for size in [500, 1000, 1500, 2000]:
        for run_id in range(5):
            for da in [True, False]:
                for dk in [True, False]:
                    # DK
                    ds = dataset
                    start = time.time()
                    cmd = """CUDA_VISIBLE_DEVICES=3 python train_ditto.py \
                  --task %s \
                  --logdir results_ditto/ \
                  --finetuning \
                  --batch_size %d \
                  --lr 3e-5 \
                  --fp16 \
                  --lm %s \
                  --n_epochs %d \
                  --size %d \
                  --run_id %d""" % (ds, batch_size, lm, epochs, size, run_id)
                    if 'Company' in ds:
                        cmd += ' --summarize'
                    if da:
                        cmd += ' --da %s' % op
                    if dk:
                        cmd += ' --dk general'
                    print(cmd)
                    os.system(cmd)
