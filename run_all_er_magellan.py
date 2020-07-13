import os
import time

datasets = """Dirty/DBLP-ACM
Dirty/DBLP-GoogleScholar
Dirty/iTunes-Amazon
Dirty/Walmart-Amazon
Structured/Amazon-Google
Structured/Beer
Structured/DBLP-ACM
Structured/DBLP-GoogleScholar
Structured/Fodors-Zagats
Structured/iTunes-Amazon
Structured/Walmart-Amazon
Textual/Abt-Buy
Textual/Company""".split('\n')

special_datasets = {
    'Structured/Beer': (32, 40),
    'Structured/iTunes-Amazon': (32, 40),
    'Structured/Fodors-Zagats': (32, 40),
    'Dirty/iTunes-Amazon': (32, 40),
    'Textual/Company': (32, 3)
}

ops = """swap
swap
append_col
del
swap
drop_col
swap
swap
append_col
drop_col
drop_col
swap
del""".split('\n')


lms = ['roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta',
       'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'bert']

# lms = ['xlnet', 'roberta', 'roberta', 'roberta', 'xlnet', 'bert',
#        'bert', 'xlnet', 'roberta', 'bert', 'roberta', 'roberta', 'bert']

# lms = """distilbert
# bert
# xlnet
# roberta""".split('\n')

for dataset, op, lm in zip(datasets, ops, lms):
    if dataset in special_datasets:
        batch_size, epochs = special_datasets[dataset]
    else:
        batch_size, epochs = 32, 15

    for da in [True, False]:
        for dk in [True, False]:
            for run_id in range(5):
                cmd = """CUDA_VISIBLE_DEVICES=3 python train_ditto.py \
              --task %s \
              --logdir results_ditto/ \
              --finetuning \
              --batch_size %d \
              --lr 3e-5 \
              --fp16 \
              --lm %s \
              --n_epochs %d \
              --run_id %d""" % (dataset, batch_size, lm, epochs, run_id)
                if 'Company' in dataset:
                    cmd += ' --summarize'
                if da:
                    cmd += ' --da %s' % op
                if dk:
                    cmd += ' --dk general'
                print(cmd)
                os.system(cmd)
