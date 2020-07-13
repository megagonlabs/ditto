datasets = ["all","computers", "cameras", "shoes", "watches"]
attrs = ['title', 'title_description', 'title_description_brand', 'title_description_brand_specTableContent']
sizes = ["small", "medium", "large", "xlarge"]

import os
import time

gpu_id = 0

for d in datasets:
    for attr in attrs:
        for size in sizes:
            dataset = '_'.join(['wdc', d, attr, size])
            for dk in [True, False]:
                for da in [True, False]:
                    for run_id in range(5):
                        cmd = """CUDA_VISIBLE_DEVICES=%d python train_ditto.py \
                          --task %s \
                          --logdir results_wdc/ \
                          --fp16 \
                          --finetuning \
                          --batch_size 64 \
                          --lr 3e-5 \
                          --n_epochs 10 \
                          --run_id %d""" % (gpu_id, dataset, run_id)
                        if da:
                            cmd += ' --da del'
                        if dk:
                            cmd += ' --dk product'
                        if attr != 'title':
                            cmd += ' --summarize'
                        print(cmd)
                        os.system(cmd)
