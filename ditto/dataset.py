import random

from .augment import Augmenter
from snippext.dataset import SnippextDataset, get_tokenizer

class DittoDataset(SnippextDataset):
    def __init__(self,
                 source,
                 vocab,
                 taskname,
                 max_len=512,
                 lm='distilbert',
                 size=None,
                 augment_op=None):
        self.tokenizer = get_tokenizer(lm=lm)

        # tokens and tags
        sents, tags_li = [], [] # list of lists
        self.max_len = max_len

        if type(source) is str:
            sents, tags_li = self.read_classification_file(source)
            if size is not None:
                sents = sents[:size]
                tags_li = tags_li[:size]
        else:
            for sent in source:
                sents.append(sent)
                tags_li.append(vocab[0])

        # assign class variables
        self.sents, self.tags_li = sents, tags_li
        self.vocab = vocab

        # index for tags/labels
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.vocab)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.vocab)}
        self.taskname = taskname

        # augmentation op
        self.augment_op = augment_op
        if augment_op == 't5':
            self.load_t5_examples(source)
        elif augment_op != None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None

    def read_classification_file(self, path):
        """Read a train/eval classification dataset from file

        Args:
            path (str): the path to the dataset file

        Returns:
            list of str: the input sequences
            list of str: the labels
        """
        sents, labels = [], []
        for line in open(path):
            items = line.strip().split('\t')
            # only consider sentence and sentence pairs
            if len(items) < 2 or len(items) > 3:
                continue
            try:
                if len(items) == 2:
                    sents.append(items[0])
                    labels.append(items[1])
                else:
                    sents.append(items[0] + ' [SEP] ' + items[1])
                    labels.append(items[2])
            except:
                print('error @', line.strip())
        return sents, labels

    def __getitem__(self, idx):
        """Return the ith item of in the dataset.

        Args:
            idx (int): the element index
        Returns (TODO):
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
        words, tags = self.sents[idx], self.tags_li[idx]
        original = words

        if self.augment_op == 't5':
            if len(self.augmented_examples[idx]) > 0:
                words, _ = random.choice(self.augmented_examples[idx])
        elif self.augmenter != None:
            words = self.augmenter.augment_sent(words, self.augment_op)

        if ' [SEP] ' in words:
            sents = words.split(' [SEP] ')
            try:
                x = self.tokenizer.encode(text=sents[0], text_pair=sents[1], add_special_tokens=True, truncation="longest_first", max_length=self.max_len)
            except:
                print(sents[0], sents[1])
                sents = original.split(' [SEP] ')
                x = self.tokenizer.encode(text=sents[0], text_pair=sents[1], add_special_tokens=True, truncation="longest_first", max_length=self.max_len)
        else:
            try:
                x = self.tokenizer.encode(text=words, add_special_tokens=True, truncation="longest_first", max_length=self.max_len)
            except:
                print(words)
                x = self.tokenizer.encode(text=original, add_special_tokens=True, truncation="longest_first", max_length=self.max_len)

        y = self.tag2idx[tags] # label
        is_heads = [1] * len(x)
        mask = [1] * len(x)

        assert len(x)==len(mask)==len(is_heads), \
          f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"
        # seqlen
        seqlen = len(mask)

        return words, x, is_heads, tags, mask, y, seqlen, self.taskname

