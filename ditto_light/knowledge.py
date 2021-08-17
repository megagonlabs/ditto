import numpy as np
import csv
import sys
import os
import spacy

from collections import Counter

class DKInjector:
    """Inject domain knowledge to the data entry pairs.

    Attributes:
        config: the task configuration
        name: the injector name
    """
    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.initialize()

    def initialize(self):
        pass

    def transform(self, entry):
        return entry

    def transform_file(self, input_fn, overwrite=False):
        """Transform all lines of a tsv file.

        Run the knowledge injector. If the output already exists, just return the file name.

        Args:
            input_fn (str): the input file name
            overwrite (bool, optional): if true, then overwrite any cached output

        Returns:
            str: the output file name
        """
        out_fn = input_fn + '.dk'
        if not os.path.exists(out_fn) or \
            os.stat(out_fn).st_size == 0 or overwrite:

            with open(out_fn, 'w') as fout:
                for line in open(input_fn):
                    LL = line.split('\t')
                    if len(LL) == 3:
                        entry0 = self.transform(LL[0])
                        entry1 = self.transform(LL[1])
                        fout.write(entry0 + '\t' + entry1 + '\t' + LL[2])
        return out_fn


class ProductDKInjector(DKInjector):
    """The domain-knowledge injector for product data.
    """
    def initialize(self):
        """Initialize spacy"""
        self.nlp = spacy.load('en_core_web_lg')

    def transform(self, entry):
        """Transform a data entry.

        Use NER to regconize the product-related named entities and
        mark them in the sequence. Normalize the numbers into the same format.

        Args:
            entry (str): the serialized data entry

        Returns:
            str: the transformed entry
        """
        res = ''
        doc = self.nlp(entry, disable=['tagger', 'parser'])
        ents = doc.ents
        start_indices = {}
        end_indices = {}

        for ent in ents:
            start, end, label = ent.start, ent.end, ent.label_
            if label in ['NORP', 'GPE', 'LOC', 'PERSON', 'PRODUCT']:
                start_indices[start] = 'PRODUCT'
                end_indices[end] = 'PRODUCT'
            if label in ['DATE', 'QUANTITY', 'TIME', 'PERCENT', 'MONEY']:
                start_indices[start] = 'NUM'
                end_indices[end] = 'NUM'

        for idx, token in enumerate(doc):
            if idx in start_indices:
                res += start_indices[idx] + ' '

            # normalizing the numbers
            if token.like_num:
                try:
                    val = float(token.text)
                    if val == round(val):
                        res += '%d ' % (int(val))
                    else:
                        res += '%.2f ' % (val)
                except:
                    res += token.text + ' '
            elif len(token.text) >= 7 and \
                 any([ch.isdigit() for ch in token.text]):
                res += 'ID ' + token.text + ' '
            else:
                res += token.text + ' '
        return res.strip()



class GeneralDKInjector(DKInjector):
    """The domain-knowledge injector for publication and business data.
    """
    def initialize(self):
        """Initialize spacy"""
        self.nlp = spacy.load('en_core_web_lg')

    def transform(self, entry):
        """Transform a data entry.

        Use NER to regconize the product-related named entities and
        mark them in the sequence. Normalize the numbers into the same format.

        Args:
            entry (str): the serialized data entry

        Returns:
            str: the transformed entry
        """
        res = ''
        doc = self.nlp(entry, disable=['tagger', 'parser'])
        ents = doc.ents
        start_indices = {}
        end_indices = {}

        for ent in ents:
            start, end, label = ent.start, ent.end, ent.label_
            if label in ['PERSON', 'ORG', 'LOC', 'PRODUCT', 'DATE', 'QUANTITY', 'TIME']:
                start_indices[start] = label
                end_indices[end] = label

        for idx, token in enumerate(doc):
            if idx in start_indices:
                res += start_indices[idx] + ' '

            # normalizing the numbers
            if token.like_num:
                try:
                    val = float(token.text)
                    if val == round(val):
                        res += '%d ' % (int(val))
                    else:
                        res += '%.2f ' % (val)
                except:
                    res += token.text + ' '
            elif len(token.text) >= 7 and \
                 any([ch.isdigit() for ch in token.text]):
                res += 'ID ' + token.text + ' '
            else:
                res += token.text + ' '
        return res.strip()
