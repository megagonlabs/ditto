## The WDC Product Matching Benchmark

We obtained this dataset from [WDC Product Data Corpus](http://webdatacommons.org/largescaleproductcorpus/v2/index.html).

* Primpeli, Anna, Ralph Peeters, and Christian Bizer. "The WDC training dataset and gold standard for large-scale product matching." WWW 2019.

We currently only stored the train/valid/test data with the preprocessed title attribute only. The other attribute combinations (e.g., title+description) can be obtained by
```
wget https://ditto-em.s3.us-east-2.amazonaws.com/wdc.zip
unzip wdc.zip
```
