#!/usr/bin/env python3

"""
This script loads a corpus and produces a version of it where each word is
replaced by an index corresponding to the word's position in a given
vocabulary. The corpus must be a list in msgpack.gz format, with each list
element representing a sentence with the following structure:
    {
        [ word_1, ..., word_N ],
        [ lemma_1, ..., lemma_N ],
        [ pos_tag_1, ..., pos_tag_N ],
        [ ( basic_gr_1, head_1, dep_1), ..., (basic_gr_M, head_M, dep_M) ],
        [ ( enhanced_gr_1, head_1, dep_1), ..., (enhanced_gr_K, head_K, dep_K) ]
    }
If using the `--lemmas` switch, the vocabulary must have been built for
the lemmas.

Usage:
    index_corpus.py [--lemmas] <corpus> <vocab> <indexed-corpus>

Options:
    -h --help       Show this screen
    --lemmas        Work with lemmas instead of raw words
"""

import gzip
from time import time
from collections import Counter
import docopt
import msgpack
from utils import cleanup_token

args = docopt.docopt(__doc__)

index2name = []
index2count = []
with open(args["<vocab>"]) as fin:
    for line in fin:
        if len(line.split("\t")) != 2:
            print(line)
        name, count = line.strip().split("\t")
        index2name.append(name)
        index2count.append(int(count))
name2index = {n: i for i, n in enumerate(index2name)}

# index the corpus
sentences_seen = 0
report_interval = 1000000
start_time = time()
with gzip.open(args["<corpus>"], "rb") as fin, open(args["<indexed-corpus>"], "w") as fout:
    unp = msgpack.Unpacker(fin, encoding="utf-8")
    fout.write(" "*20+"\n")  # reserve space for the number of sentences
    for sentences_seen, sentence in enumerate(unp):
        if args["--lemmas"]:
            tokens = sentence[1]
        else:
            tokens = sentence[0]
        fout.write(" ".join(str(name2index[token]) for token in tokens
                   if token in name2index)+"\n")
        if sentences_seen % report_interval == 0:
            elapsed_mins = (time() - start_time) // 60
            print("Indexed "+str(sentences_seen)+" sentences in "+str(elapsed_mins)+" minutes")

# go to the beginning of the file and add the number of sentences
with open(args["<indexed-corpus>"], "r+") as fout:
    fout.seek(0)
    fout.write(str(sentences_seen+1))
