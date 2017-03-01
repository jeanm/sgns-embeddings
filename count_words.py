#!/usr/bin/env python3

"""
This script counts words in a corpus and produces a plain-text vocabulary file
with counts. The corpus must be a list in msgpack.gz format, with each list
element representing a sentence with the following structure:
    {
        [ word_1, ..., word_N ],
        [ lemma_1, ..., lemma_N ],
        [ pos_tag_1, ..., pos_tag_N ],
        [ ( basic_gr_1, head_1, dep_1), ..., (basic_gr_M, head_M, dep_M) ],
        [ ( enhanced_gr_1, head_1, dep_1), ..., (enhanced_gr_K, head_K, dep_K) ]
    }

Usage:
    count_words.py [--lemmas] [--min-count N] <corpus> <vocab>

Options:
    -h --help       Show this screen
    --lemmas        Work with lemmas instead of raw words
    --min-count N   Minimum word count threshold [default: 5]
"""

import gzip
from time import time
from collections import Counter
import docopt
import msgpack
from utils import cleanup_token

args = docopt.docopt(__doc__)

if args["--lemmas"]:
    print("Counting lemmas...")
else:
    print("Counting words...")

min_count = int(args["--min-count"])

counts = Counter()
sentences_seen = 0
report_interval = 1000000
start_time = time()
with gzip.open(args["<corpus>"], "rb") as fin:
    unp = msgpack.Unpacker(fin, encoding="utf-8")
    for words, lemmas, *_ in unp:
        if args["--lemmas"]:
            words = lemmas
        counts.update(cleanup_token(word) for word in words)
        sentences_seen += 1
        if sentences_seen % report_interval == 0:
            elapsed_mins = (time() - start_time) // 60
            print("Processed "+str(sentences_seen)+" sentences in "+str(elapsed_mins)+" minutes")

print("Found "+str(len(counts))+" unique tokens")
print("Corpus contains "+str(sum(counts.values()))+" tokens")

with open(args["<vocab>"], "w") as fout:
    for word, count in counts.most_common():
        if count < min_count:
            break
        fout.write(word+"\t"+str(count)+"\n")
print("Vocabulary written to "+args["<vocab>"])
