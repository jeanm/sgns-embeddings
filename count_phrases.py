#!/usr/bin/env python3

"""
This script counts adjective-noun, noun-noun, verb-object, verb-subject,
determiner-noun, and negated verb phrases in a corpus and produces a plain-text
vocabulary file with counts. It takes as input a corpus and a word vocabulary
(that must have been precomputed). If working on lemmas, the word vocabulary
must have also been built for lemmas. The corpus must be a list in msgpack.gz
format, with each list element representing a sentence with the following
structure:
    {
        [ word_1, ..., word_N ],
        [ lemma_1, ..., lemma_N ],
        [ pos_tag_1, ..., pos_tag_N ],
        [ ( basic_gr_1, head_1, dep_1), ..., (basic_gr_M, head_M, dep_M) ],
        [ ( enhanced_gr_1, head_1, dep_1), ..., (enhanced_gr_K, head_K, dep_K) ]
    }

Usage:
    count_phrases.py [--lemmas] [--min-count N] <corpus> <vocab> <output-dir>

Options:
    -h --help       Show this screen
    --lemmas        Work with lemmas instead of raw words
    --min-count N   Minimum phrase count threshold [default: 2]
"""

import gzip
from time import time
from collections import Counter
import docopt
import msgpack
from utils import cleanup_token, find_phrases

args = docopt.docopt(__doc__)

min_count = int(args["--min-count"])

vocab = set()
with open(args["<vocab>"]) as fin:
    for line in fin:
        vocab.add(line.split("\t")[0].strip())

if args["--lemmas"]:
    print("Counting lemmatised phrases...")
else:
    print("Counting phrases...")

# iterates through the corpus, extracting phrases
nn_counts = Counter()
an_counts = Counter()
vs_counts = Counter()
vo_counts = Counter()
vps_counts = Counter()
vpo_counts = Counter()
nvs_counts = Counter()
nvo_counts = Counter()
dn_counts = Counter()
sentences_seen = 0
report_interval = 1000000
start_time = time()
with gzip.open(args["<corpus>"], "rb") as fin:
    unp = msgpack.Unpacker(fin, encoding="utf-8")
    for words, lemmas, tags, _, extended_grs in unp:
        if args["--lemmas"]:
            words = lemmas
        words = [cleanup_token(word) for word in words]
        phrases = find_phrases(tags, extended_grs, words)
        # adjective-noun
        for dep, head in phrases.an:
            if words[head] in vocab and words[dep] in vocab:
                an_counts[(words[dep], words[head])] += 1
        # verb-subject
        for dep, head in phrases.vs:
            if words[head] in vocab and words[dep] in vocab:
                vs_counts[(words[dep], words[head])] += 1
        # verb-object
        for head, dep in phrases.vo:
            if words[head] in vocab and words[dep] in vocab:
                vo_counts[(words[head], words[dep])] += 1
        # verb-subject with particle
        for vps in phrases.vps:
            if words[vps[0]] in vocab and words[vps[1]] in vocab and words[vps[2]] in vocab:
                if len(vps) == 3:
                    vps_counts[(words[vps[0]], words[vps[1]], words[vps[2]])] += 1
                elif len(vps) == 4 and words[vps[3]] in vocab:
                    vps_counts[(words[vps[0]], words[vps[1]], words[vps[2]], words[vps[3]])] += 1
        # verb-object with particle
        for vpo in phrases.vpo:
            if words[vpo[0]] in vocab and words[vpo[1]] in vocab and words[vpo[2]] in vocab:
                if len(vpo) == 3:
                    vpo_counts[(words[vpo[0]], words[vpo[1]], words[vpo[2]])] += 1
                elif len(vpo) == 4 and words[vpo[3]] in vocab:
                    vpo_counts[(words[vpo[0]], words[vpo[1]], words[vpo[2]], words[vpo[3]])] += 1
        # determiner-noun
        for dep, head in phrases.dn:
            if words[head] in vocab and words[dep] in vocab:
                dn_counts[(words[dep], words[head])] += 1
        # negated verb-object
        for neg, head, dep in phrases.nvo:
            if words[neg] in vocab and words[head] in vocab and words[dep] in vocab:
                nvo_counts[(words[neg], words[head], words[dep])] += 1
        # negated verb-subject
        for dep, neg, head in phrases.nvs:
            if words[neg] in vocab and words[head] in vocab and words[dep] in vocab:
                nvs_counts[(words[dep], words[neg], words[head])] += 1
        # noun-noun
        for left, right in phrases.nn:
            if words[left] in vocab and words[right] in vocab:
                nn_counts[(words[left], words[right])] += 1
        sentences_seen += 1
        if sentences_seen % report_interval == 0:
            elapsed_mins = (time() - start_time) // 60
            print("Processed "+str(sentences_seen)+" sentences in "+str(elapsed_mins)+" minutes")

with open(args["<output-dir>"]+"/an-vocab.txt", "w") as fout:
    for phrase, count in an_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")

with open(args["<output-dir>"]+"/nn-vocab.txt", "w") as fout:
    for phrase, count in nn_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")

with open(args["<output-dir>"]+"/vs-vocab.txt", "w") as fout:
    for phrase, count in vs_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")

with open(args["<output-dir>"]+"/vo-vocab.txt", "w") as fout:
    for phrase, count in vo_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")

with open(args["<output-dir>"]+"/dn-vocab.txt", "w") as fout:
    for phrase, count in dn_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")

with open(args["<output-dir>"]+"/nvo-vocab.txt", "w") as fout:
    for phrase, count in nvo_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")

with open(args["<output-dir>"]+"/nvs-vocab.txt", "w") as fout:
    for phrase, count in nvs_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")

with open(args["<output-dir>"]+"/vpo-vocab.txt", "w") as fout:
    for phrase, count in vpo_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")

with open(args["<output-dir>"]+"/vps-vocab.txt", "w") as fout:
    for phrase, count in vps_counts.most_common():
        if count < min_count:
            break
        fout.write(" ".join(phrase)+"\t"+str(count)+"\n")
