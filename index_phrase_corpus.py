#!/usr/bin/env python3

"""
This script loads a corpus and produces an indexed list of phrase ids and
corresponding context ids, with contexts sorted by increasing distance from the
head of the phrase. The phrases are indexed according to a phrase vocab, and
the contexts are indexed according to the word vocab. For example, if working
on adjective-noun phrases, the sentence "the red apple is very sweet" yields
(using strings instead of ids) `red_apple is very the sweet`.  The corpus must
be a list in msgpack.gz format, with each list element representing a sentence
with the following structure:

    {
        [ word_1, ..., word_N ],
        [ lemma_1, ..., lemma_N ],
        [ pos_tag_1, ..., pos_tag_N ],
        [ ( basic_gr_1, head_1, dep_1), ..., (basic_gr_M, head_M, dep_M) ],
        [ ( enhanced_gr_1, head_1, dep_1), ..., (enhanced_gr_K, head_K, dep_K) ]
    }
If using the `--lemmas` switch, the vocabularies must have been built for
the lemmas.

Usage:
    index_corpus.py [--lemmas] <corpus> <vocabs-dir> <indexed-corpora-dir>

Options:
    -h --help       Show this screen
    --lemmas        Work with lemmas instead of raw words
"""

import gzip
from time import time
from collections import Counter
import docopt
import msgpack
from itertools import zip_longest
from utils import cleanup_token, find_phrases

args = docopt.docopt(__doc__)

# dictionary that will contain vocabs, count, and reverse vocabs for words
# as well as every phrase type
index2name = {}
index2count = {}
name2index = {}

# load word vocab
index2name["words"] = []
index2count["words"] = []
with open(args["<vocabs-dir>"]+"/vocab.txt") as fin:
    for line in fin:
        if len(line.split("\t")) != 2:
            print(line)
        name, count = line.strip().split("\t")
        index2name["words"].append(name)
        index2count["words"].append(int(count))
name2index["words"] = {n: i for i, n in enumerate(index2name["words"])}

# given the phrase type and the positions of its words, return the
# position of the head
def head_position(phrase_identifier, phrase):
    if phrase_identifier in ["an", "dn", "vs", "nn", "nvo"]:
        return phrase[1]
    elif phrase_identifier in ["vo", "vpo"]:
        return phrase[0]
    else:
        raise ValueError("Phrase identifier "+str(phrase_identifier)+" unknown")

# given the words of a sentence, the position of the words forming a phrase,
# and the position of the phrase's head, return the contexts of the phrase
# sorted by increasing distance from the head's position
def phrase_contexts(words, phrase, head_pos):
    exclude = set(phrase)
    return (words[a] for b in zip_longest(range(head_pos, -1, -1),
                                   range(head_pos+1, len(words)))
            for a in b if a is not None and a not in exclude)


# index the corpus
def index(phrase_identifiers=["an"]):
    # load phrase vocabs
    for phrase in phrase_identifiers:
        index2name[phrase] = []
        index2count[phrase] = []
        with open(args["<vocabs-dir>"]+"/"+phrase+"_vocab.txt") as fin:
            for line in fin:
                name, count = line.strip().split("\t")
                index2name[phrase].append(tuple(name.split(" ")))
                index2count[phrase].append(int(count))
        name2index[phrase] = {n: i for i, n in enumerate(index2name[phrase])}

    # open all output files and initialise phrase counters
    fouts = {}
    phrases_added = {}
    for phrase_identifier in phrase_identifiers:
        fouts[phrase_identifier] = open(args["<indexed-corpora-dir>"]+"/corpus_"+phrase_identifier+".txt", "w")
        phrases_added[phrase_identifier] = 0

    # index the corpus
    sentences_seen = 0
    report_interval = 1000000
    start_time = time()
    with gzip.open(args["<corpus>"], "rb") as fin:
        unp = msgpack.Unpacker(fin, encoding="utf-8")
        for fout in fouts.values():
            fout.write(" "*20+"\n")  # reserve space for the number of sentences

        for words, lemmas, tags, _, extended_grs in unp:
            if args["--lemmas"]:
                words = lemmas
            words = [cleanup_token(word) for word in words]
            phrases = find_phrases(tags, extended_grs, words)

            for phrase_identifier in phrase_identifiers:
                for phrase in getattr(phrases, phrase_identifier):
                    phrase_name = tuple(map(lambda x: words[x], phrase))
                    if phrase_name in name2index[phrase_identifier]:
                        head_pos = head_position(phrase_identifier, phrase)
                        contexts = phrase_contexts(words, phrase, head_pos)
                        fouts[phrase_identifier].write(str(name2index[phrase_identifier][phrase_name])+" ")
                        fouts[phrase_identifier].write(
                            " ".join(
                                str(name2index["words"][context]) for context in contexts
                                                                  if context in name2index["words"]
                            ) + "\n"
                        )
                        phrases_added[phrase_identifier] += 1
                sentences_seen += 1

            if sentences_seen % report_interval == 0:
                elapsed_mins = (time() - start_time) // 60
                print("Indexed "+str(sentences_seen)+" sentences in "+str(elapsed_mins)+" minutes")

    # close all output files
    for fout in fouts.values():
        fout.close()

    # go to the beginning of each output file and add the number of phrases
    for phrase_identifier in phrase_identifiers:
        with open(args["<indexed-corpora-dir>"]+"/corpus_"+phrase_identifier+".txt", "r+") as fout:
            fout.seek(0)
            fout.write(str(phrases_added[phrase_identifier]))

index(["nvo", "vo", "vpo", "an", "nn", "dn"])
