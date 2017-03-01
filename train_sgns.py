#!/usr/bin/env python3

import json
import numpy as np
from word2vec import Word2Vec

dev_data = json.load(open("../../datasets/MEN/MEN.dev.json"))

counts = []
vocab = []
with open("vocabs_words/vocab.txt") as fin:
    for line in fin:
        word, count = line.strip().split("\t")
        counts.append(int(count))
        vocab.append(word)

w2v = Word2Vec(dev_data=dev_data, dimension=100)
w2v.load_vocab(vocab, counts)
embeddings, _, _ = w2v.train_sentences("corpus_words.txt", epochs=3)
np.save("emb_100_words.npy", embeddings)
