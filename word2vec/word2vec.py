from time import time
import logging
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from .word2vec_fast import train_sentence, train_single
import gzip

floatX = np.float32

logging.basicConfig(level=logging.INFO,format="[%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

class Word2Vec():

    def __init__(self, dimension=100, alpha=0.025, window=5, negative=5,
                 sample=1e-3, dev_data=None):
        self.name2index = {} # string -> word index
        self.index2name = [] # word index -> string
        self.index2count = [] # word index -> word count
        self.index2sample = [] # word index -> word sampling threshold
        self.ns_table = None
        self.dim = dimension
        self.alpha = alpha
        self.cur_alpha = alpha
        self.window = window
        self.negative = negative
        self.sample = sample
        self.dev = False
        if dev_data:
            self.dev_words = [(x,y) for x,y,*_ in dev_data]
            self.devsims = np.asarray([float(x) for _,_,x,*_ in dev_data])
            self.dev = True

    def _downsample_vocab(self):
        retain_total = sum(self.index2count)
        # Precalculate each vocabulary item"s threshold for sampling
        if not self.sample:
            # no words downsampled
            threshold_count = retain_total
        else:
            # set parameter as proportion of total
            threshold_count = self.sample * retain_total

        self.index2sample = []
        downsample_total, downsample_unique = 0, 0
        for w in range(len(self.index2name)):
            v = self.index2count[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            self.index2sample.append(int(round(word_probability * 2**32)))

        logger.info("sample=%g downsampled the %s most common words", self.sample, downsample_unique)
        logger.info("downsampling will decrease corpus size to approximately %.1f%%",
                    downsample_total * 100.0 / max(retain_total, 1))

    def _make_ns_table(self, power=0.75, domain=2**31 - 1):
        vocabsize = len(self.index2name)
        self.ns_table = np.zeros(vocabsize, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.index2count[index]**power for index in range(vocabsize)]))
        cumulative = 0.0
        for word_index in range(vocabsize):
            cumulative += self.index2count[word_index]**power / train_words_pow
            self.ns_table[word_index] = round(cumulative * domain)
        if len(self.ns_table) > 0:
            assert self.ns_table[-1] == domain

    def _finalise_vocab(self):
        # precalculate sampling thresholds for words
        self._downsample_vocab()
        # build the table for drawing random words (for negative sampling)
        self._make_ns_table()

    def load_vocab(self, vocab, counts):
        self.index2name = vocab
        self.index2count = counts
        logger.info("loaded a word vocabulary of size %s", len(self.index2name))
        self.name2index = {e:i for i,e in enumerate(self.index2name)}
        self._finalise_vocab()
        # set context vectors to zero
        self.reset_weights()

    def reset_weights(self, what=None):
        vocabsize = len(self.index2name)
        self.contexts = np.zeros((vocabsize, self.dim), dtype=floatX, order="C")
        logger.info("initialised a %s x %s context matrix", vocabsize, self.dim)

    def test_dev(self, embed):
        # array of dot products between pairs of (normalised) noun vectors
        # in the development set
        scores = np.asarray([1-cosine(embed[self.name2index[a]],embed[self.name2index[b]]) for a,b in self.dev_words])
        # return spearman rank between human similarity judgements and
        # our own similarity judgements
        return spearmanr(scores,self.devsims)

    def train_sentences(self, corpus_infile, epochs=1, report_freq=20):
        if len(self.index2sample) == 0:
            logger.error("attempted to start training but vocabulary has not been loaded")
            raise RuntimeError("You must build/load the vocabulary before training the model")
        epochs = int(epochs) or 1
        # initialise temporary work memory and word vectors
        work = np.zeros(self.dim, dtype=floatX)
        embeddings = np.ascontiguousarray((np.random.rand(len(self.index2name), self.dim) - 0.5) / self.dim,dtype=floatX)
        logger.info("initialised a %s x %s embedding matrix", len(self.index2name), self.dim)
        with gzip.open(corpus_infile, "r") as fin:
            total_words = 0
            # read the number of sentences in the corpus
            corpus_sentences = int(next(fin).strip())
            total_sentences = epochs * corpus_sentences
            logger.info("loaded corpus with %s sentences, training for %d epochs", corpus_sentences, epochs)

            start_time = time()
            tic = time()
            word_count = 0
            for epoch in range(epochs):
                fin.seek(0)
                next(fin) # skip first line with number of sentences
                for sentence_num, line in enumerate(fin,start=epoch*corpus_sentences):
                    alpha = self.alpha * (1 - sentence_num / total_sentences)
                    sentence = list(map(int,line.strip().split()))
                    word_count += len(sentence)
                    train_sentence(self, sentence, alpha, embeddings, work)
                    if time() - tic >= report_freq:
                        tic = time()
                        if self.dev:
                            cor = self.test_dev(embeddings)
                            logger.info("%.2f%% sentences @ %s words/s, alpha %.6f, corr %.5f (p %.2e)" %
                                (100 * sentence_num / total_sentences, word_count / report_freq, alpha, cor[0], cor[1]))
                        else:
                            logger.info("%.2f%% sentences @ %s words/s, alpha %.6f" %
                                (100 * sentence_num / total_sentences, word_count / report_freq, alpha))
                        total_words += word_count
                        word_count = 0
                total_words += word_count
        elapsed_time = time() - start_time
        logger.info("trained on %s sentences (%s words) in %s min @ %s words/s" %
                (total_sentences, total_words, elapsed_time/60.0,
                    total_words / elapsed_time))
        cor = self.test_dev(embeddings)
        logger.info("correlation on development set %.5f (p %.2e)" % cor)
        return embeddings, self.index2name, self.index2count

    # trains phrase vectors one by one, leaving the context weights constant
    def train_phrases(self, corpus_infile, phrase_vocab, phrase_counts, epochs=1, report_freq=20):
        if len(self.index2sample) == 0:
            logger.error("attempted to start training but vocabulary has not been loaded")
            raise RuntimeError("You must build/load the vocabulary before training the model")
        epochs = int(epochs) or 1

        # count the number of phrase vectors to be learned
        phrase_index2count = phrase_counts
        phrase_index2name = phrase_vocab
        vocabsize = len(phrase_index2count)

        # initialise temporary work memory and phrase vectors
        work = np.zeros(self.dim, dtype=floatX)
        embeddings = np.ascontiguousarray((np.random.rand(vocabsize, self.dim) - 0.5) / self.dim,dtype=floatX)
        logger.info("initialised a %s x %s phrase embedding matrix", vocabsize, self.dim)

        with gzip.open(corpus_infile, "r") as fin:
            total_words = 0
            # read the number of sentences in the corpus
            corpus_sentences = int(next(fin).strip())
            total_sentences = epochs * corpus_sentences
            logger.info("loaded corpus with %s examples, training for %d epochs", corpus_sentences, epochs)

            start_time = time()
            tic = time()
            word_count = 0
            for epoch in range(epochs):
                fin.seek(0)
                next(fin) # skip first line with number of sentences
                for sentence_num, line in enumerate(fin,start=epoch*corpus_sentences):
                    sentence = list(map(int,line.strip().split()))[:self.window+1]
                    if len(sentence) <= 1: continue
                    alpha = self.alpha * (1 - sentence_num / total_sentences)
                    word_count += len(sentence)-1
                    train_single(self, sentence, alpha, embeddings, work)
                    if time() - tic >= report_freq:
                        tic = time()
                        logger.info("%.2f%% examples @ %s words/s, alpha %.6f" %
                            (100 * sentence_num / total_sentences, word_count / report_freq, alpha))
                        total_words += word_count
                        word_count = 0
                total_words += word_count
        elapsed_time = time() - start_time
        logger.info("trained on %s words (%s examples) in %s min @ %s words/s" %
                (total_words, total_sentences, elapsed_time/60.0,
                    total_words / elapsed_time))
        return embeddings, phrase_index2name, phrase_index2count

    # save/load context vectors
    def get_contexts(self):
        return self.contexts, self.index2name, self.index2count
    def load_contexts(self, contexts, vocab, counts):
        shape = contexts.shape
        if shape[0] != len(contexts.index2name) or shape[1] != self.dim:
            logger.error("vocabulary/contexts shape mismatch")
            raise RuntimeError("vocabulary/contexts shape mismatch")
        self.contexts = contexts
        self.index2name = vocab
        self.index2count = counts
        self.name2index = {e:i for i,e in enumerate(self.index2name)}
        logger.info("loaded a %s x %s context matrix", len(self.index2name), self.dim)
        self._finalise_vocab()
