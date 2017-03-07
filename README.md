# Word and phrase embeddings

## Corpus

The corpus used is a 2015-08-05 dump of Wikipedia parsed by Stanford CoreNLP 3.5.2. It is stored as a list in `msgpack` format. List elements correspond to sentences, and have the following format:

    {
        [ word_1, ..., word_N ],
        [ lemma_1, ..., lemma_N ],
        [ pos_tag_1, ..., pos_tag_N ],
        [ ( basic_gr_1, head_1, dep_1), ..., (basic_gr_M, head_M, dep_M) ],
        [ ( enhanced_gr_1, head_1, dep_1), ..., (enhanced_gr_K, head_K, dep_K) ]
    }

The Wikipedia dump or any of the resulting files are not included in this distribution due to their large size. However, given these descriptions, they should be easy to recreate should anyone want to do that.

## Scripts

`count_words.py` iterates through each word in the corpus and counts the occurrences of each word.

`count_phrases.py` iterates through the corpus, identifies certain phrase types from the dependency structure, and counts their occurrences.

`index_corpus.py` produces an indexed version of the corpus. That is, it turns each word in the corpus into the corresponding vocabulary index. The output is a plain text file, where the first line contains the number of sentences in the corpus, and each subsequent line is a sentence, made up of space-separated word indices.

`index_phrase_corpus.py` produces an indexed list of sentences containing a certain type of phrase. The output is a plain text file, where the first line contains the number of sentences, and each subsequent line represents a sentence. The first number in each sentence is the phrase id (indexed according to the phrase vocabulary) and the subsequent numbers are context ids (indexed according to the word vocabulary), sorted by increasing distance from the head of the phrase.

`train_sgns.py` trains vectors using skip-gram with negative sampling, using the Cython code in `word2vec/`. You will need to make sure the Cython code has been compiled by running `make` inside `word2vec/`, and possibly editing the `Makefile` to match the correct paths on your system.
