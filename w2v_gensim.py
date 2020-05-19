# Tosin Adewumi

"""
gensim word2vec models
"""

import re
import time
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus
import logging
import os
from multiprocessing import cpu_count


LANGUAGE = "english"
DIRECTORY = "data"
OUTPUT_FILE1 = "out_w2v.txt"


class ReadLinebyLine():
    def __init__(self, corpus_location, language):
        """

        :param corpus_location:
        """
        self.directory = corpus_location
        assert isinstance(self.directory, str), "method requires string corpus location"
        self.stop_words = set(stopwords.words(language))    # no support for yoruba
        self.stop_words.update(['.',',',':',';','(',')','#','--','...','"','_','|','Â»','%','[',']','{','}'])


    def __iter__(self):
        """

        :return:
        """
        for fname in os.listdir(self.directory):                                        # go over all files in directory
            for line in open(os.path.join(self.directory, fname), encoding="utf8"):     # read file line by line in utf8
                line = line.lower()
                line = re.sub("<.*?>", "", line)            # removes html tags
                tokenized_text = word_tokenize(line)
                if len(tokenized_text) == 0:                # replace empty lists with BLANKTOKEN
                    tokenized_text = ["BLANKTOKEN"]
                tokenized_text = [nonum for nonum in tokenized_text if not nonum.isnumeric()]   # remove numbers
                yield [w for w in tokenized_text if w not in self.stop_words]   # returns memory-efficient generator


if __name__ == "__main__":
    """
    Matrix of variables for experiment:
    # size: 300, 1200, 1800, 2400, 3000
    # window: 4, 8
    # sg: 1, 0 (CBoW)
    # hs: 1, 0 (negative sampling if negative > 0)
    # negative: 5
    # iter: 5, 10
    """
    # logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=1)
    starttime = time.time()
    processed_corpus = ReadLinebyLine(DIRECTORY, LANGUAGE)     # memory-efficient iterator for regular files
    model = Word2Vec(processed_corpus, min_count=5, size=300, workers=cpu_count(), window=8, sg=1, hs=1, negative=0, iter=5, compute_loss=True)
    time_elapsed = time.time() - starttime
    # saving and loading own models | using gzip/bz2 input also works | saved model can be loaded & trained with more
    #model.save(DIRECTORY + "/" + "word2vec_m5_s350_w8_s1_h1_n0_i20")
    # get vocab size and save
    print("Vocab", len(model.wv.vocab))
    #print("Training loss", model.get_latest_training_loss())
    print("Time elapsed", time_elapsed)
    # evaluate on word analogies - modern version of accuracy
    human_word_sim = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
    analogy_scores = model.wv.evaluate_word_analogies(datapath("questions-words-swedish2.txt"))
    print("Analogy score: ", analogy_scores)
    with open(OUTPUT_FILE1, "w+") as f:
        s = f.write("Vocab: " + str(len(model.wv.vocab)) + "\n" "Training loss: " +
                    str(model.get_latest_training_loss()) + "\n" "Time elapsed: " + str(time_elapsed) +
                    "\n" "Analogy Scores: " + str(analogy_scores) + "\n" "Human Word Similarity:" + str(human_word_sim))
