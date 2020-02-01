import gensim
import gensim.models
import os

os.chdir('/u/scratch/d/datduong/Onto2Vec/GOVectorData/2017')

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open('AllAxioms.lst'):
            yield line.split()

sentences =gensim.models.word2vec.LineSentence('AllAxioms.lst') # a memory-friendly iterator
ssmodel =gensim.models.Word2Vec(sentences,min_count=0, size=768, window=10, sg=1, negative=4, iter=25)
#Store vector of each  class
GOvectors={}
word_vectors=ssmodel.wv
file= open ('VecResults_768.txt', 'w')
with open('AllClasses.lst') as f:
    for line in f:
        GO_class=line.rstrip()
        if GO_class in word_vectors.vocab:
            GOvectors[GO_class]=ssmodel[GO_class]
            file.write (str(GO_class) + ' '+ ' '.join(str(val) for val in GOvectors[GO_class]) +'\n')


file.close()
