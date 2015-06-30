from nltk.corpus import gutenberg
from nltk.util import ngrams
from kneser_ney import KneserNeyLM

gut_ngrams = (
    ngram for sent in gutenberg.sents() for ngram in ngrams(sent, 3,
    pad_left=True, pad_right=True, pad_symbol='<s>'))
lm = KneserNeyLM(3, gut_ngrams, end_pad_symbol='<s>')
for _ in range(5):
    print(lm.generate_sentence())
