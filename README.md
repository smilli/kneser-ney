# kneser-ney
An implementation of [Kneser-Ney](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing) language modeling in Python3.
This is not a particularly optimized implementation, but is hopefully helpful for learning and works fine for corpuses that aren't too large.

# Usage

  The KneserNey class does language model estimation when given a sequence of ngrams.

```python
class KneserNey:

  def __init__(self, highest_order, ngrams, start_pad_symbol='<s>', end_pad_symbol='</s>'):
    """
    Constructor for KneserNeyLM.

    Params:
        highest_order [int] The order of the language model.
        ngrams [list->tuple->string] Ngrams of the highest_order specified.
            Ngrams at beginning / end of sentences should be padded.
        start_pad_symbol [string] The symbol used to pad the beginning of
            sentences.
        end_pad_symbol [string] The symbol used to pad the beginning of
            sentences.
    """
 ```

It is easy to create a KneserNeyLM out of an NLTK corpus (see example.py).

```python
from nltk.corpus import gutenberg
from nltk.util import ngrams
from kneser_ney import KneserNeyLM

gut_ngrams = (
    ngram for sent in gutenberg.sents() for ngram in ngrams(sent, 3,
    pad_left=True, pad_right=True, pad_symbol='<s>'))
lm = KneserNeyLM(3, gut_ngrams, end_pad_symbol='<s>')
```

The language model can then be used to score sentences or generate sentences.

```python
lm.score_sent(('This', 'is', 'a', 'sample', 'sentence', '.'))
lm.generate_sentence()
```
