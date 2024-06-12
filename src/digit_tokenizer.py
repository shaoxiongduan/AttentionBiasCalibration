from vocab import build_vocab_from_iterator
import string

#
# The tokenizer only tokenizes sequences, vocabulary should be from other places. Since we treat digits as tokens, and we are not
# handling bases higher than 36 (i.e., we are using only alphanumerics), our tokenizer is very simple. It just split the sequence
# into characters. 
#

class DigitTokenizer:
    #
    # The `__call__` method is used to make an instance of a class callable like a function. The `self` 
    # parameter is required in the `__call__` method just like any other instance method in a class.  
    #
    def __call__(self, src: str) -> str:
        tokens = [x for x in src.replace(" ", "")]    # remove its space first
        return tokens
