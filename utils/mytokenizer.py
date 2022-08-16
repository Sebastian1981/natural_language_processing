from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# write custom tokenizer class to be passed to CountVectorizer instance 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        lemma_tokens = [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
        lemma_tokens_alpha = [t for t in lemma_tokens if t.isalpha()]
        lemma_tokens_alpha_long = [t for t in lemma_tokens_alpha if len(t)>=3]
        return lemma_tokens_alpha_long