import pandas as pd
import nltk

# Ensuring required data is installed.
try:
    nltk.data.find('corpora/wordnet.zip')
except:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4.zip')
except:
    nltk.download('omw-1.4')
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

toktok = nltk.tokenize.ToktokTokenizer()
snowball = nltk.stem.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
punkt = nltk.tokenize.PunktSentenceTokenizer()

def lemmatize(sentence, lemmatizer:object = wordnet) -> str:
    words = sentence.split(' ')
    out = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(out)


def stem(sentence, stemmer:object = snowball) -> str:
    words = sentence.split(' ')
    out = [stemmer.stem(word) for word in words]
    return ' '.join(out)


def word_tokenize(string:str, tokenizer:object = toktok) -> str:
    tokens =  tokenizer.tokenize(string)
    return ' '.join(tokens)

def sent_tokenize(string:str, tokenizer:object = punkt) -> list:
    """Requires punctuation"""
    tokens =  tokenizer.tokenize(string)
    return '\n\n'.join(tokens)

def remove_stopwords(string:str, extra_words:list[str] = [], exclude_words:list[str] = []) -> str:
    tokens = string.split(' ')
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords += extra_words
    stopwords = [word for word in stopwords if not word in exclude_words]
    out = [word for word in tokens if not word in stopwords]
    return ' '.join(out)

def basic_string_clean( string: str, 
                        strip=True, 
                        lower=True, 
                        normalize=True, 
                        drop_special=True, 
                        drop_punctuation=True) -> str:
    """Returns the same string with the following alterations by default:
    - convert all chars to lowercase
    - maps charcters to fit within ASCII character set (converts accented chars to unaccented counterparts)
    - drops anything that didn't get mapped
    - removes special characters and punctuation
    TODO: Hyphen strategy argument?
    """
    import re
    import unicodedata
    if strip:
        string = string.strip()
    if lower:
        string = string.lower()
    if normalize:
        # Handle curly quotes
        charmap = { 0x201c : u'"',
                    0x201d : u'"',
                    0x2018 : u"'",
                    0x2019 : u"'" }
        string = string.translate(charmap)
        string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    if drop_special:
        string = re.sub(r"[\n-]", ' ', string) # Hyphen strategy for now
        regex = r"[^\w\d\s\.\?\!\:\,\']|[_]"
        string = re.sub(regex, '', string)
    if drop_punctuation:
        regex = r"[\.\?\!\:\,]"
        string = re.sub(regex, '', string)

    return string

def sanitize(document: str, **kwargs):
    defaultKwargs = {
        'strip': True, 
        'lower': True, 
        'normalize': True, 
        'drop_special': True, 
        'drop_punctuation': True,
        'extra_words': [], 
        'exclude_words': []
    }
    kwargs = { **defaultKwargs, **kwargs }

    document = basic_string_clean(document,
                                strip=kwargs['strip'], 
                                lower=kwargs['lower'], 
                                normalize=kwargs['normalize'], 
                                drop_special=kwargs['drop_special'], 
                                drop_punctuation=kwargs['drop_punctuation'])
                            
    document = word_tokenize(document)

    document = remove_stopwords(    document,
                                    extra_words=kwargs['extra_words'], 
                                    exclude_words=kwargs['exclude_words'])

    return document

def make_ngrams(document, n):
    if type(document) == pd.core.series.Series:
        document = ' '.join(document)

    words = document.split(' ')
    ngrams =  list(nltk.ngrams(words, n))
    ngrams = ['_'.join(ngram) for ngram in ngrams]
    out = ' '.join(ngrams)
    return out

def get_word_freq(document, max=None, min=None):
    if type(document) != str:
        document = ' '.join(document)
    bag =  document.split(' ')
    series = pd.Series(bag)
    vc = series.value_counts()
    if max:
        vc = vc[vc <= max]
    if min:
        vc = vc[vc >= min]
    return vc
