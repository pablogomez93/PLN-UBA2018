from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def stop_word_vectorizer():
    return CountVectorizer(stop_words=set(stopwords.words('spanish')))

def binary_vectorizer():
    return CountVectorizer(binary=True)

def better_tokenizer_vectorizer():
    return CountVectorizer(tokenizer=word_tokenize)

def stemmizer_vectorizer():
    return CustomVectorizer_Stemmizer()


class CustomVectorizer_Stemmizer(CountVectorizer):
    def build_analizer(self):
        # Code extracted from: http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        # Overriding the analyzer, we have replaced "preprocessor" and "tokenizer".
        nltkSpanishStemmer = SnowballStemmer('spanish')
        analizer = super(CustomVectorizer_Stemmizer, self).build_analizer()

        # Stemmize each one.
        return lambda doc: ([nltkSpanishStemmer.stem(d) for d in analyzer(doc)])


classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}

vectorizers = {
    'stop_words': stop_word_vectorizer,
    'binary': binary_vectorizer,
    'better_tokenizer': better_tokenizer_vectorizer,
    'stemmizer': stemmizer_vectorizer,
    'countVectorizer': CountVectorizer,
}

class SentimentClassifier(object):

    def __init__(self, clf='svm', vectorizer="countVectorizer"):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        vectorizer -- vectorizer to use for the model (default: 'countVectorizer').
        """
        self._clf = clf
        self._vectorizer = vectorizers[vectorizer]()
        self._pipeline = pipeline = Pipeline([
            ('vect', vectorizers[vectorizer]()),
            ('clf', classifiers[clf]()),
        ])

    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)
