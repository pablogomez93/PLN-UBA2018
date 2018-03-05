from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


class CustomVectorizer_Stemmizer(CountVectorizer):
    def build_analizer(self):
        # Code extracted from: http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        # Overriding the analyzer, we have replaced "preprocessor" and "tokenizer".
        nltkSpanishStemmer = SnowballStemmer('spanish')
        analizer = super(CustomVectorizer_Stemmizer, self).build_analizer()

        # Stemmize each one.
        return lambda doc: ([nltkSpanishStemmer.stem(d) for d in analyzer(doc)])


class SentimentClassifier(object):

    def __init__(self, clf='svm'):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        self._clf = clf
        self._pipeline = pipeline = Pipeline([
            #('vect', CountVectorizer(stop_words=set(stopwords.words('spanish')), binary=True, tokenizer=word_tokenize)),
            ('vect', CustomVectorizer_Stemmizer()),
            ('clf', classifiers[clf]()),
        ])

    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)
