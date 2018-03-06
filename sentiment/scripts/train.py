"""Train a Sentiment Analysis model.

Usage:
  train.py [-m <model>] [-c <clf>] [-v <vectorizer>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: basemf]:
                  basemf: Most frequent sentiment
                  clf: Machine Learning Classifier
  -c <clf>      Classifier to use if the model is a MEMM [default: svm]:
                  maxent: Maximum Entropy (i.e. Logistic Regression)
                  svm: Support Vector Machine
                  mnb: Multinomial Bayes
  -v <vectorizer>        Vectorizer to use if the model is clf [default: countVectorizer]:
                    stop_words: CountVectorizer with the stop_words parameter using the nltk stopwords for spanish
                    binary: CountVectorizer with the binary parameter setted
                    better_tokenizer: CountVectorizer using the nltk tokenizer
                    stemmizer:  CountVectorizer with stemmizing, using the nltk SnowballStemmer for spanish
                    countVectorizer: Standard CountVectorizer
  -o <file>    Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from sentiment.tass import InterTASSReader, GeneralTASSReader
from sentiment.baselines import MostFrequent
from sentiment.classifier import SentimentClassifier


models = {
    'basemf': MostFrequent,
    'clf': SentimentClassifier,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load corpora
    reader1 = InterTASSReader('TASS/InterTASS/tw_faces4tassTrain1000rc.xml')
    X1, y1 = list(reader1.X()), list(reader1.y())
    reader2 = GeneralTASSReader('TASS/GeneralTASS/general-tweets-train-tagged.xml', simple=True)
    X2, y2 = list(reader2.X()), list(reader2.y())
    X, y = X1 + X2, y1 + y2

    # train model
    model_type = opts['-m']
    if model_type == 'clf':
        model = models[model_type](clf=opts['-c'], vectorizer=opts['-v'])
    else:
        model = models[model_type]()  # baseline

    model.fit(X, y)

    # save model
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
