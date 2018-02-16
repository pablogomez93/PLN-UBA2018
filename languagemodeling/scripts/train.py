"""Train an n-gram model.

Usage:
  train.py [-m <model>] -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  inter: N-grams with interpolation smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import nltk

# from nltk.corpus import gutenberg
from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram


models = {
    'ngram': NGram,
    'addone': AddOneNGram,
    'inter': InterpolatedNGram
}

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpusReader = nltk.corpus.reader.plaintext.PlaintextCorpusReader
    corpus = corpusReader("../corpus/darwin", "las5obras.txt")
    sents = corpus.sents()

    # train the model
    n = int(opts['-n'])
    model = NGram(n, sents)
    model_class = models[opts['-m']]
    model = model_class(n, sents)

    # print(model.count(('de', 'cada')))
    # print(model.cond_prob("país", ["de","cada"]))

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
