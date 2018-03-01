from featureforge.vectorizer import Vectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from tagging.features import (History, word_lower, word_istitle, word_isupper,
                              word_isdigit, NPrevTags, PrevWord, NextWord)


classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


class MEMM:

    def __init__(self, n, tagged_sents, clf='svm'):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        # 1. build the pipeline
        # WORK HERE!!
        self.n = n
        basic_features = [word_lower, word_istitle, word_isupper, word_isdigit]
        features = basic_features + [cf(f) for f in basic_features for cf in [PrevWord, NextWord]] + [NPrevTags(i) for i in range(1, self.n+1)]
        vect = Vectorizer(features)

        self._pipeline = pipeline = Pipeline([
            ('vect', vect),
            ('clf', classifiers[clf]())
        ])

        # 2. train it
        print('Training classifier...')
        tagged_sents_list = list(tagged_sents)
        X = self.sents_histories(tagged_sents_list)
        y = self.sents_tags(tagged_sents_list)

        pipeline.fit(list(X), list(y))

        # 3. build known words set
        # WORK HERE!!
        voc = set()
        for sent in tagged_sents:
            for word, tag in sent:
                voc.add(word)

        self.vocabulary = voc

    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        for sent in tagged_sents:
            for h in self.sent_histories(sent):
                yield h

    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        prev_tags = ('<s>',) * (self.n - 1)
        sent = [w for w, _ in tagged_sent]
        for i, (w, t) in enumerate(tagged_sent):
            yield History(sent, prev_tags, i)
            prev_tags = (prev_tags + (t,))[1:]

    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        for sent in tagged_sents:
            for t in self.sent_tags(sent):
                yield t

    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        return (t for _, t in tagged_sent)

    def tag(self, sent):
        """Tag a sentence using beam inference with beam of size 1.

        sent -- the sentence.
        """
        # WORK HERE!!
        ret_tag = []

        for i in range(len(sent)):
            if i == 0:
                prevs = tuple("<s>") * (self.n - 1)
            else:
                prevs = (prevs + tuple(ret_tag[i-1]))[1:]

            history = History(sent, prevs, i)
            ret_tag += [self.tag_history(history)]

        return ret_tag

    def tag_history(self, h):
        """Tag a history.

        h -- the history.
        """
        # WORK HERE!!
        return self._pipeline.predict([h])[0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        # WORK HERE!!
        return w not in self.vocabulary
