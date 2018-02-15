# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math
import operator
import functools


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        result = 0.0
        for i, sent in enumerate(sents):
            lp = self.sent_log_prob(sent)
            if lp == -math.inf:
                return lp
            result += lp
        return result

    def cross_entropy(self, sents):
        log_prob = self.log_prob(sents)
        n = sum(len(sent) + 1 for sent in sents)  # count '</s>' events
        e = - log_prob / n
        return e

    def perplexity(self, sents):
        return math.pow(2.0, self.cross_entropy(sents))


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)
        tokens_count = 0

        # WORK HERE!!
        for sent in sents:
            sent = self.add_start_and_end_of_sentence(sent)

            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i:i+n])
                count[ngram] += 1
                count[ngram[:-1]] += 1

            tokens_count += len(sent)

        self._tokens_count = tokens_count
        self._count = dict(count)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        # WORK HERE!!
        if prev_tokens is not None:
            numerator = self.count(prev_tokens + (token,))
            denominator = self.count(prev_tokens)
        else:
            numerator = self.count((token,))
            denominator = self._tokens_count

        return 0.0 if denominator == 0 else numerator / denominator

    def prob_of_each_ngram_in_a_sentence(self, sent):
        estimated_probabilities = []

        for i in range(len(sent) - self._n + 1):
            ngram_as_a_list = sent[i:i+self._n]

            token = ngram_as_a_list[self._n-1]
            prev_tokens = tuple(ngram_as_a_list[0:self._n-1])

            ngram_prob = self.cond_prob(token, prev_tokens)
            estimated_probabilities.append(ngram_prob)

        return estimated_probabilities

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        # WORK HERE!!
        sent = self.add_start_and_end_of_sentence(sent)

        estimated_probabilities = self.prob_of_each_ngram_in_a_sentence(sent)
        return functools.reduce(operator.mul, estimated_probabilities, 1.0)

    def add_start_and_end_of_sentence(self, sent):
        """ If n is the order of the model, add <s> n-1 times before the sentence
            and </s> and the end of it

            sent -- the sentence as a list of tokens
        """
        return (["<s>"]*(self._n-1)) + sent + ["</s>"]

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        # WORK HERE!!
        sent = self.add_start_and_end_of_sentence(sent)

        probs = self.prob_of_each_ngram_in_a_sentence(sent)
        log_probs = list(map(extended_log2, probs))

        return functools.reduce(operator.add, log_probs, 0.0)


def extended_log2(x):
    return -math.inf if x <= 0 else math.log(x, 2)
