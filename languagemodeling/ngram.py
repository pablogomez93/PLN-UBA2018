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
            sent = self.add_start_and_end_of_sentence(sent, n)

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
        if not prev_tokens:
            # if prev_tokens not given, assume 0-uple:
            prev_tokens = ()
        assert len(prev_tokens) == self._n - 1

        return self.cond_prob_ml(token, prev_tokens)

    def cond_prob_ml(self, token, prev_tokens=None):
        """Conditional probability of a token using the maximum likelihood estimation

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if not prev_tokens:
            # if prev_tokens not given, assume 0-uple:
            prev_tokens = ()

        # WORK HERE!!
        numerator = self.count(prev_tokens + (token,))
        denominator = self.count(prev_tokens)

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
        sent = self.add_start_and_end_of_sentence(sent, self._n)

        estimated_probabilities = self.prob_of_each_ngram_in_a_sentence(sent)
        return functools.reduce(operator.mul, estimated_probabilities, 1.0)

    def add_start_and_end_of_sentence(self, sent, ngram_order):
        """ If n is the order of the model, add <s> n-1 times before the sentence
            and </s> and the end of it

            sent -- the sentence as a list of tokens
        """
        return (["<s>"]*(ngram_order-1)) + sent + ["</s>"]

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        # WORK HERE!!
        sent = self.add_start_and_end_of_sentence(sent, self._n)

        probs = self.prob_of_each_ngram_in_a_sentence(sent)
        log_probs = list(map(extended_log2, probs))

        return functools.reduce(operator.add, log_probs, 0.0)


def extended_log2(x):
    return -math.inf if x <= 0 else math.log(x, 2)


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = voc = set()
        # WORK HERE!!
        compute_vocabulary(sents, voc)

        self._V = len(voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            # if prev_tokens not given, assume 0-uple:
            prev_tokens = ()
        assert len(prev_tokens) == n - 1

        # WORK HERE!!
        if prev_tokens is not None:
            numerator = self.count(prev_tokens + (token,)) + 1
            denominator = self.count(prev_tokens) + self.V()
        else:
            numerator = self.count((token,))
            denominator = self._tokens_count

        return 0.0 if denominator == 0 else numerator / denominator


def compute_vocabulary(sents, voc):
    for sent in sents:
        for token in sent:
            voc.add(token)
    voc.add("</s>")


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        # WORK HERE!!
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N
        count_kgrams = defaultdict(int)

        for sent in train_sents:

            # 0-GRAMS
            count_kgrams[tuple()] += len(sent) + 1  # + 1 because the '</s>'

            # K-GRAMS WITH 1 <= K <= N
            for j in range(1, n+1):
                sent = self.add_start_and_end_of_sentence(sent, j)

                for i in range(len(sent) - j + 1):
                    ngram = tuple(sent[i:i+j])
                    count_kgrams[ngram] += 1

        self.count_kgrams = count_kgrams

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            # WORK HERE!!
            compute_vocabulary(train_sents, voc)

            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # use grid search to choose gamma
            min_gamma, min_p = None, float('inf')

            # WORK HERE!! TRY DIFFERENT VALUES BY HAND:
            for gamma in [i/100 for i in range(1, 100)]:
                self._gamma = gamma
                p = self.perplexity(held_out_sents)
                print('  {} -> {}'.format(gamma, p))

                if p < min_p:
                    min_gamma, min_p = gamma, p

            print('  Choose gamma = {}'.format(min_gamma))
            self._gamma = min_gamma

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        # WORK HERE!! (JUST A RETURN STATEMENT)
        return self.count_kgrams.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            # if prev_tokens not given, assume 0-uple:
            prev_tokens = ()
        assert len(prev_tokens) == n - 1

        # WORK HERE!!
        # SUGGESTED STRUCTURE:
        prob = 0.0
        cum_lambda = 0.0  # sum of previous lambdas

        for i in range(n):
            # i-th term of the sum
            if i < n - 1:
                # COMPUTE lambdaa AND cond_ml.
                igram_count = self.count(prev_tokens[i:])
                lambdaa = (1-cum_lambda)*igram_count/(igram_count+self._gamma)

                cond_ml = self.cond_prob_ml(token, prev_tokens[i:])

            else:
                # COMPUTE lambdaa AND cond_ml.
                lambdaa = (1 - cum_lambda)

                # LAST TERM: USE ADD ONE IF NEEDED!
                if self._addone:
                    num = self.count((token,)) + 1
                    denom = self.count(tuple()) + self._V

                    cond_ml = 0.0 if denom == 0 else num / denom

                else:
                    cond_ml = self.cond_prob_ml(token)

            prob += lambdaa * cond_ml
            cum_lambda += lambdaa

        return prob
