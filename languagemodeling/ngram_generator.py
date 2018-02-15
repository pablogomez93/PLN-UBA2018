from collections import defaultdict
import random


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n

        # compute the probabilities
        probs = defaultdict(dict)
        # WORK HERE!!

        for tokens in model._count.keys():
            if len(tokens) == self._n:
                token = tokens[self._n-1]
                prevs = tokens[:-1]
                probs[prevs][token] = model.cond_prob(token, tuple(prevs))

        self._probs = dict(probs)

        # sort in descending order for efficient sampling
        self._sorted_probs = sorted_probs = {}

        # WORK HERE!!
        for key in probs:
            sorted_probs[key] = sorted(probs[key].items())

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self._n

        sent = []
        prev_tokens = ['<s>'] * (n - 1)
        token = self.generate_token(tuple(prev_tokens))
        while token != '</s>':
            # WORK HERE!!
            sent.append(token)
            new_nminus1gram = prev_tokens[1:] + [token]
            prev_tokens = new_nminus1gram[(len(new_nminus1gram)-(n-1)):]
            token = self.generate_token(tuple(prev_tokens))

        return sent

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n

        if not prev_tokens:
            prev_tokens = ()
        assert len(prev_tokens) == n - 1

        probs = self._sorted_probs[prev_tokens]

        # WORK HERE!!
        return sample(probs)


def sample(problist):
    r = random.random()
    i = 0
    word, prob = problist[0]
    acum = prob
    while r > acum:
        i += 1
        word, prob = problist[i]
        acum += prob

    return word
