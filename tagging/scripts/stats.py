"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from collections import defaultdict
import operator

from ancora import SimpleAncoraCorpusReader


class POSStats:
    """Several statistics for a POS tagged corpus.
    """

    def __init__(self, tagged_sents):
        """
        tagged_sents -- corpus (list/iterable/generator of tagged sentences)
        """
        # WORK HERE!!
        # WORK HERE!!
        # COLLECT REQUIRED STATISTICS INTO DICTIONARIES.

        words_frequency = defaultdict(int)
        tags_frequency = defaultdict(int)
        count_of_words_by_tag = defaultdict(lambda: defaultdict(int))
        count_of_tags_by_word = defaultdict(int)
        words_by_count_of_tags = defaultdict(set)
        sents_count = 0
        tokens_count = 0

        for sent in tagged_sents:
            sents_count += 1
            for t in sent:
                words_frequency[t[0]] += 1
                tags_frequency[t[1]] += 1
                tokens_count += 1
                count_of_words_by_tag[t[1]][t[0]] += 1
                count_of_tags_by_word[t[0]] += 1

        for word, tags_count in count_of_tags_by_word.items():
            words_by_count_of_tags[tags_count].add(word)

        most_frequent_tags = sorted(tags_frequency.items(), key=operator.itemgetter(1), reverse=True)[:10]
        most_frequent_tags_data = {}
        for tag, count in most_frequent_tags:
            words = count_of_words_by_tag[tag]
            words_with_count = sorted(words.items(), key=operator.itemgetter(1), reverse=True)[:5]
            most_frequent_words = list(map(lambda x: x[0], words_with_count))

            most_frequent_tags_data[tag] = {
                "count" : count,
                "percentaje" : count / tokens_count,
                "frequent_words" : most_frequent_words
            }

        # Export pre computed statistics
        self.sents_count = sents_count
        self.tokens_count = tokens_count
        self.words_frequency = words_frequency
        self.words_vocabulary = words_frequency.keys()
        self.tags_frequency = tags_frequency
        self.tags_vocabulary = tags_frequency.keys()
        self._tcount = count_of_words_by_tag
        self.words_by_count_of_tags = words_by_count_of_tags
        self.most_frequent_tags_data = most_frequent_tags_data

    def sent_count(self):
        """Total number of sentences."""
        # WORK HERE!!
        return self.sents_count

    def token_count(self):
        """Total number of tokens."""
        # WORK HERE!!
        return self.tokens_count

    def words(self):
        """Vocabulary (set of word types)."""
        # WORK HERE!!
        return self.words_vocabulary

    def word_count(self):
        """Vocabulary size."""
        # WORK HERE!!
        return len(self.words_vocabulary)

    def word_freq(self, w):
        """Frequency of word w."""
        # WORK HERE!!
        return self.words_frequency[w]

    def unambiguous_words(self):
        """List of words with only one observed POS tag."""
        # WORK HERE!!
        return self.words_by_count_of_tags[1]


    def ambiguous_words(self, n):
        """List of words with n different observed POS tags.

        n -- number of tags.
        """
        # WORK HERE!!
        return self.words_by_count_of_tags[1]

    def tags(self):
        """POS Tagset."""
        # WORK HERE!!
        return self.tags_vocabulary

    def tag_count(self):
        """POS tagset size."""
        # WORK HERE!!
        return len(self.tags_vocabulary)

    def tag_freq(self, t):
        """Frequency of tag t."""
        # WORK HERE!!
        return self.tags_frequency[t]

    def tag_word_dict(self, t):
        """Dictionary of words and their counts for tag t."""
        return dict(self._tcount[t])


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora/ancora-3.0.1es/')
    sents = corpus.tagged_sents()

    # compute the statistics
    stats = POSStats(sents)

    print('Basic Statistics')
    print('================')
    print('sents: {}'.format(stats.sent_count()))
    token_count = stats.token_count()
    print('tokens: {}'.format(token_count))
    word_count = stats.word_count()
    print('words: {}'.format(word_count))
    print('tags: {}'.format(stats.tag_count()))
    print('')

    print('Most Frequent POS Tags')
    print('======================')
    tags = [(t, stats.tag_freq(t)) for t in stats.tags()]
    sorted_tags = sorted(tags, key=lambda t_f: -t_f[1])
    print('tag\tfreq\t%\ttop')
    for t, f in sorted_tags[:10]:
        words = stats.tag_word_dict(t).items()
        sorted_words = sorted(words, key=lambda w_f: -w_f[1])
        top = [w for w, _ in sorted_words[:5]]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(t, f, f * 100 / token_count, ', '.join(top)))
    print('')

    print('Word Ambiguity Levels')
    print('=====================')
    print('n\twords\t%\ttop')
    for n in range(1, 10):
        words = list(stats.ambiguous_words(n))
        m = len(words)

        # most frequent words:
        sorted_words = sorted(words, key=lambda w: -stats.word_freq(w))
        top = sorted_words[:5]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(n, m, m * 100 / word_count, ', '.join(top)))
