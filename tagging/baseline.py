from collections import defaultdict


class BadBaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        pass

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        return 'nc0s000'

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return True


class BaselineTagger:

    def __init__(self, tagged_sents, default_tag='nc0s000'):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        default_tag -- tag for unknown words.
        """
        # WORK HERE!!
        distinct_words = set()
        count_tags_by_words = defaultdict(lambda: defaultdict(int))
        for sent in tagged_sents:
            for word_tag in sent:
                distinct_words.add(word_tag[0])
                count_tags_by_words[word_tag[0]][word_tag[1]] += 1

        max_tag_by_word = defaultdict(lambda: default_tag)
        for word in count_tags_by_words.keys():
            tags = count_tags_by_words[word]

            max_tag_count, max_tag = 0, default_tag
            for tag in tags.keys():
                tag_count = tags[tag]

                if max_tag_count < tag_count:
                    max_tag_count, max_tag = tag_count, tag

            max_tag_by_word[word] = max_tag

        self.max_tag_by_word = max_tag_by_word
        self.distinct_words = distinct_words

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        # WORK HERE!!
        return self.max_tag_by_word[w]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        # WORK HERE!!
        return w not in self.distinct_words
