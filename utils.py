import re
import numpy as np
from collections import Counter

class TextProcessor:
    def __init__(self, context_half_size):
        self.context_half_size = context_half_size

    def sentence_tokenize(self, words: str):
        words = [token for token in re.findall(r'\w+|W',words)
                if token.isalpha() or token=='.'
                ]
        return words

    def word2idx(self, words):
        word_idx_dict = {word: idx for idx, word in enumerate(set(words))}
        return word_idx_dict

    def idx2word(self, words):
        idx_word_dict = {idx: word for idx, word in enumerate(set(words))}
        return idx_word_dict

    def get_word_freqs(self, words: (list|tuple|set)):
        return dict(Counter(words))

    def word2vec(self, word, word2indx: dict):
        array = np.zeros(len(word2indx))
        array[word2indx[word]] = 1

        return array

    def make_inputs_target(self, words):
        '''
        Creates a (input, target) using a sliding window
        '''
        assert(isinstance(words, (list)))

        for i in range(self.context_half_size, len(words) - self.context_half_size):
            target_word = words[i]
            context_words = words[i - self.context_half_size:i] + words[(i + 1) : (i + self.context_half_size + 1)]
            
            yield context_words, target_word

    def get_training_example(self, words, word2indx):
        try:
            context_words, target_word = next(self.make_inputs_target(words))
        except StopIteration:
            raise StopIteration("No more training examples available")

        target_vector = self.word2vec(target_word, word2indx)
        context_vectors = np.array([self.word2vec(context_word, word2indx) for context_word in context_words]).mean(axis=0)
        return context_vectors, target_vector

    def get_batch_examples(self, words, word2indx, batch_size):
        examples = list(self.make_inputs_target(words))
        np.random.shuffle(examples)

        index = 0
        while index + batch_size <= len(examples):
            context_vectors_batch = []
            for i in range(batch_size):
                context_words, target_word = examples[index + i]
                target_vector = self.word2vec(target_word, word2indx)
                context_vector = np.array([self.word2vec(context_word, word2indx) for context_word in context_words]).mean(axis=0)
                context_vectors_batch.append(context_vector)
            yield np.vstack(context_vectors_batch), np.array([self.word2vec(target_word, word2indx) for _, target_word in examples[index:index+batch_size]])
            index += batch_size

        if index < len(examples):
            context_vectors_batch = []
            for i in range(index, len(examples)):
                context_words, target_word = examples[i]
                target_vector = self.word2vec(target_word, word2indx)
                context_vector = np.array([self.word2vec(context_word, word2indx) for context_word in context_words]).mean(axis=0)
                context_vectors_batch.append(context_vector)
            yield np.vstack(context_vectors_batch), np.array([self.word2vec(target_word, word2indx) for _, target_word in examples[index:]])
