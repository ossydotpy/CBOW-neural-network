import re
import numpy as np
from collections import Counter


def sentence_tokenize(words: str):
     words = [token for token in re.findall(r'\w+|W',words)
            if token.isalpha() or token=='.'
            ]
     return words

def word2idx(words):
    word_idx_dict = {word: idx for idx, word in enumerate(set(words))}
    return word_idx_dict

def idx2word(words):
    idx_word_dict = {idx: word for idx, word in enumerate(set(words))}
    return idx_word_dict

def get_word_freqs(words: (list|tuple|set)):
    return dict(Counter(words))

def word2vec(word, word2indx: dict):
    array = np.zeros(len(word2indx))
    array[word2indx[word]] = 1

    return array

def make_inputs_target(words, context_half_size):
    '''
    Creates a (input, target) using a sliding window
    '''
    assert(isinstance(words, (list)))

    for i in range(context_half_size, len(words) - context_half_size):
        target_word = words[i]
        context_words = words[i - context_half_size:i] + words[(i + 1) : (i + context_half_size + 1)]
        
        yield context_words, target_word


def get_training_example(words, context_half_size, word2indx):
    try:
        context_words, target_word = next(make_inputs_target(words, context_half_size))
    except StopIteration:
        raise StopIteration("No more training examples available")

    target_vector = word2vec(target_word, word2indx)
    context_vectors = np.array([word2vec(context_word, word2indx) for context_word in context_words]).mean(axis=0)
    yield context_vectors, target_vector



def get_batch_examples(words, context_half_size, word2indx, batch_size):
    examples = list(make_inputs_target(words, context_half_size))
    np.random.shuffle(examples)
    
    examples_generated = 0
    index = 0
    context_vectors_batch = []
    
    while examples_generated < batch_size:
        if index >= len(examples):
            break
        
        context_words, target_word = examples[index]
        index += 1

        target_vector = word2vec(target_word, word2indx)
        context_vector = np.array([word2vec(context_word, word2indx) for context_word in context_words]).mean(axis=0)
        
        context_vectors_batch.append(context_vector)
        
        if len(context_vectors_batch) == batch_size:
            context_vectors_batch_stacked = np.vstack(context_vectors_batch)
            context_vectors_batch = []
            
            yield context_vectors_batch_stacked, target_vector
            examples_generated += batch_size

