import os
import numpy as np
from tqdm import tqdm
from CBOW import Model
from CBOW.config import DATA_DIR, load_data
from CBOW.utils import DataGenerator, TextProcessor

np.random.seed(42)
CONTEXT_HALF_SIZE=2
BATCH_SIZE = 128
FILE = 'tmp_data.txt'
FULL_PATH = os.path.join(DATA_DIR, FILE)


data = load_data(FULL_PATH)
processor = TextProcessor(context_half_size=CONTEXT_HALF_SIZE)
datagen = DataGenerator(context_half_size=CONTEXT_HALF_SIZE)

words = processor.sentence_tokenize(data)
word2idx_dict = processor.map_word2idx(words)

vocab_size = len(word2idx_dict)

model = Model(vocab_size, embedding_dim=28, batch_size=BATCH_SIZE)

def train_model(model: Model, words, word2idx_dict, num_epochs=5, batch_size=BATCH_SIZE, initial_alpha=0.03, model_name='model'):
    costs = []

    for epoch in tqdm(range(num_epochs)):
        alpha = initial_alpha

        for x, y in tqdm(datagen.get_batch_examples(words, word2idx_dict, batch_size)):
            model.set_inputs(x, y)
            model.forward()
            cost = model.CategoricalCrossEntropy()

            grad_w1, grad_b1, grad_w2, grad_b2 = model.back_propagate()
            model.gradient_descent(grad_w1, grad_b1, grad_w2, grad_b2, learning_rate=alpha)

        costs.append(cost)
        print(f'Epoch {epoch+1}/{num_epochs}[ Cost: {cost}]')

        if (epoch + 1) % 10 == 0:
            alpha *= 0.68

    model.save_model(model_name)

    return costs

costs = train_model(model, words, word2idx_dict, num_epochs=5, batch_size=BATCH_SIZE ,initial_alpha=0.03, model_name='try_1')

# def train_model(model: Model, words, word2idx_dict, num_epochs=5, batch_size=BATCH_SIZE, initial_alpha=0.03, model_name='model'):
#     costs = []

#     iteration = 0

#     for x, y in tqdm(datagen.get_batch_examples(words, word2idx_dict, batch_size)):
#         model.set_inputs(x, y)
#         model.forward()
#         cost = model.CategoricalCrossEntropy()

#         costs.append(cost)
#         if ( (iteration+1) % 10 == 0):
#             print(f'iteration {iteration+1}/{num_epochs}[ Cost: {cost}]')

#         grad_w1, grad_b1, grad_w2, grad_b2 = model.back_propagate()
#         model.gradient_descent(grad_w1, grad_b1, grad_w2, grad_b2, learning_rate=initial_alpha)

#         iteration+=1
#         if iteration == num_epochs: 
#             break
#         if iteration % 100 == 0:
#             initial_alpha *= 0.66

#     model.save_model(model_name)

#     return costs