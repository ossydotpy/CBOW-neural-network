import os
import numpy as np
from tqdm import tqdm
from utils import TextProcessor, DataGenerator
from model import CBOW

np.random.seed(42)

BATCH_SIZE = 128
DATA_DIR = 'datasets'
SHAKESPEAR_FILE = 'tmp_data.txt'
FULL_PATH = os.path.join(DATA_DIR, SHAKESPEAR_FILE)

with open(FULL_PATH) as f:
    data = f.read()

tmp_data = 'i am because i going in i well go far in here weel well weel'


processor = TextProcessor(context_half_size=2)
datagen = DataGenerator(context_half_size=2)

words = processor.sentence_tokenize(data)
word2idx_dict = processor.map_word2idx(words)

vocab_size = len(word2idx_dict)

model = CBOW(vocab_size, embedding_dim=28, batch_size=BATCH_SIZE)

def train_model(model: CBOW, words, word2idx_dict, num_epochs=5, batch_size=BATCH_SIZE, initial_alpha=0.03, model_name='model'):
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