import os
import numpy as np
from utils import TextProcessor
from model import CBOW

np.random.seed(32)

DATA_DIR = 'datasets'
SHAKESPEAR_FILE = 'shakespeare_data.txt'
FULL_PATH = os.path.join(DATA_DIR, SHAKESPEAR_FILE)

with open(FULL_PATH) as f:
    data = f.read()

tmp_data = 'so it began, a new dawn had befalling man.'


preprocesser = TextProcessor(context_half_size=2)
words = preprocesser.sentence_tokenize(data)
word2idx_dict = preprocesser.word2idx(words)

batch_gen = preprocesser.get_batch_examples(words, word2idx_dict, batch_size=3)
tmp_input, tmp_target = next(batch_gen)

vocab_size = len(word2idx_dict)

model = CBOW(tmp_input, tmp_target, vocab_size, embedding_dim=3, batch_size=3)
model.summary()
print('forwarding...')

num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    model.forward()
    print('end forwarding...')
    print('Calculating costs...')
    cost = model.CategoricalCrossEntropy()
    print(cost)
    grad_w1, grad_b1, grad_w2, grad_b2 = model.back_propagate()

    model.gradient_descent(grad_w1, grad_b1, grad_w2, grad_b2)
        
    print(f"Epoch {epoch+1}, Loss: {total_loss}")


