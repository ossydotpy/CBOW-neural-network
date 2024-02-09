import os
import numpy as np
from CBOW.config import DATA_DIR, load_data
from CBOW.utils import TextProcessor


FILE = 'tmp_data.txt'
FULL_PATH = os.path.join(DATA_DIR, FILE)


data = load_data(FULL_PATH)

processor = TextProcessor(context_half_size=2)
words = processor.sentence_tokenize(data)
word2idx_dict = processor.map_word2idx(words)

embs_file = np.load('saved_weights/embeddings.npz')
embs = embs_file['embeddings']
vocab = processor.get_vocabulary(words=words)

word_emb_dict = {}
for word in vocab:
    word_idx = word2idx_dict[word]
    word_emb_dict[word] = embs[:,word_idx]

print(word_emb_dict)