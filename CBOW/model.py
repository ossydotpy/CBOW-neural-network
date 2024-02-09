import numpy as np
import os

from .config import SAVED_MODEL_DIR

np.random.seed(42)

class Model:

    def __init__(self, vocab_size, embedding_dim, batch_size):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim        

        self.weight_1 = np.random.rand(embedding_dim, vocab_size)
        self.bias_1 = np.random.rand(embedding_dim, 1)

        self.weight_2 = np.random.rand(vocab_size, embedding_dim)
        self.bias_2 = np.random.rand(vocab_size, 1)


    def summary(self):
        if self.inputs is None or self.targets is None:
            raise ValueError("Inputs and targets have not been set. Use the set_inputs method before calling summary.")
        
        print("CBOW Model Summary:")
        print("-------------------")
        print("Vocabulary Size:\t\t", self.vocab_size)
        print('_ '*20)
        print("Embedding Dimension:\t\t", self.embedding_dim)
        print('_ '*20)
        print("Input Shape:\t\t", self.inputs.shape)
        print('_ '*20)
        print("Target Shape:\t\t", self.targets.shape)
        print('_ '*20)
        print("Weight 1 Shape:\t\t", self.weight_1.shape)
        print('_ '*20)
        print("Bias 1 Shape:\t\t", self.bias_1.shape)
        print('_ '*20)
        print("Weight 2 Shape:\t\t", self.weight_2.shape)
        print('_ '*20)
        print("Bias 2 Shape:\t\t", self.bias_2.shape)
        print('_ '*20)

    def set_inputs(self, inputs, targets):
        self.inputs = inputs.T
        self.targets = targets.T

    def relu(self, value):
        return np.maximum(0, value)
    
    def softmax(self, value):
        return np.exp(value)/np.sum(np.exp(value), axis=0)

    def forward(self):
        self.raw_preds1 = self.weight_1.dot(self.inputs) + self.bias_1
        self.relu_preds = self.relu(self.raw_preds1)
        self.raw_preds2 = self.weight_2.dot(self.relu_preds) + self.bias_2
        self.softmax_preds = self.softmax(self.raw_preds2)
        

    def CategoricalCrossEntropy(self):
        log_probabilities = np.multiply(np.log(self.softmax_preds),self.targets )
        cost = - 1/self.batch_size * np.sum(log_probabilities)
        cost = np.squeeze(cost)
        return cost
    
    
    def back_propagate(self):
        error_term = self.weight_2.T.dot(self.softmax_preds-self.targets)
        
        error_term[self.raw_preds1 < 0] = 0

        grad_w1 = (1/self.batch_size) * error_term.dot(self.inputs.T)
        grad_w2 = (1/self.batch_size) * (self.softmax_preds-self.targets).dot(self.relu_preds.T)
        grad_b1 = (1/self.batch_size) * np.sum(error_term, axis=1, keepdims=True)
        grad_b2 = (1/self.batch_size) * np.sum(self.softmax_preds-self.targets, axis=1, keepdims=True)

        return grad_w1, grad_b1, grad_w2, grad_b2
    

    def gradient_descent(self, grad_w1, grad_b1, grad_w2, grad_b2, learning_rate=0.03):
        self.weight_1 -= (learning_rate * grad_w1)
        self.weight_2 -= (learning_rate * grad_w2)
        self.bias_1 -= (learning_rate * grad_b1)
        self.bias_2 -= (learning_rate * grad_b2)

    
    def predict(self, inputs: np.ndarray):
        x = inputs.T
        z = self.weight_1.dot(x) + self.bias_1
        activation = self.relu(z)
        output = self.weight_2.dot(activation) + self.bias_2
        y_hat = self.softmax(output)

        return np.argmax(y_hat, axis=1)

    

    def save_model(self, file_path):
        model_params = {
            'weight_1': self.weight_1,
            'bias_1': self.bias_1,
            'weight_2': self.weight_2,
            'bias_2': self.bias_2
        }
        file_path = os.path.join(SAVED_MODEL_DIR, file_path)
        np.savez(file_path, **model_params)
        print("Model saved successfully.")

    
    @classmethod
    def load_model(cls, file_path):
        model_params = np.load(file=file_path)
        model = cls(
            vocab_size=model_params['weight_1'].shape[1], 
            embedding_dim=model_params['weight_1'].shape[0], 
            batch_size=None
            )
        
        model.weight_1 = model_params['weight_1']
        model.weight_2 = model_params['weight_2']
        model.bias_1 = model_params['bias_1']
        model.bias_2 = model_params['bias_2']

        return model