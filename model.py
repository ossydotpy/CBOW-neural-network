import numpy as np


class CBOW:

    def __init__(self, inputs, targets, vocab_size, embedding_dim, batch_size):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.inputs = inputs
        self.targets = targets
        

        self.weight_1 = 0.1 * np.random.rand(embedding_dim, vocab_size)
        self.bias_1 = 0.1 * np.random.rand(vocab_size, 1)

        self.weight_2 = 0.1 * np.random.rand(vocab_size, embedding_dim)
        self.bias_2 = 0.1 * np.random.rand(embedding_dim, 1)

        self.raw_preds1 = None
        self.raw_preds2 = None
        self.relu_preds = None

    def summary(self):
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


    def relu(self, value):
        return np.maximum(0, value)
    
    def softmax(self, value):
        exp_values = np.exp(value - np.max(value, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward(self):
        self.raw_preds1 = self.weight_1.T.dot(self.inputs) + self.bias_1
        self.relu_preds = self.relu(self.raw_preds1)
        self.raw_preds2 = self.weight_2.T.dot(self.relu_preds) + self.bias_2
        self.softmax_preds = self.softmax(self.raw_preds2)
        

    def CategoricalCrossEntropy(self):
        log_probabilities = np.multiply(np.log(self.softmax_preds),self.targets )
        cost = - (1/self.batch_size) * np.sum(log_probabilities)
        cost = np.squeeze(cost)
        return cost
    
    
    def back_propagate():
        pass
    # // TODO

    

    def gradient_descent(self, grad_w1, grad_b1, grad_w2, grad_b2, learning_rate=0.03):
        self.weight_1 -= (learning_rate * grad_w1)
        self.weight_2 -= (learning_rate * grad_w2)
        self.bias_1 -= (learning_rate * grad_b1)
        self.bias_2 -= (learning_rate * grad_b2)