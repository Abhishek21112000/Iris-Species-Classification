
import numpy as np

class MLP:
    def __init__(self, layers:list[int], hidden_activation:str='relu', output_activation:str='softmax', learning_rate:float=0.01) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        
        # Mapping from string to actual functions
        activations = {
            'relu': (self.relu, self.relu_derivative),
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'tanh': (self.tanh, self.tanh_derivative),
            'softmax': (self.softmax, None)  # Softmax usually does not need a derivative in the same way
        }
        
        self.hidden_activation, self.hidden_activation_prime = activations[hidden_activation]
        self.output_activation, _ = activations[output_activation]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def relu(self, x:np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_derivative(self, x:np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def sigmoid(self, x:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x:np.ndarray) -> np.ndarray:
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def tanh(self, x:np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def softmax(self, x:np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X:np.ndarray) -> dict[str, np.ndarray]:
        """
        Perform the forward pass of the MLP.

        :param X: Input data matrix of shape (number of samples, number of input features). Each row represents one sample
        :return: Dictionary with cached intermediate values.(activation & linear combinations) for backpropagation.
        """    
        cache = {'A0': X}
        A = X # A = activation from previous layer initailly set to X
        for i in range(len(self.weights) - 1):
            weighted_sum = A.dot(self.weights[i]) + self.biases[i]
            A = self.hidden_activation(weighted_sum)
            cache[f'weighted_sum{i + 1}'] = weighted_sum
            cache[f'A{i + 1}'] = A
        
        # Output layer with separate activation function
        weighted_sum = A.dot(self.weights[-1]) + self.biases[-1]
        A = self.output_activation(weighted_sum)
        cache[f'weighted_sum{len(self.weights)}'] = weighted_sum
        cache[f'A{len(self.weights)}'] = A
        return cache

    def backprop(self, cache:dict[str, np.ndarray], y:np.ndarray) -> None:
        """
        Perform the backpropagation algorithm to update weights and biases.

        :param cache: Dictionary with cached intermediate values from forward pass.
        :param y: True labels, not one-hot encoded.
        """
        grads = {}
        samples_in_batch = y.shape[0]
        
        # Convert labels to one-hot encoding
        one_hot_y = np.zeros_like(cache[f'A{len(self.weights)}'])
        one_hot_y[np.arange(samples_in_batch), y] = 1
        
        # Gradient of the loss with respect to the activation of the output layer
        grad_A = cache[f'A{len(self.weights)}'] - one_hot_y
        
        for i in reversed(range(len(self.weights))):
            grads[f'grad_weighted_sum{i + 1}'] = grad_A
            grads[f'grad_Weights{i + 1}'] = (1 / samples_in_batch) * cache[f'A{i}'].T.dot(grads[f'grad_weighted_sum{i + 1}'])
            grads[f'grad_bias{i + 1}'] = (1 / samples_in_batch) * np.sum(grads[f'grad_weighted_sum{i + 1}'], axis=0, keepdims=True)
            
            if i > 0:
                grad_A = grads[f'grad_weighted_sum{i + 1}'].dot(self.weights[i].T)
                grad_A *= self.hidden_activation_prime(cache[f'weighted_sum{i}'])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads[f'grad_Weights{i + 1}']
            self.biases[i] -= self.learning_rate * grads[f'grad_bias{i + 1}']

    def fit(self, X:np.ndarray, y:np.ndarray, epochs:int=10000) -> None:
        """
        Train the MLP using the given input data and labels.

        :param X: Input data matrix.
        :param y: True labels, not one-hot encoded.
        :param epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            cache = self.forward(X)
            self.backprop(cache, y)
            if epoch % 1000 == 0:
                loss = -np.mean(np.log(cache[f'A{len(self.weights)}'][np.arange(y.size), y]))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X:np.ndarray)->np.ndarray:
        """
        Predict the classes of the input data.

        :param X: Input data matrix.
        :return: Predicted labels.
        """
        cache = self.forward(X)
        return np.argmax(cache[f'A{len(self.weights)}'], axis=1)
