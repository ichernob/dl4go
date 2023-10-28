import numpy as np

from .layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias]
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)
    
    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias
    
    def backward(self):
        data = self.get_forward_input()
        delta = self.get_backward_input()

        self.delta_b += delta
        self.delta_w += np.dot(delta, data.transpose())
        self.output_delta = np.dot(self.weight.transpose(), delta)
    
    def update_params(self, rate):
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b
    
    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)
    
    def describe(self):
        print(f"|--- {self.__class__.__name__}")
        print(f"  |-- dimensions: ({self.input_dim}, {self.output_dim})")
    