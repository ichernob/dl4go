from .layer import Layer
from activations import sigmoid, sigmoid_prime



class ActivationLayer(Layer):
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim
    
    def forward(self):
        data = self.get_forward_input()
        self.output_data = sigmoid(data)
    
    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)
    
    def describe(self):
        print(f"|-- {self.__class__.__name__}")
        print(f"  |-- dimensions: ({self.input_dim}, {self.output_dim})")
