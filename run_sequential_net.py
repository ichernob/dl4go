from data import mnist
from layers.activation_layer import ActivationLayer
from layers.dense_layer import DenseLayer
from sequential_net import SequentialNetwork

def run_sequential_network():
    train_data, val_data, test_data = mnist.load_data()

    net = SequentialNetwork()
    net.add(DenseLayer(784, 392))
    net.add(ActivationLayer(392))
    net.add(DenseLayer(392, 196))
    net.add(ActivationLayer(196))
    net.add(DenseLayer(196, 10))
    net.add(ActivationLayer(10))
    
    net.train(
        training_data=train_data,
        test_data=test_data,
        epochs=3, mini_batch_size=10, lr=3.0
    )

    # Epoch 0: 1591, 10000
    # Epoch 1: 3560, 10000
    # Epoch 2: 3834, 10000

if __name__ == "__main__":
    run_sequential_network()