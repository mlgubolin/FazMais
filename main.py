import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from dataset_preprocessing import DatasetPreprocessing
from neuron_network import NeuronNetwork


def load_environment_parameters(env_file='.env'):
    """
    Load environment parameters from a .env file.
    """
    load_dotenv(env_file)
    return {key: os.getenv(key) for key in os.environ}


def run_makemore_original_network():
    dataset = DatasetPreprocessing('data/names.txt')

    neuron_network = NeuronNetwork(block_size=dataset.block_size, mini_batch_size=128)
    neuron_network.train(dataset.X_train, dataset.Y_train, train_steps=200000, learning_rate=0.1)
    neuron_network.train(dataset.X_train, dataset.Y_train, train_start=200000, train_steps=300000, learning_rate=0.01)
    plt.plot(neuron_network.step_i, neuron_network.loss_i)
    plt.show()


if __name__ == "__main__":
    run_makemore_original_network()
