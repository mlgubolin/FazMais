import os
import torch

import os
from dotenv import load_dotenv

from dataset_preprocessing import DatasetPreprocessing
from neuron_network import NeuronNetwork


def load_environment_parameters(env_file='.env'):
    """
    Load environment parameters from a .env file.
    """
    load_dotenv(env_file)
    return {key: os.getenv(key) for key in os.environ}


if __name__ == "__main__":
    dataset = DatasetPreprocessing('data/names.txt')

    neuron_network = NeuronNetwork()
    neuron_network.train(dataset.X_train,dataset.Y_train, 2000, 0.1)