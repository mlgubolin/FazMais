import torch
import random

class DatasetPreprocessing:

    def __init__(self, filename: str):
        self.X_train = None
        self.Y_train = None
        self.X_dev = None
        self.Y_dev = None
        self.X_test = None
        self.Y_test = None
        self.itos = None
        self.stoi = None
        self.words = None
        self.block_size = 3

        self.load_file(filename)
        self.separate_for_training()

    def load_file(self, filename: str):
        self.words = open(filename, 'r').read().splitlines()
        print(f"File {filename} loaded.")
        print(f"{self.words[:5]}")

        chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}

    def build_dataset(self, subset_words):
        X_temp, Y_temp = [], []
        for w in subset_words:

            # print(w)
            context = [0] * self.block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X_temp.append(context)
                Y_temp.append(ix)
                # print(''.join(itos[i] for i in context), '--->', itos[ix])
                context = context[1:] + [ix]  # crop and append

        return torch.tensor(X_temp), torch.tensor(Y_temp)

    def separate_for_training(self, seed=42):
        random.seed(seed)
        random.shuffle(self.words)
        n1 = int(0.8 * len(self.words))
        n2 = int(0.9 * len(self.words))

        self.X_train, self.Y_train = self.build_dataset(self.words[:n1])
        self.X_dev, self.Y_dev = self.build_dataset(self.words[n1:n2])
        self.X_test, self.Y_test = self.build_dataset(self.words[n2:])
