import torch
import torch.nn.functional as functional
import matplotlib.pyplot as plt


# The notation here is the one from Bengio et al. (2003). Names will be translated from makemore's notation
class NeuronNetwork:
    generator: any
    b: torch.tensor  # b2 in makemore's notation
    d: torch.tensor  # b1
    U: torch.tensor  # W2
    W: torch.tensor  # Zero in makemore
    H: torch.tensor  # W1
    C: torch.tensor
    amount_letters = 27  # Change if you are using another alphabet.
    mini_batch_size = 30
    theta: any

    def __init__(self, c_dimension: int = 10, nn_dimension: int = 200):
        self.generator = torch.Generator().manual_seed(2147483647)

        self.C = torch.randn((self.amount_letters, c_dimension), generator=self.generator).float()
        self.H = torch.randn((self.mini_batch_size, nn_dimension), generator=self.generator).float()  # W1
        self.d = torch.randn(nn_dimension, generator=self.generator).float()  # b1
        self.U = torch.randn((nn_dimension, self.amount_letters), generator=self.generator) .float() # W2
        self.b = torch.randn(self.amount_letters, generator=self.generator).float()  # b2
        self.W = torch.tensor(0).float()

        #self.theta = [self.b, self.d, self.W, self.U, self.H, self.C]

        self.theta = [self.b, self.d, self.U, self.H, self.C]

        for p in self.theta:
            p.requires_grad = True
        print("Bengio Model MakeMore:")
        print(f"Total parameters: {sum(p.nelement() for p in self.theta)}")

    def train(self, X_train, Y_train, train_steps: int = 200000, learning_rate: float = 0.1):
        lri = []
        loss_i = []
        step_i = []
        for i in range(train_steps):

            # minibatch construct
            ix = torch.randint(0, X_train.shape[0], (self.amount_letters,))

            # forward pass
            emb = self.C[X_train[ix]]  # (32, 3, 2)
            h = torch.tanh(emb.view(-1, self.mini_batch_size) @ self.H + self.d)  # (32, 100)
            logits = h @ self.U + self.b  # (32, 27)
            loss = functional.cross_entropy(logits, Y_train[ix])
            # print(loss.item())

            # backward pass
            for p in self.theta:
                p.grad = None
            loss.backward()
            # update
            # lr = lrs[i]
            # lr = 0.1 if i < 100000 else 0.01\
            for p in self.theta:
                # print(learning_rate, p.data)
                p.data += -learning_rate * p.grad

            # track stats
            # lri.append(lre[i])
            step_i.append(i)
            loss_i.append(loss.log10().item())
        plt.plot(step_i, loss_i)
        plt.show()

