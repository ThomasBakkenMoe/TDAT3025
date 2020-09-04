import torch
import matplotlib.pyplot as plt
import numpy as np


class SigmoidModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Cross Entropy loss function
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

if __name__ == '__main__':
    learning_rate = 0.1
    epochs = 10000

    model = SigmoidModel()

    x_train = torch.tensor([[0.0], [1.0]])
    y_train = torch.tensor([[1.0], [0.0]])

    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)

    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    #Tests
    print("Test: Not(1.0) = " + str(torch.round(model.f(torch.tensor([1.0])).detach()).item()))
    print("Test: Not(0.0) = " + str(torch.round(model.f(torch.tensor([0.0])).detach()).item()))

    fig = plt.figure()

    plot = fig.add_subplot()

    plt.plot(x_train.detach(), y_train.detach(), 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')

    plt.xlabel("x")
    plt.ylabel("y")

    out = torch.reshape(torch.tensor(np.linspace(0, 1, 100).tolist()), (-1, 1))

    plot.set_xticks([0, 1])  # x range from 0 to 1
    plot.set_yticks([0, 1])  # y range from 0 to 1

    x, indices = torch.sort(out, 0)

    # Plot sigmoid regression curve.
    plt.plot(x, model.f(x).detach())

    plt.show()
