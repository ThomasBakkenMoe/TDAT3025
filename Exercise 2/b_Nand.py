import torch
import matplotlib.pyplot as plt
import numpy as np


class SigmoidModel:
    def __init__(self):
        self.W = torch.rand((2, 1), dtype=torch.float, requires_grad=True)
        self.b = torch.rand((1, 1), dtype=torch.float, requires_grad=True)

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

    x_train = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    y_train = torch.tensor([[1.0], [0.0], [1.0], [1.0]])

    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)

    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    fig = plt.figure("NAND")

    plot = fig.add_subplot(111, projection='3d') #make the plot 3D

    plt.plot(x_train[:, 0].squeeze().detach(), x_train[:, 1].squeeze().detach(), y_train[:, 0].squeeze().detach(), 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$', color="green")

    x1_grid, x2_grid = torch.tensor(np.meshgrid(np.linspace(-0.25, 1.25, 10),
                                                np.linspace(-0.25, 1.25, 10)), dtype=torch.float)
    y_grid = torch.tensor(np.empty([10,10]), dtype=torch.float)

    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]])).detach()

    plot_f = plot.plot_wireframe(x1_grid, x2_grid, y_grid, color="red")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
