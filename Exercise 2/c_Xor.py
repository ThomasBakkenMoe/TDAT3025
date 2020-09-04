import torch
import matplotlib.pyplot as plt
import numpy as np


class SigmoidModel:
    def __init__(self):

        # Non-functional model variables
        '''
        self.W1 = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float, requires_grad=True)
        self.b1 = torch.tensor([[0.0, 0.0]], dtype=torch.float, requires_grad=True)
        self.W2 = torch.tensor([[0.0], [0.0]], dtype=torch.float, requires_grad=True)
        self.b2 = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)
        '''


        # Functional model variables
        self.W1 = torch.tensor([[1.0, 0.5], [0.5, 0.6]], dtype=torch.float, requires_grad=True)
        self.b1 = torch.tensor([[1.0, 1.0]], dtype=torch.float, requires_grad=True)

        self.W2 = torch.tensor([[1.0], [1.0]], dtype=torch.float, requires_grad=True)
        self.b2 = torch.tensor([[1.0]], dtype=torch.float, requires_grad=True)


    def logits(self, x):
        return x @ self.W2 + self.b2

    # First layer
    def layer1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    # Second layer
    def layer2(self, x):
        return torch.sigmoid(x @ self.W2 + self.b2)

    # predictor
    def f(self, x):
        return self.layer2(self.layer1(x))

    # Cross Entropy loss function
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(self.layer1(x)), y)

if __name__ == '__main__':
    learning_rate = 5.0
    epochs = 5000

    model = SigmoidModel()

    x_train = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y_train = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

    optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], learning_rate)

    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("W = %s, b = %s, loss = %s" % (model.W1, model.b1, model.loss(x_train, y_train)))
    print("W = %s, b = %s, loss = %s" % (model.W2, model.b2, model.loss(x_train, y_train)))
    print("[0,1] is 1 = ", round(model.f(torch.tensor([0.0, 1.0])).item()))

    fig = plt.figure("XOR")

    plot = fig.add_subplot(111, projection='3d') #make the plot 3D

    plt.plot(x_train[:, 0].squeeze().detach(), x_train[:, 1].squeeze().detach(), y_train[:, 0].squeeze().detach(), 'o', label='$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$', color="red")

    x1_grid, x2_grid = torch.tensor(np.meshgrid(np.linspace(-0.25, 1.25, 10),
                                                np.linspace(-0.25, 1.25, 10)), dtype=torch.float)
    y_grid = torch.tensor(np.empty([10,10]), dtype=torch.float)

    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]])).detach()

    plot_f = plot.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green",
                                 label="$y=f(x)=\\sigma(xW+b)$")

    plot_info = fig.text(0.01, 0.02, "$W1=\\genfrac{[}{]}{0}{}{%.2f}{%.2f}$\n"
                                      "$b1=[%.2f]$\n"
                                      "$W2=\\genfrac{[}{]}{0}{}{%.2f}{%.2f}$\n"
                                      "$b2=[%.2f]$\n"
                                      "$\n$loss = %.2f$"
                          % (model.W1[0, 0], model.W1[1, 0], model.b1[0, 0], model.W2[0, 0], model.W2[1, 0],
                             model.b2[0, 0],
                             model.loss(x_train, y_train)))

    plot.set_xlabel("$x_1$")
    plot.set_ylabel("$x_2$")
    plot.set_zlabel("$y$")
    plot.legend(loc="upper left")
    plot.set_xticks([0, 1])
    plot.set_yticks([0, 1])
    plot.set_zticks([0, 1])
    plot.set_xlim(-0.25, 1.25)
    plot.set_ylim(-0.25, 1.25)
    plot.set_zlim(-0.25, 1.25)


    table = plt.table(cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                      colWidths=[0.1] * 3,
                      colLabels=["$x_1$", "$x_2$", "$f(x)$"],
                      cellLoc="center",
                      loc="lower right")
    table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[0., 0.]], dtype=torch.float)).detach())
    table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[0., 1.]], dtype=torch.float)).detach())
    table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[1., 0.]], dtype=torch.float)).detach())
    table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[1., 1.]], dtype=torch.float)).detach())

    plot_f.remove()
    x1_grid, x2_grid = torch.tensor(np.meshgrid(np.linspace(-0.25, 1.25, 10),
                                                np.linspace(-0.25, 1.25, 10)), dtype=torch.float)
    y_grid = torch.tensor(np.empty([10, 10]), dtype=torch.float)
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]])).detach()
    plot1_f = plot.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

    plot.set_proj_type('ortho')

    plt.show()
