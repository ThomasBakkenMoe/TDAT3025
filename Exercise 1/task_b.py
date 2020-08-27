import torch
import matplotlib.pyplot as plt
import _csv
import pandas as pd

learning_rate = 0.0001
epoch_amount = 65000


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


data = pd.read_csv("csv/day_length_weight.csv")

x_data = [data["length"].tolist(), data["weight"].tolist()]
y_data = data["day"].tolist()


# Observed/training input and output
x_train = torch.tensor(x_data, dtype=torch.float).t()

y_train = torch.tensor(y_data, dtype=torch.float).t().reshape(-1, 1)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], learning_rate)

for epoch in range(epoch_amount):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
figure = plt.figure().gca(projection='3d')

# Plot data points
figure.scatter(data["length"].tolist(), data["weight"].tolist(), data["day"].tolist(), c='red')

# Plot Regression line
figure.scatter(data["length"].tolist(), data["weight"].tolist(), model.f(x_train).detach(), label='$y = f(x) = xW+b$')

# Plot labels
figure.set_xlabel('Length (cm)')
figure.set_ylabel('Weight (kg)')
figure.set_zlabel('Age (days)')

plt.show()
