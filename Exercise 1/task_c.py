import torch
import matplotlib.pyplot as plt
import pandas as pd

learning_rate = 0.00000001
epoch_amount = 50000


class NonLinearRegressionModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


data = pd.read_csv("csv/day_head_circumference.csv")

x_data = data["day"].tolist()
y_data = data["headCircumference"].tolist()


# Observed/training input and output
x_train = torch.tensor(x_data, dtype=torch.float).reshape(-1, 1)

y_train = torch.tensor(y_data, dtype=torch.float).reshape(-1, 1)


model = NonLinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)

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
# Plot data points
plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')

# set x-label
plt.xlabel('Age (days)')
# set y-label
plt.ylabel('Head circumference (unknown)')

x, indices = torch.sort(x_train, 0)
plt.plot(x, model.f(x).detach(), label='$y = f(x) = 20*σ(xW+b)+31$', c="red")

plt.legend()
plt.show()