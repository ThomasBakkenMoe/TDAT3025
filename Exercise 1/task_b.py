import torch
import matplotlib.pyplot as plt
import _csv

learning_rate = 0.0001
epoch_amount = 65000

day = []
length = []
weight = []

x_array = []
y_array = []
first_row = True

with open('csv/day_length_weight.csv', 'r') as file:
    reader = _csv.reader(file)
    for row in reader:
        if first_row:
            first_row = False
            continue

        day.append(float(row[0]))
        length.append(float(row[1]))
        weight.append(float(row[2]))

        y_array.append(float(row[0]))  # Add the number of days to the y training array
        print(row)

x_array = [length, weight]

# Observed/training input and output
x_train = torch.tensor(x_array).t()  # x_train = [[1], [1.5], [2], [3], [4], [5], [6]]
y_train = torch.tensor(y_array).t().reshape(-1, 1)  # y_train = [[5], [3.5], [3], [4], [3], [1.5], [2]]


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


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

figure.scatter(length, weight, day, c='red')

figure.scatter(length, weight, model.f(x_train).detach(), label='$y = f(x) = xW+b$')

# Plot labels
figure.set_xlabel('Length cm')
figure.set_ylabel('Weight kg')
figure.set_zlabel('Age in days')

plt.show()
