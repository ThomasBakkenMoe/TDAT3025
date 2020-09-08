import torch
import matplotlib.pyplot as plt
import torchvision


class SoftmaxModel:
    def __init__(self):
        self.W = torch.rand((784, 10), requires_grad=True)
        self.b = torch.rand((1, 10), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss function
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

if __name__ == '__main__':
    learning_rate = 0.1
    #epochs = 10000

    model = SoftmaxModel()

    #Training data
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

    #Test data
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

    #Begin training
    optimizer = torch.optim.SGD([model.W, model.b], learning_rate, momentum=0.5)

    # Train model until accuracy is above 0.91 (91%)
    epoch = 0
    while model.accuracy(x_test, y_test).item() < 0.92:
        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch: %s, Loss: %s, Accuracy: %s" % (
            epoch + 1, model.loss(x_test, y_test).item(), model.accuracy(x_test, y_test).item()))
        epoch += 1

    #Print result of training

    print("\nModel complete: Loss = %s, Accuracy: %s" % (model.loss(x_test, y_test).item(), model.accuracy(x_test, y_test).item()))

    # ****VISUALS****
    # Show the input of the first observation in the training set
    plt.imshow(x_train[0, :].reshape(28, 28))

    # Save images of W
    for i in range(10):
        plt.imsave("%i.png" % i, model.W[:, i].reshape(28, 28).detach())

    plt.show()
