# PyTorch example performing linear regression

## Prerequisites

Python3 should be installed, as well as the following python libraries: numpy, matplotlib and torch.

On Arch Linux/Manjaro you can install these packages using `pacman`:

```sh
pip3 install python-numpy python-matplotlib python-pytorch
```

On other systems `pip3` can be used:

```sh
pip3 install numpy matplotlib torch torchvision
```

Note that CUDA optimized packages exist in case you have an Nvidia graphics card.

## Run example

```sh
python3 main.py
```

## Introduction

PyTorch is a machine learning library that include popular machine learning methods, but also reuses
much of the Numpy library API. Although, there are some differences:

```py
import torch
import numpy as np

# In Numpy one would use a multidimensional array to represent a matrix:
print(np.array([[1, 2], [3, 4]]))
# Output:
# [[1 2]
#  [3 4]]

# In PyTorch one instead use tensor, which basically is the same as np.array, to represent a matrix:
print(torch.tensor([[1, 2], [3, 4]]))
# Output:
# tensor([[1, 2],
#         [3, 4]])

# However, the usage of functions like matrix multiplication are similar:
print(np.array([[1, 2], [3, 4]]) @ np.array([[1, 2], [3, 4]]))
# Output:
# [[ 7 10]
#  [15 22]]
print(torch.tensor([[1, 2], [3, 4]]) @ torch.tensor([[1, 2], [3, 4]]))
# Output:
# tensor([[ 7 10]
#         [15 22]])
```

## Transforming data

### Reshape

The function `reshape()` is a commonly used in numpy/pytorch, and is very useful when working with
datasets:

```py
import torch

# Create a 2x2 matrix from a 1 dimensional table:
print(torch.reshape(torch.tensor([1, 2, 3, 4]), (2, 2)))
# Output:
# tensor([[1, 2],
#         [3, 4]])

# Can also specify one dimension as -1, and let reshape infer this dimension size
print(torch.reshape(torch.tensor([1, 2, 3, 4]), (-1, 2)))
# Output:
# tensor([[1, 2],
#         [3, 4]])
print(torch.reshape(torch.tensor([1, 2, 3, 4, 5, 6]), (-1, 2)))
# Output:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6])

# Another possibility is to convert an 1 dimensional table to a matrix with a single row:
print(torch.reshape(torch.tensor([1, 2, 3, 4, 5, 6]), (1, -1)))
# Output:
# tensor([[1, 2, 3, 4, 5, 6]])

# Or an 1 dimensional table to a single column matrix:
print(torch.reshape(torch.tensor([1, 2, 3, 4, 5, 6]), (-1, 1)))
# Output:
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6]])

# You can also reshape a matrix to a vector (a method also called squeeze):
print(torch.reshape(torch.tensor([[1, 2, 3, 4, 5, 6]]), (-1, )))
# Output:
# tensor([1, 2, 3, 4, 5, 6])

# You can call reshape, and many other methods, on the tensor objects directly:
print(torch.tensor([[1, 2, 3, 4, 5, 6]]).reshape(-1, ))
# Output:
# tensor([1, 2, 3, 4, 5, 6])

# Reshape can act as transpose (T), but the resulting matrix shape is specified and easier to see
print(torch.tensor([[1, 2, 3, 4, 5, 6]]).T)
print(torch.tensor([[1, 2, 3, 4, 5, 6]]).reshape(-1, 1))
# Both outputs:
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6]])

```

### Indexing

Although not used in main.py, PyTorch also supports Numpy-style indexing:

```py
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Read single value:
print(A[0, 1])
# Output:
# tensor(2)

# Read second row:
print(A[1, :])
# Output:
# tensor([4, 5, 6])

# Read first column:
print(A[:, 0])
# Output:
# tensor([1, 4])

# Read first and last columns:
print(A[:, [0, 2]])
# Output:
# tensor([[1, 3],
#         [4, 6]])
```

## Linear regression

A linear regression model `y = f(x) = xW + b` is found by minimizing a loss function
`torch.mean(torch.square(f(x_train) - y_train))` called Mean Squared Error, where `x_train` and
`y_train` are the observed input and output we want to model. To minimize the loss function in
PyTorch, the model variables `W` and `b` are adjusted through a method called Stochastic Gradient
Descent (`torch.optim.SGD`). After the model variables `W` and `b` have converged, we can make new
`y` predictions from `x` using the model `f(x)`.

## Further reading

There are many PyTorch tutorials, but one that goes into slightly more detail is
[Understanding PyTorch with an example: a step-by-step tutorial](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
