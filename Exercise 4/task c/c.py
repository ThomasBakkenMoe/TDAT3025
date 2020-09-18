import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[-1].reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '  0
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' 1
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' 2
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' 3
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r' 4
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c' 5
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'f' 6
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'l' 7
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'm' 8
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'p' 9
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 's' 10
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'o' 11
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'n' 12
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # '🎩' 13
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # '🐀' 14
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # '🐈' 15
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # '🏢' 16
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # '👨' 17
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # '🧢' 18
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  # '👦' 19
]
encoding_size = len(char_encodings)

index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n', '🎩', '🐀', '🐈', '🏢', '👨', '🧢', '👦']

x_train = torch.tensor([[char_encodings[1], char_encodings[4], char_encodings[5], char_encodings[6], char_encodings[8], char_encodings[5], char_encodings[10]],
                        [char_encodings[2], char_encodings[2], char_encodings[2], char_encodings[7], char_encodings[2], char_encodings[2], char_encodings[11]],
                        [char_encodings[3], char_encodings[3], char_encodings[3], char_encodings[2], char_encodings[3], char_encodings[9], char_encodings[12]],
                        [char_encodings[0], char_encodings[0], char_encodings[0], char_encodings[3], char_encodings[3], char_encodings[0], char_encodings[0]]])

y_train = torch.tensor([char_encodings[13],
                        char_encodings[14],
                        char_encodings[15],
                        char_encodings[16],
                        char_encodings[17],
                        char_encodings[18],
                        char_encodings[19]])

model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)

def generate(input_text):
    # Generate emoji from the inputText
    model.reset(1)
    letters = []
    for char in input_text:

        help_list = []
        help_list.append(char_encodings[index_to_char.index(char)])  # Translate chars to index values
        letters.append(help_list)

    letters.append(help_list)
    letters = torch.tensor(letters)
    y = model.f(letters)
    return index_to_char[y.argmax(1)]  # Return the emoji (or char if something goes wrong) that the model finds most likely


for epoch in range(1000):
    model.reset(x_train.size(1))
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 50 == 9:
        print("Epoch %s: Loss: %s | 'hat '=>%s, 'rat'=>%s, 'cat'=>%s, 'flat'=>%s, 'matt'=>%s, 'cap'=>%s, 'son'=>%s 'rt  '=>%s, 'mt  '=>%s"
              % (epoch, loss.item(), generate('hat '), generate('rat '), generate("cat"), generate("flat"), generate("matt"), generate("cap "), generate("son "), generate('rt  '), generate('mt  ')))
