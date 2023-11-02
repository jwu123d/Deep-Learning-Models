import torch
import torch.nn as nn
import numpy as np

# Load the data
with open('shakespeare.txt', 'r') as f:
    text = f.read()

# Create a mapping from characters to integers
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Prepare the dataset of input to output pairs encoded as integers
seq_length = 40
step_size = 3  # step size for semi-redundant sequences
dataX = []
dataY = []
for i in range(0, len(text) - seq_length, step_size):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

X = np.reshape(dataX, (len(dataX), seq_length, 1))
X = X / float(len(chars))
y = np.eye(len(chars))[dataY]

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=40, hidden_size=hidden_size)  # Fix the input size here
        self.dense = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input):
        out, _ = self.lstm(input.view(1, 1, -1))
        out = self.dense(out.view(1, -1))
        return self.softmax(out)

# Create and train the model
model = LSTM(1, 200, len(chars))  # input_size is 1 because we're feeding characters one by one
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # 50 epochs
    for i, (x, y_target) in enumerate(zip(X, y)):
        inputs = torch.from_numpy(x).type(torch.FloatTensor)
        targets = torch.from_numpy(np.array([np.argmax(y_target)])).type(torch.LongTensor)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:  # print loss every 1000 steps
            print(f'Epoch: {epoch}, Step: {i}, Loss: {loss.item()}')
            
torch.save(model.state_dict(), f'model.pth')
# Function to generate poem
