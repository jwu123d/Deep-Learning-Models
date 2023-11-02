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
step_size = 4  # step size for semi-redundant sequences
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


# Load the state dictionary
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode if you're doing inference
model.eval()




def sample(model, start_string, length, temperature=1.0):
    # Convert the start_string to integer sequences and normalize
    start_string = [char_to_int[c] for c in start_string]
    start_string = torch.tensor(start_string).type(torch.FloatTensor).view(1, -1, 1) / float(len(chars))
    
    model.eval()  # switch the model to evaluation mode
    with torch.no_grad():  # we don't need to compute gradients
        output_string = [int_to_char[np.argmax(start_string[0, -1].numpy())]]
        for i in range(length):
            out = model(start_string[:, -seq_length:])  # Get the model's prediction based on the current string
            out_dist = out.view(-1).div(temperature).exp()  # Apply temperature and create a distribution
            top_char = torch.multinomial(out_dist, 1)[0]  # Draw a sample from this distribution
            start_string = torch.cat([start_string, torch.tensor([[top_char]]).type(torch.FloatTensor).view(1, 1, 1) / float(len(chars))], 1)  # Append this new character to our string
            output_string.append(int_to_char[top_char.item()])
    return ''.join(output_string)

# Generate a poem
print(sample(model, 'shall i compare thee to a summer\'s day?\n', 500, temperature=1.5))
print(sample(model, 'shall i compare thee to a summer\'s day?\n', 500, temperature=0.75))
print(sample(model, 'shall i compare thee to a summer\'s day?\n', 500, temperature=0.25))
