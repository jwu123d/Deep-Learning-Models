import torch
import torch.nn as nn
import numpy as np

with open('shakespeare.txt', 'r') as f:
    text = f.read()

# Create a mapping from characters to integers
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)  # Fix the input size here
        self.dense = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input):
        out, _ = self.lstm(input.view(1, 1, -1))
        out = out.view(-1, self.hidden_size)  # Modify the output shape here
        out = self.dense(out)
        return self.softmax(out)
    
model = LSTM(40, 200, len(chars))
temperatures = [1.5, 0.75, 0.25]  # different temperature values
seed_string = "shall i compare thee to a summer\'s day?\n"

# Convert the seed string to the model's input format
pattern = [char_to_int[char] for char in seed_string]
pattern = pattern[:40]  # truncate to the first 40 characters
model.load_state_dict(torch.load('model.pth'))

# Generate a sequence for each temperature
for temperature in temperatures:
    print(f'\n--- Temperature: {temperature} ---\n')
    model.eval()  # put the model in evaluation mode
    output = [int_to_char[value] for value in pattern]

    for i in range(400):  # generate 400 characters
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(chars))
        x = torch.from_numpy(x).type(torch.FloatTensor)
        prediction = model(x.view(1, len(x[0]), -1)) # Adjust view here

        # Apply the temperature
        prediction = prediction / temperature
        prediction = nn.functional.softmax(prediction, dim=0).detach()

        # Select a character index from the probability distribution
        index = torch.multinomial(prediction, num_samples=1).item()

        # Update the input pattern by including the predicted character and dropping the first character
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

        # Convert the integers to characters and add to the output
        output.append(int_to_char[index])

    # Print the generated poem
    print(''.join(output))