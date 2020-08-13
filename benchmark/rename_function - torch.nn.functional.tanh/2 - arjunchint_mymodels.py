import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		n_input = 178
		n_hidden = 36
		n_output = 5
		self.hidden1 = nn.Linear(n_input, n_hidden)
		self.hidden2 = nn.Linear(n_hidden, n_hidden)
		self.out = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = nn.functional.sigmoid(self.hidden1(x))
		x = nn.functional.relu(self.hidden2(x))
		x = self.out(x)
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(6, 8, 5)
		self.fc1 = nn.Linear(in_features=8 * 41, out_features=60)
		# self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(60, 5)
		self.m = nn.Dropout(p=0.25)
		
	def forward(self, x):
		x = self.pool(nn.functional.relu(self.conv1(x)))
		x = self.pool(nn.functional.relu(self.conv2(x)))
		x = x.view(-1, 8 * 41)
		x = nn.functional.relu(self.fc1(self.m(x)))
		# x = nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.fcin = nn.Linear(in_features=1, out_features=5)
		self.rnn = nn.GRU(input_size=5, hidden_size=50, num_layers=1, batch_first=True, dropout=0.5)
		self.fc = nn.Linear(in_features=50, out_features=5)

	def forward(self, x):
		x, _ = self.rnn(self.fcin(x))
		x = x[:, -1, :]
		x = self.fc(x)
		return x

		# x, _ = self.rnn(x)
		# x = nn.functional.tanh(x[:, -1, :])
		# x = self.fc(x)		
		# return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc1 = nn.Linear(in_features=dim_input, out_features=10)
		# use tanh after first layer
		self.rnn = nn.GRU(input_size=10, hidden_size=20, num_layers=3, batch_first=True, dropout=0.5)
		self.fc2 = nn.Linear(in_features=20, out_features=2)

	def last_timestep(self, unpacked, lengths):
		# Index of the last output for each sequence.
		idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),unpacked.size(2)).unsqueeze(1)
		return unpacked.gather(1, idx).squeeze()

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		x, lengths = input_tuple

		x = nn.functional.sigmoid(self.fc1(x))
		x = pack_padded_sequence(x, lengths, batch_first=True)
		x, _ = self.rnn(x)
		x, _ = pad_packed_sequence(x, batch_first=True)
		x = self.last_timestep(x, lengths)
		x = nn.functional.tanh(x)		
		x = self.fc2(x)			

		return x
		# return seqs