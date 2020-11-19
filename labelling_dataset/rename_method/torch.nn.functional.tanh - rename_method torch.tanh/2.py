import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyVariableRNN(nn.Module):
	
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