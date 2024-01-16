import torch


class TextProcess:
	def __init__(self):
		char_map_str = """
		' 0
		<SPACE> 1
		a 2
		b 3
		c 4
		d 5
		e 6
		f 7
		g 8
		h 9
		i 10
		j 11
		k 12
		l 13
		m 14
		n 15
		o 16
		p 17
		q 18
		r 19
		s 20
		t 21
		u 22
		v 23
		w 24
		x 25
		y 26
		z 27
		"""
		self.char_map = {}
		self.index_map = {}
		for line in char_map_str.strip().split('\n'):
			ch, index = line.split()
			self.char_map[ch] = int(index)
			self.index_map[int(index)] = ch
		self.index_map[1] = ' '

	def text_to_int_sequence(self, text):
		""" Use a character map and convert text to an integer sequence """
		int_sequence = []
		for c in text:
			if c == ' ':
				ch = self.char_map['<SPACE>']
			else:
				ch = self.char_map[c]
			int_sequence.append(ch)
		return int_sequence

	def int_to_text_sequence(self, labels):
		""" Use a character map and convert integer labels to an text sequence """
		string = []
		for i in labels:
			string.append(self.index_map[i])
		return ''.join(string).replace('<SPACE>', ' ')


textprocess = TextProcess()

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
	"""
	Given the output tensor from a neural network, the corresponding labels, and the lengths of the labels, this function performs greedy decoding to convert the output tensor into sequences of decoded labels. 

	:param output: The output tensor from the neural network. The shape of the tensor should be (batch_size, sequence_length, num_classes).
	:param labels: The corresponding labels for each input in the batch. The shape of the tensor should be (batch_size, max_label_length).
	:param label_lengths: The lengths of the labels for each input in the batch. The shape of the tensor should be (batch_size,).
	:param blank_label: The index of the blank label in the label set. The default value is 28.
	:param collapse_repeated: A flag indicating whether repeated labels should be collapsed. If set to True, repeated labels will be collapsed into a single label. The default value is True.

	:return: A tuple containing two lists: `decodes` and `targets`. `decodes` is a list of sequences of decoded labels, where each sequence corresponds to the decoded labels for a single input in the batch. `targets` is a list of sequences of target labels, where each sequence corresponds to the target labels for a single input in the batch.
	"""
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(textprocess.int_to_text_sequence(
				labels[i][:label_lengths[i]].tolist()))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(textprocess.int_to_text_sequence(decode))
	return decodes, targets
