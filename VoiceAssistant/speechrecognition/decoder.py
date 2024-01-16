from neuralnet.utils import TextProcess
import ctcdecode
import torch

textprocess = TextProcess()

labels = [
    "'",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 27
    "_",  # 28, blank
]

def DecodeGreedy(output, blank_label=28, collapse_repeated=True):
	"""
	Decodes the output of a greedy decoder.

	Args:
		output (torch.Tensor): The output tensor from the decoder. It should have shape (batch_size, seq_length, num_classes).
		blank_label (int, optional): The label index representing the blank label. Defaults to 28.
		collapse_repeated (bool, optional): Whether to collapse repeated labels. Defaults to True.

	Returns:
		str: The decoded text sequence.
	"""
	arg_maxes = torch.argmax(output, dim=2).squeeze(1)
	decode = []
	for i, index in enumerate(arg_maxes):
		if index != blank_label:
			if collapse_repeated and i != 0 and index == arg_maxes[i -1]:
				continue
			decode.append(index.item())
	return textprocess.int_to_text_sequence(decode)

class CTCBeamDecoder:

    def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
        """
        Initializes the BeamSearchWithLM class.

        Parameters:
            beam_size (int): The size of the beam for beam search. Defaults to 100.
            blank_id (int): The index of the blank label in the labels list. Defaults to the index of '_'.
            kenlm_path (str): The path to the KenLM language model file. Defaults to None.

        Returns:
            None
        """
        print("loading beam search with lm...")
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels, alpha=0.522729216841, beta=0.96506699808,
            beam_width=beam_size, blank_id=labels.index('_'),
            model_path=kenlm_path)
        print("finished loading beam search")

    def __call__(self, output):
        """
        Calls the decoder's `decode` method to generate beam search results for the given `output` tensor.

        Args:
            output (Tensor): The output tensor of the model.

        Returns:
            str: The converted string representation of the best beam search result.
        """
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        """
        Converts a list of tokens to a string using a given vocabulary.

        Args:
            tokens (List[int]): A list of integer tokens.
            vocab (Dict[int, str]): A dictionary mapping integer tokens to strings.
            seq_len (int): The length of the sequence to be converted.

        Returns:
            str: The string representation of the tokens up to the specified sequence length.
        """
        return ''.join([vocab[x] for x in tokens[0:seq_len]])
