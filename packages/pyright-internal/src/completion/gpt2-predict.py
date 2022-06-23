import json
import sys
import torch
import time

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import RegexpTokenizer
from numpy import argmax

max_top_next = 20
max_new_tokens = 2

# to find all words that are not special characters
word_tokens = r'\w+'
regexp_tokenizer = RegexpTokenizer(word_tokens)

class Model:

	def __init__(self, model_file_path: str):
		self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_file_path, local_files_only=True, padding_side='left')
		self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_file_path, local_files_only=True)

	def predict(self, context: 'tuple[str]') -> list :

		inputs = self.tokenizer.encode(context, return_tensors="pt")
		predictions = self.beam_search(inputs, 2, 3)

		# Generate method for greedy search, only 1 sequence
		inputs = self.tokenizer.encode(context, return_tensors="pt")
		predictions_classic = self.model.generate(inputs, num_beams=3, max_new_tokens=2, num_return_sequences=3)

		return predictions

	def greedy_search(self, inputs: 'list[int]', num_new_tokens: int) -> 'list[int]':
		# Generate previous states recursively
		# TODO: Cache
		past_key_values = None
		for input in inputs:
			outputs = self.model.forward(torch.tensor([input]), past_key_values=past_key_values, use_cache=True, return_dict=True)
			past_key_values = outputs.past_key_values

		sequence: 'list[int]' = []
		while len(sequence) < num_new_tokens :
			max = outputs.logits.argmax()
			sequence.append(max.item())
			outputs = self.model.forward(max.unsqueeze(0), past_key_values=outputs.past_key_values, use_cache=True, return_dict=True)
		
		return sequence

	def beam_search(self, inputs: torch.LongTensor, num_new_tokens: int, num_beams: int) -> 'list[list[int]]':

		if num_beams < 2:
			return ValueError('num_beams should be greater or equal to 2, otherwise use greedy_search')

		# Generate previous states
		# TODO: Cache
		outputs = self.model.forward(inputs, use_cache=True, return_dict=True)
		
		# Collect top tokens from logits
		last_logits = outputs.logits[-1, -1]
		top_values, top_indices = last_logits.topk(num_beams)

		# Initialize memory
		sequences = top_indices.unsqueeze(1)
		pkv_mem = [outputs.past_key_values] * num_beams
		logit_sums = top_values

		# Recursive beam search
		for iteration in range(num_new_tokens - 1):

			# Prepare past_key_values
			N_LAYERS = 12
			past_key_values = [(torch.empty(0, 12, 3, 64), torch.empty(0, 12, 3, 64))] * N_LAYERS
			for layer, (keys, values) in enumerate(past_key_values):
				keys = torch.cat(tuple(pkv[layer][0] for pkv in pkv_mem), dim=0)
				values = torch.cat(tuple(pkv[layer][1] for pkv in pkv_mem), dim=0)
				past_key_values[layer] = (keys, values)
			
			# Prepare next inputs (end of generated sequences)
			next_inputs = sequences[:, -1].unsqueeze(1)

			# Forward pass
			outputs = self.model.forward(next_inputs, past_key_values=tuple(past_key_values), use_cache=True, return_dict=True)

			# Extract most probable tokens
			last_logits = outputs.logits[:, 0]
			top_values, top_indices = last_logits.topk(num_beams, sorted=True)
			# Multiply logits by previous logits
			new_logit_sums = torch.add(top_values, logit_sums.unsqueeze(1))

			# Generate new sequences, their probabilities and past key values
			new_sequences = torch.empty((0, iteration + 2), dtype=torch.int32)
			line_indices = [0] * num_beams
			for i in range(num_beams):
				# Find highest value for all logits, knowing that topk also does ordering
				max_sums = [new_logit_sums[i, line_indices[i]].item() for i in range(num_beams)]
				column_index = argmax(max_sums)

				# Form a new sequence with the best token and its corresponding previous sequence
				sequence = torch.cat((sequences[column_index], top_indices[column_index, line_indices[column_index]].unsqueeze(0)))
				new_sequences = torch.cat((new_sequences, sequence.unsqueeze(0)))

				# Update logit products
				logit_sums[i] = new_logit_sums[column_index, line_indices[column_index]]

				# Update past keys and values stack
				new_pkv = []
				for keys, values in past_key_values:
					layer = keys[column_index].unsqueeze(0), values[column_index].unsqueeze(0)
					new_pkv.append(layer)
				pkv_mem[i] = new_pkv

				# Consider next best element from selected context
				line_indices[column_index] += 1

			sequences = new_sequences

		return sequences

def trim(text: str) -> str:
	words = regexp_tokenizer.tokenize(text)
	if words:
		return words[0]
	else:
		return ""
		
def write_json(any):
	print(json.dumps(any), flush=True)

def main():
	[pyScriptPath, model_file_path] = sys.argv

	write_json({ "event": "state", "value": "initializing" })
	model = Model(model_file_path)
	write_json({ "event": "state", "value": "ready" })

	while True:
		try:
			x = json.loads(input())
			method = x['method']
			params = x['params']
		except:
			print("invalid input", file=sys.stderr, flush=True)

		if method == 'predict':
			context: str = params['context']
			predictions = model.predict(context)
			# predictions = list(map(trim, predictions))
			# write_json(predictions)
			print(predictions)
		if method == 'exit':
			break

if __name__ == "__main__":
	main()