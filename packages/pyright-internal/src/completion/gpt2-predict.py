import json
import sys
import torch
import time

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import RegexpTokenizer

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

		inputs = self.tokenizer.encode(context)
		predictions = self.greedy_search(inputs, 60)

		# Generate method for greedy search, only 1 sequence
		inputs = self.tokenizer.encode(context, return_tensors="pt")
		predictions_classic = self.model.generate(inputs, max_new_tokens=60)
		
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