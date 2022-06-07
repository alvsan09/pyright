import dill as pickle
import json
import sys
import re

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import RegexpTokenizer

max_top_next = 20
max_new_tokens = 2

# to find all words that are not special characters
word_tokens = r'\w+'
regexp_tokenizer = RegexpTokenizer(word_tokens)

def write_json(any):
	print(json.dumps(any), flush=True)

def load(model_file_path: str) -> tuple:
	tokenizer = GPT2Tokenizer.from_pretrained(model_file_path, local_files_only=True, padding_side='left')
	model = GPT2LMHeadModel.from_pretrained(model_file_path, local_files_only=True)
	return model, tokenizer

def predict(model, tokenizer, context: tuple[str]) -> list :
	inputs = tokenizer.encode(context, return_tensors='pt')
	outputs: list[str] = model.generate(inputs, num_return_sequences=max_top_next, num_beams=max_top_next, num_beam_groups=max_top_next,
                           diversity_penalty=float(max_top_next + 1), max_new_tokens=max_new_tokens)
	return list(map(lambda output: tokenizer.decode(output)[len(context):], outputs))

def trim(text: str) -> str:
	words = regexp_tokenizer.tokenize(text)
	if words:
		return words[0]
	else:
		return ""

def main():
	[pyScriptPath, model_file_path] = sys.argv

	write_json({ "event": "state", "value": "initializing" })
	model, tokenizer = load(model_file_path)
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
			predictions = predict(model, tokenizer, context)
			predictions = list(map(trim, predictions))
			write_json(predictions)

if __name__ == "__main__":
	main()