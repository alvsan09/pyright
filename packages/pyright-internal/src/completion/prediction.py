import dill as pickle
import json
import sys

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import RegexpTokenizer

import nltk_helper as nlh

n = 3

def write_json(any):
	print(json.dumps(any), flush=True)

def load(file: str) -> any:
	with open(file, 'rb') as fin:
		model = pickle.load(fin)
	return model

def tokenize(text: str) -> list[str]:
	tokenizer = RegexpTokenizer(r'\w+|@|\!|\#|\$|\^|&|\*|\<|\>|\?|~|\||%|\.|\,|\(|\)|\[|\]|\"|\'|\`|\:|\{|\}|;|=|\/|\\|\s+|\+|-')
	return tokenizer.tokenize()

def predict(model, context: tuple[str]) -> list :
	if model == None : return []
	return nlh.top(model, context)

# def tokens() -> list[str]:
# 	if model == None : return []
# 	return []

def main():
	[pyScriptPath, pkl_file] = sys.argv

	write_json({ "event": "state", "value": "initializing" })
	model = load(pkl_file)
	write_json({ "event": "state", "value": "ready" })

	tokenizer = RegexpTokenizer(r'\w+|@|\!|\#|\$|\^|&|\*|\<|\>|\?|~|\||%|\.|\,|\(|\)|\[|\]|\"|\'|\`|\:|\{|\}|;|=|\/|\\|\s+|\+|-')

	while True:
		try:
			x = json.loads(input())
			method = x['method']
			params = x['params']
		except:
			print("invalid input", file=sys.stderr, flush=True)

		if method == 'predict':
			context: str = params['context']
			tokens = tokenizer.tokenize(context)
			last_tokens = tokens[slice(-n + 1, len(tokens))]
			predictions = predict(model, tuple(last_tokens))
			write_json(predictions)

if __name__ == "__main__":
	main()