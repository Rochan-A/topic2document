import argparse
import pandas as pd

import gensim
from gensim import corpora
from gensim.models import LdaModel

import nltk, re, spacy, string

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models import CoherenceModel

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatize = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
words = set(nltk.corpus.words.words())

def preprocess(documents):
	"""
	Processes sentences before passing on to train model

	Arguments
	---------
	documents: List of reviews split into sentences

	Returns
	---------
	tokens: tokenized, de-accent and lowercased word list
	filtered: filtered numbers, symbols, stopwords etc, list of words
	"""

	# Simple tokens, de-accent and lowercase processor
	tokens = []
	for i in range(len(documents)):
		tokens.append(gensim.utils.simple_preprocess(documents[i], \
							deacc=True, min_len=3))
	filtered = []

	# POS Tagging and filtering sentences
	for i in range(len(documents)):
		doc = nlp(documents[i])
		b = []
		for tok in doc:
			if tok.is_stop != True and tok.pos_ != 'SYM' and \
				tok.tag_ != 'PRP' and tok.tag_ != 'PRP$' and \
				tok.tag_ != '_SP' and tok.pos_ != 'NUM' and \
				tok.dep_ != 'aux' and tok.dep_ != 'prep' and \
				tok.dep_ != 'det' and tok.dep_ != 'cc' and \
				tok.lemma_ != 'frac' and len(tok) != 1 and \
				tok.lemma_.lower() in words and \
				tok.lemma_.lower() not in stopwords and \
				tok.lemma_.lower() not in punctuation:
				b.append(lemmatize.lemmatize(tok.lemma_.lower()))
		filtered.append(b)
	return tokens, filtered

def pre_new(doc, dictionary):
	"""
	Preprocess a new document before infering topics

	Arguments
	---------
	doc: new documnet to preprocess
	dictionary: dictionary of the corpus used to train the lda model

	Returns
	---------
	two: preprocessed document
	"""
	one, _ = preprocess([doc])
	return dictionary.doc2bow(one[0])

def inference(df, ldamodel, dictionary):
	"""
	Run inference on a new document using a pretrained lda model

	Arguments
	---------
	inference_path: Path to the new document
	ldamodel: trained LDA model
	dictionary: Dictionary of the corpus used to train the lda model

	Returns
	---------
	l: topic list
	"""
	# Inference data
	l = []
	values = []
	for item in df:
		belong = ldamodel[pre_new(str(item), dictionary)]
		new = pd.DataFrame(belong,columns=['id','prob']).sort_values('prob',ascending=False)
		p = []
		for i, val in new.iterrows():
			p.append(dictionary.get(int(val['id'])))
		p = p[:5]
		values += p
		l.append(p)
	return l, values


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--topic_model", type=str, \
			required=True, help="Topic model file")
	parser.add_argument("--dictionary", type=str, \
			required=True, help="Dictionary file")
	parser.add_argument("--corpus", type=str, \
			required=True, help="Input corpus file")
	parser.add_argument("--output_dict", type=str, \
			required=True, help="Ouptut dictionary file")
	parser.add_argument("--output_data", type=str, \
			required=True, help="Ouptut data file")

	parser.add_argument("--title", type=int, \
			default=1, help="1 if title column should be used")
	parser.add_argument("--description", type=int, \
			default=1, help="1 if discription column should be used")
	args = parser.parse_args()

	# Load a pretrained LDA model from a file
	ldamodel = LdaModel.load(args.topic_model, mmap='r')
	dictionary = corpora.Dictionary.load(args.dictionary)

	df = pd.read_csv(args.corpus)
	df = df.drop_duplicates(subset=['resource_id'], keep='first')
	df = df['title'] + " " + df['description']

	# Infer the topic of the inferenced document
	topic_list, values = inference(df, ldamodel, dictionary)

	unique_values = list(pd.unique(values))

	import numpy as np
	np.savetxt(args.output_dict, unique_values, delimiter="\n", fmt='%s')

	save_file = []

	for i, val in enumerate(df):
		st = str(topic_list[i][0]) + " " + str(topic_list[i][1]) + " " + str(topic_list[i][2]) + " " + str(topic_list[i][3]) + " " + str(topic_list[i][4])
		save_file.append([st, val])

	save_file = pd.DataFrame(save_file, columns=['tk', 'val'])
	save_file.to_csv(args.output_data, header=True, index=False)
