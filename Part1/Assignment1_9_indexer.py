import pandas as pd
import glob, os
import sys
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def getInvertedIndex(path):
	lemmatizer	  = WordNetLemmatizer()
	StopWords	  = set(stopwords.words('english'))
	invertedIndex = dict()
	punctuations  = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''	   # punctuation marks 
	path += "/*";
	for folder in glob.glob(path):
		for file in glob.glob(folder+"/*"):
			f = open(file, 'r')
			data = ""
			for line in f:
				data+=line[:-1]
			text = data.split('<TEXT>')[1].split('</TEXT>')[0].lower()
			
			# Removing Punctuations
			for x in text: 
				if x in punctuations: 
					text = text.replace(x, " ")
			
			# Removing StopWords
			word_tokens = word_tokenize(text)
			ans = ""
			for w in word_tokens: 
				if w not in StopWords: 
					ans += w+" "
			text = ans[:-1]
			
			# Performing Lemmatization
			tokens = text.split()
			terms = []
			for token in tokens:
				terms.append(lemmatizer.lemmatize(token))

			# Adding terms to Inverted Index
			file = file.split('/')[-1]
			terms = list(set(terms))
			for term in terms:
				if(term not in invertedIndex):
					invertedIndex[term]=[]
				invertedIndex[term].append(file)
			f.close()

			# print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in invertedIndex.items()) + "}")
		print("Processing Completed till folder: "+ folder)

	with open('model_queries_9.pth', 'wb') as handle:
		pickle.dump(invertedIndex, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return

def main():
	if len(sys.argv)<2:
		print("Invalid arguments!!")
		return
	
	dataPath = sys.argv[1]
	getInvertedIndex(dataPath)

if __name__== "__main__":
	main()
