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

lemmatizer    = WordNetLemmatizer()
StopWords     = set(stopwords.words('english'))
punctuations  = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''	   # punctuation marks 

def preProcess(text):
    text = text.lower()
    
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
    
    ans = ""
    for term in terms:
        ans += term+" "

    return ans[:-1]


def getQueires(path):
	lemmatizer	  = WordNetLemmatizer()
	StopWords	  = set(stopwords.words('english'))
	invertedIndex = dict()
	
	f = open(path, 'r')
	output = open("queries_9.txt", 'w')
	for line in f:
		if(("<num>" in line) and ("</num>" in line)):
			num = line.split("<num>")[1].split("</num>")[0]
			for nextline in f:
				title = nextline.split("<title>")[1].split("</title>")[0].lower()
				newTitle = preProcess(title)
				break
			print(num, newTitle)
			output.write(str(num)+","+str(newTitle)+"\n")
	f.close()
	return


def main():
	if len(sys.argv)<2:
		print("Invalid arguments!!")
		return
	
	dataPath = sys.argv[1]
	getQueires(dataPath)

if __name__== "__main__":
	main()
