import sys
import pickle

def getResults(indexPath, queryPath):
	with open(indexPath, 'rb') as handle:
		invertedIndex = pickle.load(handle)

	f = open(queryPath, 'r')
	output = open("Assignment1_9_results.txt", 'w')
	for line in f:
		temp  = line.split(',')
		num   = temp[0]
		title = temp[1][:-1]
		print(num,title)
		
		queryTerms = title.split(' ')
		result = set()
		for term in queryTerms:
			if(term not in invertedIndex):
				result = []
				break
			if(len(result)==0):
				result = set(invertedIndex[term])
			else:
				result = result.intersection(invertedIndex[term])
		result = list(result)
		
		ans = ""
		for res in result:
			ans += res+" "
		ans = ans[:-1]
		output.write(str(num)+":"+ans+"\n")
	output.close()
	f.close()

def main():
	if len(sys.argv)<3:
		print("Invalid arguments!!")
		return
	
	indexPath = sys.argv[1]
	queryPath = sys.argv[2]
	getResults(indexPath, queryPath)

if __name__== "__main__":
	main()
