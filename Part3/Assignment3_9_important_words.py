import numpy as np
from numpy import ma
import pandas as pd
import glob, os
import sys
import nltk
import pickle
import collections
import string
import pandas as pd
import copy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
lemmatizer    = WordNetLemmatizer()
StopWords     = set(stopwords.words('english'))
punctuations  = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''           # punctuation marks 

#==========================================================
# Loading Inverted Index
def load_invertedIndex(model_path):
    indexPath = model_path                                 # indexPath = "./model_queries_9.pth" 
    with open(indexPath, 'rb') as handle:
        invertedIndex = pickle.load(handle)
    return invertedIndex

# Get Vocabulary
def get_Vocabulary(invertedIndex):
    V = sorted(invertedIndex.keys())
    return V

# Auxiliary Function to map term to index
def get_t2i(V):
    i = 0
    term_2_i = dict()  
    for term in V:
        term_2_i[term] = i
        i+=1

    i = 0
    i_2_term = dict()    
    for term in V:
        i_2_term[i] = term
        i+=1
    
    return term_2_i, i_2_term

# Term Frequency Variants
def logTermFrequency(x):
    return ((x>0).astype('int') * (1 + ma.log10(x).filled(0)))

#==========================================================
# Document Frequency Variants
def noDocumentFrequency(x):
    return x

def idfDocumentFrequency(x, N, dft):
    return x*np.log10(N/dft)

# Normalization
def cosineNormalization(x):
    return x/np.sqrt(np.sum(x*x))    

def IDF(x, N):
    return np.log10(N/x)

# lnc.ltc Scheme
def lncVector(x):  
    l = logTermFrequency(x)
    n = noDocumentFrequency(l)
    c = cosineNormalization(n)
    return c

# lnc.ltc Scheme
def ltcVector(x, N, dtf):
    l = logTermFrequency(x)
    t = idfDocumentFrequency(l, N, dtf)
    c = cosineNormalization(t)
    return c

# cosine similarity
def cosineSimilarity(x, y):
    return np.sum(x*y, axis=1)/(np.linalg.norm(x)*np.linalg.norm(y, axis=1))

#===========================================
# function of get full path of a file
def getFullPath(docName):
    search_file = docName
    for (root,dirs,files) in os.walk('./Data/en_BDNews24', topdown=True):
        if search_file in files:
            # print(os.path.join(root, search_file))
            return os.path.join(root, search_file)

# Get vector Representation of a file
def vectorRepresentation(docName, term2i, VocabSize):
    file = getFullPath(docName)
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

    d_tf       = np.zeros(VocabSize)

    for term in terms:
        d_tf[term2i[term]] += 1
    # now we have Document-Term-Frequency-Raw-vector
    f.close()
    if(np.sum(d_tf)==0):
        # print(">>>>>>>>>>>>>>>>> Vector error (all components are zero)", docName)
        return np.zeros(VocabSize)

    dlnc = lncVector(d_tf)
    return dlnc

# function to get top words from a vector
def getTopWords(psr, i2term, top):
    indexes = np.argsort(psr)[::-1][:5]
    ans = ""
    for ind in indexes:
        ans += "," + str(i2term[ind])
    return ans+'\n'

# function o find closer words based on pseudo relevance feedback
def CloserWords(outPath, term2i, i2term, V, rankedListPath):
    print("Identifying words from pseudo relevant documents...")

    ranked_List = pd.read_csv(rankedListPath)
    out = open(outPath, 'w')
    out.write("query_id,word1,word2,word3,word4,word5\n")

    for i in range(0, 50):              # for q in queries
        print("Working for query number: ", 126+i)
        
        # get top20 docs for current query
        ranked_List.iloc[i*50: i*50 + 20,:]
        data  = ranked_List.iloc[i*50: i*50 + 20,:]
        TOP20 = list(data["Document_ID"])
        
        psr_count  = 0
        psr_sum    = np.zeros(len(V))
        for docName in TOP20:
            vr = vectorRepresentation(docName, term2i, len(V))
            if(psr_count!=10):
                psr_sum   += vr
                psr_count += 1
            
        psr  = psr_sum/psr_count 
        words = getTopWords(psr, i2term, 5)
        out.write(str(126+i)+words)    
    out.close()
    return 

#============================== MAIN FUNCTION ====================================
def main():
    if len(sys.argv)<4:
        print("Invalid arguments!!")
        return
    
    # N     : Number of documents
    # V     : Vocabulary
    # DF_t  : Document Frequency for each term, DF(t)

    dataPath   = sys.argv[1]            # dataPath   = "./Data/en_BDNews24"
    indexPath  = sys.argv[2]            # indexPath  = "./model_queries_9.pth"
    csvA       = sys.argv[3]            # csvA       = "Assignment2_9_ranked_list_A.csv"

    invertedIndex = load_invertedIndex(indexPath)
    V              = get_Vocabulary(invertedIndex)
    term2i, i2term = get_t2i(V)

    CloserWords("Assignment3_9_important_words.csv", term2i, i2term, V, csvA)

if __name__== "__main__":
    main()