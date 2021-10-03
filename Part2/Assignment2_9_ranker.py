import numpy as np
from numpy import ma
import pandas as pd
import glob, os
import sys
import nltk
import pickle
import collections
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer    = WordNetLemmatizer()
StopWords     = set(stopwords.words('english'))
punctuations  = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''           # punctuation marks 


# ========================================================
# Loading Inverted Index
def load_invertedIndex(model_path):
    indexPath = model_path                                 # indexPath = "./model_queries_9.pth" 
    with open(indexPath, 'rb') as handle:
        invertedIndex = pickle.load(handle)
    return invertedIndex

# ========================================================
# Get Vocabulary
def get_Vocabulary(invertedIndex):
    V = sorted(invertedIndex.keys())
    return V

# ========================================================
# Function to get total number of Documents in given Dataset
def get_NumberOfDocs(Data_path):
    i = 0
    path = Data_path                                       # path = "./Data/en_BDNews24/*"
    for folder in glob.glob(path+"/*"):
        for file in glob.glob(folder+"/*"):
            i+=1
    return i

# ========================================================
# Auxiliary Function to map term to index
def get_t2i(V):
    i = 0
    term_2_i = dict()    
    for term in V:
        term_2_i[term] = i
        i+=1
    return term_2_i

# ========================================================
# Function to get Document Frequency vector
def get_DFt(N, V, invertedIndex):
    i = 0
    DF_t  = np.zeros(len(V))
    for term in V:
        DF_t[i] = len(invertedIndex[term])
        i+=1
        
    return DF_t

#=========================================================
# Function to lemmatize and remove Punctuations, StopWords 
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

#=========================================================
# Query Preprocessing
def PreProcessQueries(path, outname):  
    print("Preprocessing Queries...")
    lemmatizer    = WordNetLemmatizer()
    StopWords     = set(stopwords.words('english'))
    invertedIndex = dict()
    
    f = open(path, 'r')
    output = open(outname, 'w')
    for line in f:
        if(("<num>" in line) and ("</num>" in line)):
            num = line.split("<num>")[1].split("</num>")[0]
            for nextline in f:
                title = nextline.split("<title>")[1].split("</title>")[0].lower()
                newTitle = preProcess(title)
                break
            # print(num, newTitle)
            output.write(str(num)+","+str(newTitle)+"\n")
    f.close()
    return

#=========================================================
# Term Frequency Variants
def logTermFrequency(x):
    return ((x>0).astype('int') * (1 + ma.log10(x).filled(0)))

def augmentedTermFrequency(x):
    y = (x>0).astype('int')
    return (0.5 + ((0.5*x)/(1.0*np.max(x))))*y

def logaveTermFrequency(x):
    return logTermFrequency(x)/(1+x[x>0].mean())

#==========================================================
# Document Frequency Variants
def noDocumentFrequency(x):
    return x

def idfDocumentFrequency(x, N, dft):
    return x*np.log10(N/dft)

def probidfDocumentFrequency(x, N, dft):
    return x*np.maximum(0, (ma.log10((N-dft)/dft).filled(0)))

#==========================================================
# Normalization
def cosineNormalization(x):
    return x/np.sqrt(np.sum(x*x))    

#==========================================================
def IDF(x, N):
    return np.log10(N/x)

#==========================================================
# Different Schemes
def lncVector(x):  
    l = logTermFrequency(x)
    n = noDocumentFrequency(l)
    c = cosineNormalization(n)
    return c

def ltcVector(x, N, dtf):
    l = logTermFrequency(x)
    t = idfDocumentFrequency(l, N, dtf)
    c = cosineNormalization(t)
    return c
    
def LncVector(x):
    L = logaveTermFrequency(x)
    n = noDocumentFrequency(L)
    c = cosineNormalization(n)
    return c

def LpcVector(x, N, dft):
    L = logaveTermFrequency(x)
    p = probidfDocumentFrequency(L, N, dft)
    c = cosineNormalization(p)
    return c

def ancVector(x):
    a = augmentedTermFrequency(x)
    n = noDocumentFrequency(a)
    c = cosineNormalization(n)
    return c

def apcVector(x, N, dft):
    a = augmentedTermFrequency(x)
    p = probidfDocumentFrequency(a, N, dft)
    c = cosineNormalization(p)
    return c

#==================================================================================
# cosine similarity
def cosineSimilarity(x, y):
    return np.sum(x*y, axis=1)/(np.linalg.norm(x)*np.linalg.norm(y, axis=1))

#==================================================================================
# Obtaning Query vectors for 3 different schemes
def QueryVectors(queryPath, term2i, N, DF_t, V):
    print("Obtaning Query vectors...")
    f = open(queryPath, 'r')
    Qltc = np.zeros(len(V))
    QLpc = np.zeros(len(V))
    Qapc = np.zeros(len(V))
    
    Vocab = set(V)
    for line in f:
        temp  = line.split(',')
        num   = temp[0]
        title = temp[1][:-1]
        queryTerms = title.split(' ')
        q = np.zeros(len(V))
        
        # building query vector
        for term in queryTerms:            
            if(term in Vocab):
                q[term2i[term]]+=1
                
        # if not(np.sum(q)==len(queryTerms)):
        #     print("Query: ", num, title, np.sum(q))
        #     for term in queryTerms:
        #         if(term not in Vocab):
        #             print(term, " :not in Vocab")

        Qltc = np.vstack((Qltc, ltcVector(q, N, DF_t)))
        QLpc = np.vstack((QLpc, LpcVector(q, N, DF_t)))
        Qapc = np.vstack((Qapc, apcVector(q, N, DF_t)))
        
    Qltc = Qltc[1:]
    QLpc = QLpc[1:]
    Qapc = Qapc[1:]
    return Qltc, QLpc, Qapc

#==================================================================================
# function to find similarity between query and documents with 3 schemes
def Similarity(path, Qltc, QLpc, Qapc, term2i, VocabSize):
    D = 0
    Q = Qltc.shape[0]
    for folder in glob.glob(path+"/*"):
        for file in glob.glob(folder+"/*"):
            D+=1
    print("Total Documents:",D, " Total Queries:",Q)
    print("Processing...")
    Scoreslnc = np.zeros((D, Q))
    ScoresLnc = np.zeros((D, Q))
    Scoresanc = np.zeros((D, Q))
    i = -1
    num2doc = dict()
    for folder in glob.glob(path+"/*"):
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
            

            file       = file.split('/')[-1]
            i         += 1
            num2doc[i] = file
            counter    = collections.Counter(terms)
            f.close()
            
            d_tf       = np.zeros(VocabSize)
            
            for term in terms:
                d_tf[term2i[term]] += 1
            # now we have Document-Term-Frequency-Raw-vector

            dlnc = lncVector(d_tf)
            dLnc = LncVector(d_tf)
            danc = ancVector(d_tf)

            sim = cosineSimilarity(dlnc, Qltc)
            Scoreslnc[i] = sim

            sim = cosineSimilarity(dLnc, QLpc)
            ScoresLnc[i] = sim

            sim = cosineSimilarity(danc, Qapc)
            Scoresanc[i] = sim
            if(i%20==0):
                print("Processing Completed till docs: ", i, " /", D)
    return Scoreslnc, ScoresLnc, Scoresanc, num2doc

#==================================================================================
# function to output the CSV files
def RankedList(out_name, Scores, Q, num2doc, K):
    print("Generating Results: "+str(out_name))
    f = open(out_name, 'w')
    f.write("Query_ID,Document_ID,Rank\n")
    M = Scores
    q = 126
    for i in range(Q):
        top50 = np.argsort(M[:,i])[::-1][:K]
        r = 1
        for i in top50:
            doc = num2doc[i]
            f.write(str(q)+","+str(doc)+","+str(r)+"\n")
            r+=1
        q+=1
    f.close()


#============================== MAIN FUNCTION ====================================
def main():
    if len(sys.argv)<4:
        print("Invalid arguments!!")
        return
    
    # N     : Number of documents
    # V     : Vocabulary
    # DF_t  : Document Frequency for each term, DF(t)

    queryIn    = sys.argv[1]
    dataPath   = sys.argv[2]            # dataPath   = "./Data/en_BDNews24"
    indexPath  = sys.argv[3]            # indexPath  = "./model_queries_9.pth"

    queryOut   = '/'.join(dataPath.split('/')[:-2]) + "/queries_9.txt"
    ResultA    = '/'.join(dataPath.split('/')[:-2]) + "/Assignment2_9_ranked_list_A.csv"
    ResultB    = '/'.join(dataPath.split('/')[:-2]) + "/Assignment2_9_ranked_list_B.csv"
    ResultC    = '/'.join(dataPath.split('/')[:-2]) + "/Assignment2_9_ranked_list_C.csv"

    invertedIndex = load_invertedIndex(indexPath)
    V             = get_Vocabulary(invertedIndex)
    term2i        = get_t2i(V)
    N             = get_NumberOfDocs(dataPath)
    DF_t          = get_DFt(N, V, invertedIndex)

    # Preprocess queries
    PreProcessQueries(queryIn, queryOut)

    Qltc, QLpc, Qapc = QueryVectors(queryOut, term2i, N, DF_t, V)
    Slnc, SLnc, Sanc, num2doc = Similarity(dataPath, Qltc, QLpc, Qapc, term2i, len(V))

    Q = Qltc.shape[0]
    RankedList(ResultA, Slnc, Q, num2doc, 50)
    RankedList(ResultB, SLnc, Q, num2doc, 50)
    RankedList(ResultC, Sanc, Q, num2doc, 50)

if __name__== "__main__":
    main()