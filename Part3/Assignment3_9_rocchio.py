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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer    = WordNetLemmatizer()
StopWords     = set(stopwords.words('english'))
punctuations  = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''           # punctuation marks 


def get_score(D, K):
    q2score = dict()
    for k in D:
        q2score[k] = np.zeros(K)
        i = -1
        for v in D[k]:
            i += 1
            q2score[k][i] = v[1]
    return q2score

def DCG(D, atK, dim):
    deno = np.ones(dim)
    for i in range(1, dim):
        deno[i] = np.log2(i+1)

    dcg = dict()
    for k in D:
        x = D[k]
        x = x/deno
        dcg[k]=np.sum(x[:atK])
    return dcg

def NDCG(ideal, ranking_system):
    ndcg = dict()
    for k in ideal:
        if ideal[k]==0:
            ndcg[k]=0
        else:
            ndcg[k]=ranking_system[k]/ideal[k]
            ndcg[k]=round(ndcg[k],3)
    return ndcg

def avg_NDCG(ndcg):
    avg_ndcg = 0
    for k in ndcg:
        avg_ndcg += ndcg[k]
    avg_ndcg /= len(ndcg)
    return avg_ndcg

def Prec(D, dim):
    prec = dict()
    for k in D:
        prec[k] = np.zeros(dim)
        relevant = 0
        non_relevant = 0
        for v in range(dim):
            if (D[k][v]==1 or D[k][v]==2):
                relevant +=1
                prec[k][v] = relevant/(v+1)
            else:
                non_relevant += 1
                prec[k][v] = relevant/(v+1)
    return prec

def avgPrec(D, atK):
    avg_prec = dict()
    for k in D:
        avg_prec[k] = np.sum(D[k][:atK])
        avg_prec[k] /= atK
        avg_prec[k] = round(avg_prec[k],3)
    return avg_prec

def mAP(D):
    mean_AP = 0
    for k in D:
        mean_AP += D[k]
    mean_AP /= len(D)
    return mean_AP

def OutputMetric(goldPath, csvpath, param):
    file_golden   = goldPath
    filename      = csvpath
    data_C        = pd.read_csv(filename)
    data_gold_std = pd.read_excel(file_golden)

    dict_C = dict()
    for i in range(len(data_C)): 
        s = str(data_C.iloc[i, 0]) + "," + str(data_C.iloc[i, 1])
        dict_C[s]= [data_C.iloc[i, 2],-1]

    dict_std = dict()
    for i in range(len(data_gold_std)): 
        s = str(data_gold_std.iloc[i, 0]) + "," + str(data_gold_std.iloc[i, 1])
        dict_std[s] = data_gold_std.iloc[i, 2]

    for k in dict_C:
        if k in dict_std:
            dict_C[k][1] = dict_std[k]
        else:
            dict_C[k][1] = 0

    query = dict()
    for k in dict_C:
        query[k.split(',')[0]]=[]
    for k in dict_C:
        query[k.split(',')[0]].append(dict_C[k])
    for k in dict_C:
        query[k.split(',')[0]] = query[k.split(',')[0]][:20]

    query_ideal = copy.deepcopy(query)
    for k in query_ideal:
        query_ideal[k] = sorted(query_ideal[k], key = lambda x: x[1], reverse=True)

    C_score     = get_score(query,20)
    ideal_score = get_score(query_ideal,20)

    # NDCG@20
    dcg_Cat20   = DCG(C_score,20, 20)
    dcg_ideal_Cat20 = DCG(ideal_score,20,20)

    ndcg_Cat20 = NDCG(dcg_ideal_Cat20, dcg_Cat20)

    # average NDCG@20 
    averNDCG_Cat20 = avg_NDCG(ndcg_Cat20)
    averNDCG_Cat20 = round(averNDCG_Cat20, 3)

    # Precision@20
    Prec_C        = Prec(C_score, 20)
    avgPrec_Cat20 = avgPrec(Prec_C, 20)

    mAP_Cat20 = mAP(avgPrec_Cat20)
    mAP_Cat20 = round(mAP_Cat20, 3)

    (Aplha, Beta, Gamma) = param
    string_res = str(Aplha) + ',' + str(Beta) +','+ str(Gamma) +','+ str(mAP_Cat20) +','+ str(averNDCG_Cat20) + '\n'
    return string_res

def getResults(goldPath, outputfiles, files, parameters):
    rf0 = OutputMetric(goldPath, files[0], parameters[0])
    rf1 = OutputMetric(goldPath, files[1], parameters[1])
    rf2 = OutputMetric(goldPath, files[2], parameters[2])
    
    psrf0 = OutputMetric(goldPath, files[3], parameters[0])
    psrf1 = OutputMetric(goldPath, files[4], parameters[1])
    psrf2 = OutputMetric(goldPath, files[5], parameters[2])

    f = open(outputfiles[0], 'w')
    f.write("Aplha,Beta,Gamma,mAP@20,avgNDCG@20\n")
    f.write(rf0)
    f.write(rf1)
    f.write(rf2)
    f.close()

    f = open(outputfiles[1], 'w')    
    f.write("Aplha,Beta,Gamma,mAP@20,avgNDCG@20\n")
    f.write(psrf0)
    f.write(psrf1)
    f.write(psrf2)
    f.close()

    for file in files:
        os.remove(file)

# ========================================================
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

# Function to get total number of Documents in given Dataset
def get_NumberOfDocs(Data_path):
    i = 0
    path = Data_path                                       # path = "./Data/en_BDNews24/*"
    for folder in glob.glob(path+"/*"):
        for file in glob.glob(folder+"/*"):
            i+=1
    return i

# Auxiliary Function to map term to index
def get_t2i(V):
    i = 0
    term_2_i = dict()    
    for term in V:
        term_2_i[term] = i
        i+=1
    return term_2_i

# Function to get Document Frequency vector
def get_DFt(N, V, invertedIndex):
    i = 0
    DF_t  = np.zeros(len(V))
    for term in V:
        DF_t[i] = len(invertedIndex[term])
        i+=1
        
    return DF_t

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

def noDocumentFrequency(x):
    return x

def idfDocumentFrequency(x, N, dft):
    return x*np.log10(N/dft)

# Normalization
def cosineNormalization(x):
    return x/np.sqrt(np.sum(x*x))    

def IDF(x, N):
    return np.log10(N/x)

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

# cosine similarity
def cosineSimilarity(x, y):
    return np.sum(x*y, axis=1)/(np.linalg.norm(x)*np.linalg.norm(y, axis=1))

#==================================================================================
# Obtaning Query vectors for 3 different schemes
def QueryVectors(queryPath, term2i, N, DF_t, V):
    print("Obtaning Query vectors...")
    f = open(queryPath, 'r')
    Qltc = np.zeros(len(V))
    
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

        Qltc = np.vstack((Qltc, ltcVector(q, N, DF_t)))
        
    Qltc = Qltc[1:]
    return Qltc

# function to find similarity between query and documents with 3 schemes
def Similarity(path, Q_rf0, Q_psrf0, Q_rf1, Q_psrf1, Q_rf2, Q_psrf2, term2i, VocabSize):
    D = 0
    Q = Q_rf0.shape[0]
    for folder in glob.glob(path+"/*"):
        for file in glob.glob(folder+"/*"):
            D+=1
    print("Total Documents:",D, " Total Queries:",Q)
    print("Processing...")
    Scoresrf0 = np.zeros((D, Q))
    Scoresrf1 = np.zeros((D, Q))
    Scoresrf2 = np.zeros((D, Q))
    
    Scorespsrf0 = np.zeros((D, Q))
    Scorespsrf1 = np.zeros((D, Q))
    Scorespsrf2 = np.zeros((D, Q))
    
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

            sim = cosineSimilarity(dlnc, Q_rf0)
            Scoresrf0[i] = sim
            sim = cosineSimilarity(dlnc, Q_rf1)
            Scoresrf1[i] = sim
            sim = cosineSimilarity(dlnc, Q_rf2)
            Scoresrf2[i] = sim
            
            sim = cosineSimilarity(dlnc, Q_psrf0)
            Scorespsrf0[i] = sim                       
            sim = cosineSimilarity(dlnc, Q_psrf1)
            Scorespsrf1[i] = sim
            sim = cosineSimilarity(dlnc, Q_psrf2)
            Scorespsrf2[i] = sim

            if(i%20==0):
                print("Processing Completed till docs: ", i, " /", D)
    return Scoresrf0, Scorespsrf0, Scoresrf1, Scorespsrf1, Scoresrf2, Scorespsrf2, num2doc

# function to output the CSV files
def RankedList(out_name, Scores, Q, num2doc, K):
    # print("Generating Results: "+str(out_name))
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

def getFullPath(docName):
    search_file = docName
    for (root,dirs,files) in os.walk('./Data/en_BDNews24', topdown=True):
        if search_file in files:
            # print(os.path.join(root, search_file))
            return os.path.join(root, search_file)

def MakeNegativeZero(v):
    v[v<0]=0
    return v

def QueryWiseGoldStandard(goldPath):
    q2d = dict()
    data_gold_std = pd.read_excel(goldPath)
    for i in range(len(data_gold_std)): 
        q = int(data_gold_std.iloc[i, 0])
        d = str(data_gold_std.iloc[i, 1])
        v = int(data_gold_std.iloc[i, 2])
        if(v==2):
            if(int(q) not in q2d):
                q2d[int(q)]=[]
            q2d[int(q)].append(d)
    for i in range(126, 176):
        if(i not in q2d):
            q2d[i]=[]
    return q2d

# function to find Modified query using Relevance Feedback and Pseudo-Relevance Feedback
def ModifiedQueryVectors(queryPath, term2i, N, DF_t, V, rankedListPath, q2Gold, param):
    print("Obtaning Modified Query vectors...")
    f = open(queryPath, 'r')
    ranked_List = pd.read_csv(rankedListPath)

    Qrf   = np.zeros(len(V))
    Qpsrf = np.zeros(len(V))
    alpha = param[0]
    beta  = param[1]
    gamma = param[2]
    print(">>>Parameters: ", alpha, beta, gamma)
    Vocab = set(V)
    i = 0                       # query number (starting from 0 i.e. 126))
    for line in f:              # for q in queries
        print("Working for query number: ", 126+i)
        temp  = line.split(',')
        num   = temp[0]
        title = temp[1][:-1]
        queryTerms = title.split(' ')
        q = np.zeros(len(V))
        
        # building query vector
        for term in queryTerms:            
            if(term in Vocab):
                q[term2i[term]]+=1
        
        # vector representation of initial Query
        q0   = ltcVector(q, N, DF_t)
        # Qltc = np.vstack((Qltc, q0))
        
        # get top20 docs for current query
        ranked_List.iloc[i*50: i*50 + 20,:]
        data  = ranked_List.iloc[i*50: i*50 + 20,:]
        TOP20 = list(data["Document_ID"])
        
        # Implement Scheme 1 for expanding query (Relevance Feedback)
        # Implement Scheme 2 for expanding query (Pseudo Relevance Feedback)        
        r_count    = 0
        nonr_count = 0
        psr_count  = 0
        r_sum      = np.zeros(len(V))
        nonr_sum   = np.zeros(len(V))
        psr_sum    = np.zeros(len(V))
        for docName in TOP20:
            vr = vectorRepresentation(docName, term2i, len(V))
            if(psr_count!=10):
                psr_sum   += vr
                psr_count += 1
            if(docName in q2Gold[i+126]):
                r_sum    += vr
                r_count  += 1
                # print("\t", temp_counter, ":", np.sum(psr_sum), np.sum(r_sum), np.sum(nonr_sum), "GOLD")
            else:
                nonr_sum   += vr
                nonr_count += 1
        
        if(r_count!=0):
            r = r_sum/r_count
        else: 
            r = np.zeros(len(V))

        if(nonr_count!=0):
            nonr = nonr_sum/nonr_count
        else:
            nonr = np.zeros(len(V))

        psr  = psr_sum/psr_count 

        # Modified Query by Relevance Feedback
        qmrf = alpha*q0 + beta*r - gamma*nonr
        qmrf = MakeNegativeZero(qmrf)
        qmrf = cosineNormalization(qmrf)

        # Modified Query by Pseudo-Relevance Feedback
        qmpsrf = alpha*q0 + beta*psr
        qmpsrf = MakeNegativeZero(qmpsrf)
        qmpsrf = cosineNormalization(qmpsrf)

        # print("At query number:", 126+i, "Sum of qmpsrf", np.sum(qmpsrf))
        
        Qrf   = np.vstack((Qrf,   qmrf))
        Qpsrf = np.vstack((Qpsrf, qmpsrf))

        i+=1
    Qrf   = Qrf[1:]
    Qpsrf = Qpsrf[1:]    
    return Qrf, Qpsrf

# def OutputMetric(goldPath, csvpath, outPath, param):
#     file_golden   = goldPath
#     filename      = csvpath
#     outFile       = outPath
#     data_C        = pd.read_csv(filename)
#     data_gold_std = pd.read_excel(file_golden)

#     dict_C = dict()
#     for i in range(len(data_C)): 
#         s = str(data_C.iloc[i, 0]) + "," + str(data_C.iloc[i, 1])
#         dict_C[s]= [data_C.iloc[i, 2],-1]

#     dict_std = dict()
#     for i in range(len(data_gold_std)): 
#         s = str(data_gold_std.iloc[i, 0]) + "," + str(data_gold_std.iloc[i, 1])
#         dict_std[s] = data_gold_std.iloc[i, 2]

#     for k in dict_C:
#         if k in dict_std:
#             dict_C[k][1] = dict_std[k]
#         else:
#             dict_C[k][1] = 0

#     query = dict()
#     for k in dict_C:
#         query[k.split(',')[0]]=[]
#     for k in dict_C:
#         query[k.split(',')[0]].append(dict_C[k])
#     for k in dict_C:
#         query[k.split(',')[0]] = query[k.split(',')[0]][:20]

#     query_ideal = copy.deepcopy(query)
#     for k in query_ideal:
#         query_ideal[k] = sorted(query_ideal[k], key = lambda x: x[1], reverse=True)

#     C_score     = get_score(query,20)
#     ideal_score = get_score(query_ideal,20)

#     # NDCG@20

#     dcg_Cat20   = DCG(C_score,20, 20)
#     dcg_ideal_Cat20 = DCG(ideal_score,20,20)

#     ndcg_Cat20 = NDCG(dcg_ideal_Cat20, dcg_Cat20)

#     # average NDCG@20 
#     averNDCG_Cat20 = avg_NDCG(ndcg_Cat20)
#     averNDCG_Cat20 = round(averNDCG_Cat20, 3)

#     # Precision@20
#     Prec_C        = Prec(C_score, 20)
#     avgPrec_Cat20 = avgPrec(Prec_C, 20)

#     mAP_Cat20 = mAP(avgPrec_Cat20)
#     mAP_Cat20 = round(mAP_Cat20, 3)

#     f = open(outFile,'w')
#     f.write(str('queryID\tAvgPrecision@20\tNDCG@20\n'))

#     for k in avgPrec_Cat20:
#         b = str(avgPrec_Cat20[k])
#         d = str(ndcg_Cat20[k])
#         if(b=="0" or b=="0.0"):
#             b = "0.000"
#         if(d=="0" or d=="0.0"):
#             d = "0.000"
            
#         f.write(k + '\t\t' + b + '\t\t\t'+ d + '\n')

#     (Aplha, Beta, Gamma) = param
#     f.write(str('\n'))
#     f.write("Aplha, Beta, Gamma, mAP@20, avgNDCG@20\n")
#     f.write("{},     {},    {},   {},    {}".format(Aplha, Beta, Gamma, mAP_Cat20, averNDCG_Cat20))
#     f.close()

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
    goldPath   = sys.argv[3]            # goldPath   = "./Data/rankedRelevantDocList.xlsx"
    csvA       = sys.argv[4]            # csvA       = "Assignment2_9_ranked_list_A.csv"
    queryIn    = sys.argv[5]            # queryIn    = "./Data/raw_query.txt"

    queryOut   = '/'.join(dataPath.split('/')[:-2]) + "/queries_9.txt"
    MetricRF   = '/'.join(dataPath.split('/')[:-2]) + "/Assignment3_9_rocchio_RF_metrics.csv"
    MetricPsRF = '/'.join(dataPath.split('/')[:-2]) + "/Assignment3_9_rocchio_PsRF_metrics.csv"

    invertedIndex = load_invertedIndex(indexPath)
    V             = get_Vocabulary(invertedIndex)
    term2i        = get_t2i(V)
    N             = get_NumberOfDocs(dataPath)
    DF_t          = get_DFt(N, V, invertedIndex)

    # Preprocess queries
    PreProcessQueries(queryIn, queryOut)

    Qltc= QueryVectors(queryOut, term2i, N, DF_t, V)

    # Relevance Feedback and Pseudo Relevance Feedback
    parameters = [(1, 1, 0.5), (0.5, 0.5, 0.5), (1, 0.5, 0)]
    q2Gold = QueryWiseGoldStandard(goldPath)

    Q_rf0, Q_psrf0 = ModifiedQueryVectors(queryOut, term2i, N, DF_t, V, csvA, q2Gold, parameters[0])
    Q_rf1, Q_psrf1 = ModifiedQueryVectors(queryOut, term2i, N, DF_t, V, csvA, q2Gold, parameters[1])
    Q_rf2, Q_psrf2 = ModifiedQueryVectors(queryOut, term2i, N, DF_t, V, csvA, q2Gold, parameters[2])
        
    S_rf0, S_psrf0, S_rf1, S_psrf1, S_rf2, S_psrf2, num2doc = Similarity(dataPath, Q_rf0, Q_psrf0, Q_rf1, Q_psrf1, Q_rf2, Q_psrf2, term2i, len(V))

    RankedList("Result_rf_0.csv",   S_rf0,   Q_rf0.shape[0],   num2doc, 50)
    RankedList("Result_rf_1.csv",   S_rf1,   Q_rf1.shape[0],   num2doc, 50)
    RankedList("Result_rf_2.csv",   S_rf2,   Q_rf2.shape[0],   num2doc, 50)
    
    RankedList("Result_psrf_0.csv", S_psrf0, Q_psrf0.shape[0], num2doc, 50)
    RankedList("Result_psrf_1.csv", S_psrf1, Q_psrf1.shape[0], num2doc, 50)
    RankedList("Result_psrf_2.csv", S_psrf2, Q_psrf2.shape[0], num2doc, 50)

    files = ["Result_rf_0.csv", "Result_rf_1.csv", "Result_rf_2.csv", "Result_psrf_0.csv", "Result_psrf_1.csv", "Result_psrf_2.csv"]
    outputfiles = ["Assignment3_9_rocchio_RF_metrics.csv", "Assignment3_9_rocchio_PsRF_metrics.csv"]
    getResults(goldPath, outputfiles, files, parameters)
    
if __name__== "__main__":
    main()