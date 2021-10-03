import pandas as pd
import numpy as np
import copy
import sys

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

if len(sys.argv)<2:
    print("Invalid arguments!!")
    exit()
    

file_golden   = sys.argv[1]
filename      = sys.argv[2]
outFile       = "Assignment2_9_metrics_"+filename.split('.')[-2][-1]+".txt"
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

# NDCG@10, NDCG@20
dcg_Cat10   = DCG(C_score,10, 20)
dcg_Cat20   = DCG(C_score,20, 20)
dcg_ideal_Cat10 = DCG(ideal_score,10,20)
dcg_ideal_Cat20 = DCG(ideal_score,20,20)

ndcg_Cat10 = NDCG(dcg_ideal_Cat10, dcg_Cat10)
ndcg_Cat20 = NDCG(dcg_ideal_Cat20, dcg_Cat20)

# average NDCG@10, average NDCG@20 
averNDCG_Cat10 = avg_NDCG(ndcg_Cat10)
averNDCG_Cat20 = avg_NDCG(ndcg_Cat20)
averNDCG_Cat10 = round(averNDCG_Cat10, 3)
averNDCG_Cat20 = round(averNDCG_Cat20, 3)

# Precision@10, Precision@20
Prec_C        = Prec(C_score, 20)
avgPrec_Cat10 = avgPrec(Prec_C, 10)
avgPrec_Cat20 = avgPrec(Prec_C, 20)

mAP_Cat10 = mAP(avgPrec_Cat10)
mAP_Cat20 = mAP(avgPrec_Cat20)
mAP_Cat10 = round(mAP_Cat10, 3)
mAP_Cat20 = round(mAP_Cat20, 3)

f = open(outFile,'w')
f.write(str('queryID\tAvgPrecision@10\tAvgPrecision@20\tNDCG@10\t\tNDCG@20\n'))

for k in avgPrec_Cat10:
    a = str(avgPrec_Cat10[k])
    b = str(avgPrec_Cat20[k])
    c = str(ndcg_Cat10[k])
    d = str(ndcg_Cat20[k])
    if(a=="0" or a=="0.0"):
        a = "0.000"
    if(b=="0" or b=="0.0"):
        b = "0.000"
    if(c=="0" or c=="0.0"):
        c = "0.000"
    if(d=="0" or d=="0.0"):
        d = "0.000"
        
    f.write(k + '\t\t' + a + '\t\t\t' + b + '\t\t\t' + c + '\t\t' + d + '\n')

f.write(str('\n'))
f.write("mAP@10: " + str(mAP_Cat10) + "\n")
f.write("mAP@20: " + str(mAP_Cat20) + "\n")
f.write("averNDCG@10: " + str(averNDCG_Cat10) + "\n")
f.write("averNDCG@20: " + str(averNDCG_Cat20) + "\n")
f.close()