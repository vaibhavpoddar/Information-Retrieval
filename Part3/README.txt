========================================================================
Python Version:    3.6
Libraries used:    glob, os, sys, nltk, pickle, pandas, numpy, collections, string, copy, xlrd

    Intall xlrd library using:
        pip install xlrd

========================================================================
Current Directory Structure:
----------------------------
    *) Data Folder contains only two file, you have to add "en_BDNews24" folder inside it.
    *) Description of EXTRAS folder is given below.

    +── Assignment1_9_indexer.py
    ├── Assignment3_9_rocchio.py
    ├── Assignment3_9_important_words.py
    |
    ├── Assignment2_9_ranked_list_A.csv
    ├── Assignment3_9_rocchio_RF_metrics.csv
    ├── Assignment3_9_rocchio_PsRF_metrics.csv
    ├── Assignment3_9_important_words.csv
    └── README.txt

    +── Data
    │   ├── rankedRelevantDocList.xlsx
    │   ├── raw_query.txt

    +── EXTRAS
    │   ├── Assignment3_9_rocchio_PsRF_metrics_0.csv
    │   ├── Assignment3_9_rocchio_PsRF_metrics_1.csv
    │   ├── Assignment3_9_rocchio_PsRF_metrics_2.csv
    │   ├── Assignment3_9_rocchio_RF_metrics_0.csv
    │   ├── Assignment3_9_rocchio_RF_metrics_1.csv
    │   ├── Assignment3_9_rocchio_RF_metrics_2.csv
    │   ├── Result_psrf_0.csv
    │   ├── Result_psrf_1.csv
    │   ├── Result_psrf_2.csv
    │   ├── Result_rf_0.csv
    │   ├── Result_rf_1.csv
    │   └── Result_rf_2.csv


    EXTRAS folder contains:
    =======================
        +---------------------------------------------------------------------------------------------------------------+
        | Sr.No | FileName                                 |             Description                                    |
        |-------|------------------------------------------|------------------------------------------------------------|
        | (1)   | Result_rf_0.csv                          | Ranked list by RF scheme for Case0      (for each query)   |
        | (2)   | Result_rf_1.csv                          | Ranked list by RF scheme for Case1      (for each query)   |
        | (3)   | Result_rf_2.csv                          | Ranked list by RF scheme for Case2      (for each query)   |
        | (4)   | Result_psrf_0.csv                        | Ranked list by PsRF scheme for Case0    (for each query)   |
        | (5)   | Result_psrf_0.csv                        | Ranked list by PsRF scheme for Case1    (for each query)   |
        | (6)   | Result_psrf_0.csv                        | Ranked list by PsRF scheme for Case2    (for each query)   |
        |       |                                          |                                                            |
        | (7)   | Assignment3_9_rocchio_RF_metrics_0.csv   | avgPrecision@20, NDCG@20 for Case0 (RF) (for each query)   |
        | (8)   | Assignment3_9_rocchio_RF_metrics_1.csv   | avgPrecision@20, NDCG@20 for Case1 (RF) (for each query)   |
        | (9)   | Assignment3_9_rocchio_RF_metrics_2.csv   | avgPrecision@20, NDCG@20 for Case2 (RF) (for each query)   |
        | (10)  | Assignment3_9_rocchio_PsRF_metrics_0.csv | avgPrecision@20, NDCG@20 for Case0 (PsRF) (for each query) |
        | (11)  | Assignment3_9_rocchio_PsRF_metrics_1.csv | avgPrecision@20, NDCG@20 for Case1 (PsRF) (for each query) |
        | (12)  | Assignment3_9_rocchio_PsRF_metrics_2.csv | avgPrecision@20, NDCG@20 for Case2 (PsRF) (for each query) |
        +---------------------------------------------------------------------------------------------------------------+
        These files are present in EXTRAS folder
            Case0:  alpha=1,   beta=1,   gamma=0.5
            Case1:  alpha=0.5, beta=0.5, gamma=0.5
            Case2:  alpha=1,   beta=0.5, gamma=0

=====================================================================
                                Part (A)
=====================================================================

(Ignore step#1 if "model_queries_9.pth" file is already present)
1) Before Executing this part run this, to generate indexer model:
   ---------------------------------------------------------------
        $ python3 Assignment1_9_indexer.py ./Data/en_BDNews24

        Output generated:
        -----------------
            model_queries_9.pth
        Note:
        -----
            This will take ~6 minutes to produce output.

2) Now, Execute this command:
   ------------------------
        $ python3 Assignment3_9_rocchio.py ./Data/en_BDNews24 ./model_queries_9.pth ./Data/rankedRelevantDocList.xlsx Assignment2_9_ranked_list_A.csv ./Data/raw_query.txt

    Output generated:
    -----------------
        (*) Assignment3_9_rocchio_RF_metrics.csv
        (*) Assignment3_9_rocchio_PsRF_metrics.csv    
        
    Note:
    -----
        Time to process each document for all possible combinations ~ 0.3 sec
        Total estimated time is ~ 0.3*(90000) sec ==> 7.5 hrs

=====================================================================
                                Part (B)
=====================================================================
1) Execute this command:
   ---------------------
    $ python3 Assignment3_9_important_words.py ./Data/en_BDNews24 ./model_queries_9.pth Assignment2_9_ranked_list_A.csv

    Output generated:
    -----------------
        (*) Assignment3_9_important_words.csv

    Note:
    -----
        Total estimated time is ~ 1 min