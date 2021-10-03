========================================================================
Version of Python used:	3.6

Libraries used:	
	*) glob
	*) os
	*) sys
	*) nltk
	*) pickle
	*) pandas, numpy, collections, string, copy

=====================================================================
Intall xlrd library using, for using Part B:
	
	pip install xlrd


========================================================================


Data Folder contains only two file, you have to add "en_BDNews24" folder inside it.


|---Data
	|
	|---------- raw_query.txt
	|---------- rankedRelevantDocList.xlsx


=====================================================================
								Part (A)
=====================================================================
Before Executing this part run this, to generate indexer model:

	python Assignment1_9_indexer.py ./Data/en_BDNews24

Execute this command:

	 python Assignment2_9_ranker.py ./Data/raw_query.txt ./Data/en_BDNews24 ./model_queries_9.pth 


Time to process each document is ~ 0.2 sec
Total estimated time is: 5 hrs

FInal results are given in the folder.
	"Assignment2_9_ranked_list_A.csv"
	"Assignment2_9_ranked_list_B.csv"
	"Assignment2_9_ranked_list_C.csv"

=====================================================================
								Part (B)
=====================================================================
Intall xlrd library using, for using this:
	
	pip install xlrd


Execute this command:

	python Assignment2_9_evaluator.py ./Data/rankedRelevantDocList.xlsx ./Assignment2_9_ranked_list_A.csv 

	python Assignment2_9_evaluator.py ./Data/rankedRelevantDocList.xlsx ./Assignment2_9_ranked_list_B.csv 

	python Assignment2_9_evaluator.py ./Data/rankedRelevantDocList.xlsx ./Assignment2_9_ranked_list_C.csv 
