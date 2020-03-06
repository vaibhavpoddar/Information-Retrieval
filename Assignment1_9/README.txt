DataSet Link: 	https://drive.google.com/drive/folders/1mkMBQ9vjmsiGc4nfKNQl9wvIis75PpMW?usp=sharing


==============================================================================
Version of Python used:	3.6

Libraries used:	
	*) glob
	*) os
	*) sys
	*) nltk
	*) pickle

==============================================================================


For Running Task 1A (Building Index) execute the following command:

	$ python Assignment1_9_indexer.py ./Data/en_BDNews24

Output: "model_queries_9.pth"


==============================================================================


For Running Task 1B (Query Preprocessing) execute the following command:

	$ python Assignment1_9_parser.py ./Data/raw_query.txt 

Output:	"queries_9.txt"


==============================================================================


For Running Task 1C (Boolean Retrieval) execute the following command:

	$ python Assignment1_9_bool.py ./model_queries_9.pth ./queries_9.txt

Output:	"Assignment1_9_results.txt"


==============================================================================
