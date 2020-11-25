```
System Requirements:
Operating System: Windows 10
Python Version: 3.8.0
Libraries Used: os, json, glob, sys, pandas, numpy, scipy, copy, math, ast, pickle, sklearn
Setup Instructions:
Run the following commands to create a virtual environment name demo and install all of the dependencies
		$ python -m venv demo
		$ source demo/bin/activate
		$ pip install -r requirements.txt


Installation Instructions:
To generate the tf/tfifd vectors and the similarity matrix, execute the following command:
	$ python phase3_preprocessing.py <Directory> <Resolution> <Window Length> <Shift Length>
	Directory - The path to the dataset
	Resolution - An integer value
	Window Length - An integer value
	Shift Length - An integer value
	For example: $ python phase3_preprocessing.py ./test 3 3 3
To run the program for Task 1, execute the following command:
	$ python phase3_task1.py <Directory> <k> <m> <c>
	Directory - The path to the dataset
	k = An integer value
	m = An integer value
	c = A float value between 0 and 1
	For example: $ python phase3_task1.py data 3 10 0.8
To run the program for KNN Classifier in Task 2, execute the following command:
	$ python phase3_task2_knn.py <Directory> <Vector Model> <k>
	Directory - The path to the dataset
	Vector Model - ‘A string value (‘tf’ or ‘tfidf’)
	k - An integer value
	For example: $ python phase3_task2_knn.py ./test tf 10
To run the program for Decision Tree Classifier in Task 2, execute the following command:
	$ python phase3_task2_decision_tree.py <Vector Model>
	Vector Model - ‘A string value (‘1. PCA’ or ‘2.SVD’)
	For example: $ python phase3_task2_decision_tree.py 1
To run the program for PPR Classifier Task 2, execute the following command:
	$ python phase3_task2_ppr.py <Directory> <k> <m> <c>
	Directory - The path to the dataset
	k - An integer value
	m - An integer value
	c - A float value between 0 and 1
	For example: $ python phase3_task2_ppr.py test 3 10 0.8
To run the program for Task 3 execute the following command:
	$ python phase3_task3.py <L> <k> <Space> <Vector Model>
L - An integer value
	k - An integer value
	Space - An integer value (0 for original space / 1 for reduced space)
	Vector Model - ‘A string value (‘tf’ or ‘tfidf’)
	For example: $ python phase3_task3.py 8 20 0 tf
Tasks 4 and 5 are not specifically called from command line but are executed through the GUI after task 3 depending on user option
	To run the program for Task 6, execute the following command:
	$ python phase3_task6.py```