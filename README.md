# MWDB Project Phase 2

Installation Instructions:

1. Run the following command to setup a virtual environment
	$ python -m venv demo
	$ source demo/bin/activate
	
2. Run the following command to install all the dependencies in the virtual environment
	$ pip install -r requirements.txt
			
3. To run the program for Task 0A, execute the following command:
	$ python task0a.py <Directory> <Resolution> <Window Length> <Shift Length>
	Directory - The path to the dataset
	Resolution - An integer value
	Window Length - An integer value
	Shift Length - An integer value
	For example: $ python task0a.py ./test 3 3 3

4. To run the program for Task 0B, execute the following command:
	$ python task0b.py <Directory>
	Directory - The path to the dataset
	For example: $ python task0b.py ./test

5. To run the program for Task 1, execute the following command:
	$ python task1.py <Directory> <Vector Model> <User Option> <k>
	Directory - The path to the dataset
	Vector Model - ‘A string value (‘tf’ or ‘tfidf’)
	User Option - An integer value (Between 1 and 4)
	k - An integer value
	For example: $ python task1.py ./test tfidf 3 4

6. To run the program for Task 2, execute the following command:
	$ python task2.py <Directory> <Gesture File> <Vector Model> <User Option> <k>
	Directory - The path to the dataset
	Gesture File - An integer value
	Vector Model - ‘A string value (‘tf’ or ‘tfidf’)
	User Option - An integer value (Between 1 and 7)
	k - An integer value
	For example: $ python task1.py ./test 10 tfidf 3 4

7. To run the program for Task 3, execute the following command:
	$ python task3.py <Directory> <Vector Model> <User Option> <p> <k>
	Directory - The path to the dataset
	Vector Model - ‘A string value (‘tf’ or ‘tfidf’)
	User Option - An integer value (Between 1 and 7)
	p - An integer value
	K - An integer value
	For example: $ python task1.py ./test tfidf 3 4 4

8. To run the program for Tasks 4a and 4b, execute the following command:
	$ python task4ab.py 
	For example: $ python task4ab.py

9. To run the program for Tasks 4c and 4d, execute the following command:
	$ python task4cd.py <User Option> <p> <Task>
    User Option - An integer value (Between 1 and 7)
	p - An integer value
	Task - A string value (‘C’ or ‘D’)
	For example: $ python task4cd.py 3 4 C

10. After completion, run the following command to deactivate the virtual environment
	$ deactivate
