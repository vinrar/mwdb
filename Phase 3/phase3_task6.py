import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.simpledialog as tkSimpleDialog
import os, json
from gui_phase3_task3 import locality_sensitive_hashing, preprocessing
from task5 import get_ppr2

HEIGHT = 600
WIDTH = 800
GESTURE_PATH = ''
TASK_NUMBER = None
QUERY_GESTURE = None
RELEVANT_FEEDBACK = []
IRRELEVANT_FEEDBACK = []


# L = 10
# k = 6
# model = 'tf'
# t = 10


# Function to get the directory of gesture files
def getPath():
    global GESTURE_PATH
    filename = tkFileDialog.askdirectory()
    print('File directory: ', filename)
    GESTURE_PATH = filename


# Function to get the task number
def getNum():
    global TASK_NUMBER
    n = tkSimpleDialog.askinteger("Task Number", "Enter the task number (4/5)", parent=root, minvalue=4, maxvalue=5)
    print('Task Number: ', n)
    TASK_NUMBER = n


# Function to get the query gesture
def getQuery():
    global QUERY_GESTURE
    q = tkSimpleDialog.askinteger("Query Gesture", "Enter the query gesture", parent=root)
    print('Query Gesture: ', q)
    QUERY_GESTURE = str(q)


# Function to get input for Task 3
def getTask3Input():
    pass


# Function to generate similarity results
def generateResults():
    global GESTURE_PATH, TASK_NUMBER, RELEVANT_FEEDBACK, IRRELEVANT_FEEDBACK, QUERY_GESTURE
    if TASK_NUMBER == 4:
        print("Running task 4")
    if TASK_NUMBER == 5:
        print("Running task 5")

    firstRun = True
    runAgain = True
    gesture_list = []
    while runAgain:
        # generate results for similarity
        if firstRun:
            # run for the first time without feedback

            # input for LSH
            L = 10
            k = 6
            model = 'tf'
            query_gesture = QUERY_GESTURE
            t = 10
            with open(model + '_vectors.json', 'r') as fp:
                vectors = json.load(fp)

            # preprocessing for LSH
            dims = len(list(vectors.values())[0])
            hash_tables = preprocessing(L, k, dims, vectors)
            # locality_sensitive_hashing(L, k, )
            results_to_display, gesture_list = locality_sensitive_hashing(L, k, query_gesture, t, vectors, hash_tables,
                                                                          True)

            # results_to_display = [i for i in range(1,11)]
            # resultLabel['text'] = formatResult(results_to_display)
            resultLabel['text'] = results_to_display
            firstRun = False
            RELEVANT_FEEDBACK.append(query_gesture)
        else:
            # run for the subsequent times with feedback
            ppr_results_to_display = get_ppr2(len(gesture_list), 0.85, gesture_list, RELEVANT_FEEDBACK,
                                              IRRELEVANT_FEEDBACK, gui=True)
            resultLabel['text'] = ppr_results_to_display

        # get feedback
        string_relevant_feedback = tkSimpleDialog.askstring("Relevant  Feedback",
                                                            "Enter the relevant points separated by a comma")
        string_irrelevant_feedback = tkSimpleDialog.askstring("Irrelevant  Feedback",
                                                              "Enter the irrelevant points separated by a comma")
        if string_relevant_feedback and string_relevant_feedback != '':
            relevant_feedback = string_relevant_feedback.split(',')
            if string_irrelevant_feedback and string_irrelevant_feedback != '':
                irrelevant_feedback = string_irrelevant_feedback.split(',')
                IRRELEVANT_FEEDBACK.extend(irrelevant_feedback)
            RELEVANT_FEEDBACK.extend(relevant_feedback)
            RELEVANT_FEEDBACK = list(set(RELEVANT_FEEDBACK))
            IRRELEVANT_FEEDBACK = list(set(IRRELEVANT_FEEDBACK))
            print("Relevant feedback", RELEVANT_FEEDBACK)
            print("Irrelevant feedback", IRRELEVANT_FEEDBACK)
            print("Gesture List", gesture_list)
        else:
            print("Exiting feedback loop")
            runAgain = False
            feedback = []
            break


# Function to format result
def formatResult(result):
    resultString = 'The top ten similar gestures are:\n'
    for idx, res in enumerate(result):
        if idx != len(result) - 1:
            resultString += str(res) + ', '
        else:
            resultString += str(res)
    return resultString


root = tk.Tk()
root.title("Task6-GUI")

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

titleFrame = tk.Frame(root)
titleFrame.place(relheight=0.2, relwidth=1)

titleLabel = tk.Label(titleFrame, text='MWDB Phase 3 - Task 6', font=('Arial Bold', 45))
titleLabel.place(relheight=1, relwidth=1)

buttonFrame = tk.Frame(root)
buttonFrame.place(relx=0.5, rely=0.2, relwidth=0.75, relheight=0.1, anchor='n')

directoryButton = tk.Button(buttonFrame, text="Select Directory", command=lambda: getPath())
directoryButton.place(relx=0, relheight=1, relwidth=0.25)

taskButton = tk.Button(buttonFrame, text="Select Task No.", command=lambda: getNum())
taskButton.place(relx=0.25, relheight=1, relwidth=0.25)

gestureButton = tk.Button(buttonFrame, text="Select Query Gesture", command=lambda: getQuery())
gestureButton.place(relx=0.5, relheight=1, relwidth=0.25)

startButton = tk.Button(buttonFrame, text="Generate Results", command=lambda: generateResults())
startButton.place(relx=0.75, relheight=1, relwidth=0.25)

resultFrame = tk.Frame(root, bg='#000000', bd=5)
resultFrame.place(relx=0.5, rely=0.4, relwidth=0.5, relheight=0.5, anchor='n')

resultLabel = tk.Label(resultFrame, font=('Arial Bold', 15))
resultLabel.place(relwidth=1, relheight=1)

root.mainloop()
