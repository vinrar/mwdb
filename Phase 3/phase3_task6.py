import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.simpledialog as tkSimpleDialog
import os, json
from gui_phase3_task3 import locality_sensitive_hashing, preprocessing, set_updated_query, get_appropriate_ratio, \
    query_algorithm
from task5 import get_ppr2, get_ppr_changing_query
from task_4_code_try_2 import get_task4_results

HEIGHT = 600
WIDTH = 800
GESTURE_PATH = ''
TASK_NUMBER = None
QUERY_GESTURE = None
RELEVANT_FEEDBACK = []
IRRELEVANT_FEEDBACK = []
QUERY_MODE = None


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
def getTaskNum():
    global TASK_NUMBER
    n = tkSimpleDialog.askinteger("Task Number", "Enter the task number (4/5)", parent=root, minvalue=4, maxvalue=5)
    print('Task Number: ', n)
    TASK_NUMBER = n


# Function to get the query gesture
def getQueryMode():
    global QUERY_MODE
    q = tkSimpleDialog.askinteger("Query Mode", "Enter the query mode(0=Reordering results/1=Revising Query)",
                                  parent=root, minvalue=0, maxvalue=1)
    QUERY_MODE = q


# Function to get input for Task 3
def getTask3Input():
    pass


# Function to generate similarity results
def generateResults():
    global GESTURE_PATH, TASK_NUMBER, RELEVANT_FEEDBACK, IRRELEVANT_FEEDBACK, QUERY_GESTURE, QUERY_MODE

    q = tkSimpleDialog.askinteger("Query Gesture", "Enter the query gesture", parent=root)
    print('Query Gesture: ', q)
    QUERY_GESTURE = str(q)

    firstRun = True
    runAgain = True
    gesture_list = []
    feed_back_results = []
    hash_tables = []
    while runAgain:
        # generate results for similarity
        if firstRun:
            # run for the first time without feedback

            # input for LSH

            LSH_input_string = tkSimpleDialog.askstring("LSH Input",
                                                        "Enter parameters for LSH (L, k, Vector Model(tf/tfidf), t)")
            L, k, model, t = LSH_input_string.split(",")

            L = int(L)
            k = int(k)
            t = int(t)

            query_gesture = QUERY_GESTURE

            with open(model + '_vectors.json', 'r') as fp:
                vectors = json.load(fp)

            # preprocessing for LSH
            dims = len(list(vectors.values())[0])
            hash_tables = preprocessing(L, k, dims, vectors)
            # locality_sensitive_hashing(L, k, )
            results_to_display, gesture_list = locality_sensitive_hashing(L, k, query_gesture, t, vectors, hash_tables,
                                                                          gui=True)

            # results_to_display = [i for i in range(1,11)]
            # resultLabel['text'] = formatResult(results_to_display)
            resultLabel['text'] = results_to_display
            firstRun = False
            RELEVANT_FEEDBACK.append(query_gesture)
        else:
            # run for the subsequent times with feedback
            if TASK_NUMBER == 4:
                print("Need to do")
                # feed_back_results = get_task4_results(mode, vector_model, initial_similarity_results, relevance_results):
            else:
                if QUERY_MODE == 0:
                    feed_back_results = get_ppr2(len(gesture_list), 0.85, gesture_list, RELEVANT_FEEDBACK,
                                                 IRRELEVANT_FEEDBACK, gui=True)
                else:
                    irrel_gestures, ratio = get_ppr_changing_query(10, 10, 0.85, IRRELEVANT_FEEDBACK)
                    rel_gestures, ratio = get_ppr_changing_query(10, 10, 0.85, RELEVANT_FEEDBACK)
                    gesture_list = set(rel_gestures) - set(irrel_gestures)
                    for rel_ges in RELEVANT_FEEDBACK:
                        gesture_list.add(rel_ges)
                    gesture_list = list(gesture_list)
                    query_vector = set_updated_query(gesture_list,
                                                     get_appropriate_ratio(gesture_list, rel_gestures, ratio), vectors)
                    vectors[-1] = query_vector
                    result = query_algorithm(-1, vectors, hash_tables, t, k)
                    output = ''
                    for i, (k, v) in zip(range(t), result.items()):
                        print(i + 1, "Gesture: ", k, ",\tSimilarity Score: ", v)
                        import numpy as np
                        output += '{} - Gesture: {},\tSimilarity Score: {}\n'.format(i + 1, k, np.round(v, 3))
                        gesture_list.append(str(k))
                    feed_back_results = output
            resultLabel['text'] = feed_back_results

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

taskButton = tk.Button(buttonFrame, text="Select Task No.", command=lambda: getTaskNum())
taskButton.place(relx=0.25, relheight=1, relwidth=0.25)

gestureButton = tk.Button(buttonFrame, text="Select Query Mode", command=lambda: getQueryMode())
gestureButton.place(relx=0.5, relheight=1, relwidth=0.25)

startButton = tk.Button(buttonFrame, text="Select Query Gesture", command=lambda: generateResults())
startButton.place(relx=0.75, relheight=1, relwidth=0.25)

resultFrame = tk.Frame(root, bg='#000000', bd=5)
resultFrame.place(relx=0.5, rely=0.4, relwidth=0.5, relheight=0.5, anchor='n')

resultLabel = tk.Label(resultFrame, font=('Arial Bold', 15))
resultLabel.place(relwidth=1, relheight=1)

root.mainloop()
