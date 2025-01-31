import os
import numpy as np
import scipy
import json
from scipy.integrate import quad
from scipy import stats
import sys

def format_float(value):
    return "%.6f" % value

# returns the gaussian distribution (mean=0, std=0.25) value of x
def normal_distribution_function(x):
    mean = 0
    std = 0.25
    value = scipy.stats.norm.pdf(x, mean, std)
    return value

# returns the definite integral of gaussian distribution between x1 and x2
def definite_integral(x1, x2):
    res, err = quad(normal_distribution_function, x1, x2)
    return res

# returns the integer version of a numpy integer
def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Run python phase3_task0a.py <Directory> <Resolution> <Window Length> <Shift Length>')
        sys.exit(0)
    directory = sys.argv[1]
    r = int(sys.argv[2])
    w = int(sys.argv[3])
    s = int(sys.argv[4])

    print("Directory: {}\nResolution: {}\nWindow Length: {}\nShift Length: {}".format(directory, r, w, s))

    # number of bands the sensor value needs to be quantized into
    levels = 2 * r
    # list with the lengths of the band
    lengths = []  

    # append lengths of each band in the lengths list
    for i in range(1, levels + 1):
        lengths.append(2 * (definite_integral((i - r - 1) / r, (i - r) / r) / definite_integral(-1, 1)))

    # initialize the end of band 1 from -1
    lengths[0] += -1
    representative_map = {0: (-1 + lengths[0]) / 2}

    # represent the band from its mid-point
    for i in range(1, levels):
        lengths[i] += lengths[i - 1]
        representative_map[i] = (lengths[i - 1] + lengths[i]) / 2

    # map to store metadata
    result = {}
    # map to store output as per project requirement
    result_display = {}

    for dir_path, dir_names, filenames in os.walk(directory):
        for dir_name in dir_names:
            new_directory = os.path.join(directory, dir_name) + "/"
            for file in os.listdir(new_directory):
                if file.endswith(".csv"):
                    f = file.split('.')[0]
                    
                    curr_file = result[f] if f in result.keys() else {}
                    curr_file[dir_name] = {}

                    curr_file_display = result_display[f] if f in result_display.keys() else {}
                    curr_file_display[dir_name] = {}

                    # store sensor values from csv file to an array
                    arr = np.genfromtxt(os.path.join(new_directory, file), delimiter=",")

                    avg_values = arr.mean(axis=1)
                    std_values = arr.std(axis=1)

                    # normalize all the values between -1 and 1
                    arr = -1 + (2 * (arr - arr.min(axis=1)[:, None]) / (arr.max(axis=1) - arr.min(axis=1))[:, None])

                    # quantize all the values into one of the levels based on the lengths
                    discrete = np.digitize(arr, lengths, right=True)

                    for i in range(len(discrete)):
                        word_list = []
                        word_list_display = []
                        time_length = len(discrete[i])
                        # slide the window on the time series data and write window words to the file
                        for j in range((time_length - w + 1) // s):
                            word = discrete[i][j * s:j * s + w]
                            word_avg = np.vectorize(representative_map.get)(word)
                            word_avg = np.mean(word_avg)
                            word_list.append([list(word), word_avg])
                            word_list_display.append([[dir_name, i, list(word)], word_avg])
                        
                        # store metadata
                        curr_file[dir_name][i] = {"avg": avg_values[i], "stdev": std_values[i], "words": word_list}
                        # store display data
                        curr_file_display[dir_name][i] = {"avg": avg_values[i], "stdev": std_values[i], "words": word_list_display}
                    
                    result[f] = curr_file
                    result_display[f] = curr_file_display
        break

    # store the .wrd files as per the project requirement
    # note: these won't be used in future tasks
    # new_directory = directory + '/data'
    # if not os.path.exists(new_directory):
    #     os.mkdir(new_directory)
    # for f in result.keys():
    #     outF = open(os.path.join(new_directory, f + ".wrd"), "w")
    #     json.dump(result_display[f], outF, default=convert)
    #     outF.close()

    # store the .wrds files as metadata to be used in future tasks
    for f in result_display.keys():
        outF = open(os.path.join(directory, f + ".wrd"), "w")
        json.dump(result[f], outF, default=convert)
        outF.close()
