import pickle

count_of_bins = {}

def get_count_in_bins(transformed_matrix):
    for each_gesture_file in transformed_matrix:
        index = transformed_matrix[each_gesture_file].index(max(transformed_matrix[each_gesture_file])) #finding the index of the maximum element in the list
        if(str(index + 1) in  count_of_bins.keys()):
            count_of_bins[str(index + 1)].append(each_gesture_file)
        else:
            count_of_bins[str(index + 1)] = [each_gesture_file]

if __name__ == '__main__':
    dir = "/home/asim/Desktop/ankit_acad_temp/MWDB/Phase_2_stuff/Amey_task0a_wrdfiles"
    file_name = "phase_2_task_1_transformed_matrix"
    file_path = dir + "/" + file_name
    file_handler = open(file_path, 'rb')

    transformed_matrix = pickle.load(file_handler)
    get_count_in_bins(transformed_matrix)

    output_file_name = "phase_2_tas4a_output"
    file_handler = open(dir + "/" + output_file_name, 'wb')
    pickle.dump(count_of_bins, file_handler)