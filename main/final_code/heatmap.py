import seaborn as sns
import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':

    data_file_path = input("Enter the path of the gesture file: ")
    # data_file_path = "/Users/vchitteti/Projects/mwdb/main/final_code/task3_dot_sim_matrix.csv"
    heat_map = []
    x_labels = []

    with open(data_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        print(reader)
        i = 0

        for row in reader:
            if i == 0:
                i += 1
            else:
                x_labels.append(row[0])
                heat_map.append(row[1:-1])

    sns.heatmap(heat_map, cmap='CMRmap')
    plt.show()
    print("end")
