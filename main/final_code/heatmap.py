import seaborn as sns
import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':

    # data_fi le_path = input("Enter the path of the gesture file: ")
    data_file_path = "/Users/vchitteti/Projects/mwdb/main/final_code/task3_dot_sim_matrix.csv"
    heat_map = []

    with open(data_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        print(reader)
        i = 0

        for row in reader:
            if i == 0:
                i += 1
            else:
                heat_map.append(row)

    sns.heatmap(heat_map, cmap='gray')
    plt.show()
    print("end")
