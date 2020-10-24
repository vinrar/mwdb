import pandas as pd

if __name__ == "__main__":
    file_name = 'a.csv'
    p = 5
    similarity_matrix = pd.read_csv(file_name, index_col=0)
    p_groups = {}
    for file_number, col in similarity_matrix.transpose().iterrows():
        maximum = float('-inf')
        index = 0
        for i, element in col.iteritems():
            if element > maximum:
                maximum = element
                index = i
        if index not in p_groups:
            p_groups[index] = set()
        p_groups[index].add(file_number)
    print(p_groups)
