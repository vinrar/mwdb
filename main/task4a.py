import pandas as pd
from mwdb.main.utils import print_dict


# Row containing latent topics
# Col containing Gestures
# value being its contribution
def group_by_semantics(file_name='phase_3_task3_SVD_output.csv', take_absolute=False):
    similarity_matrix = pd.read_csv(file_name, index_col=0)
    p_groups = {}
    for file_number, col in similarity_matrix.transpose().iterrows():
        maximum = float('-inf')
        index = 0
        for i, element in col.iteritems():
            if take_absolute:
                element = abs(element)
            if element > maximum:
                maximum = element
                index = i
        if index not in p_groups:
            p_groups[index] = set()
        p_groups[index].add(file_number)
    print_dict(p_groups)


if __name__ == "__main__":
    group_by_semantics()
    group_by_semantics(take_absolute=True)
