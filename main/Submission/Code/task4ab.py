import pandas as pd
from utils import print_dict
import sys

# groups gestures based on their contribution to the latent components
def group_by_semantics(file_name, take_absolute=False):
    # Row containing latent topics
    # Col containing Gestures
    # value being its contribution
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


def main():
    # Task 4a
    print("\n-----------------------------------------------------------\n")
    print("Task 4a Results: ")
    print("\n-----------------------------------------------------------\n")
    group_by_semantics('phase_2_task3_SVD_contributions.csv', take_absolute=True)

    # Task 4b
    print("\n-----------------------------------------------------------\n")
    print("Task 4b Results: ")
    print("\n-----------------------------------------------------------\n")
    group_by_semantics('phase_2_task3_NMF_contributions.csv', take_absolute=True)

if __name__ == "__main__":
    main()
