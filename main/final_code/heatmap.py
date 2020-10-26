import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def main():

    if len(sys.argv) < 1:
        print('Run python heatmap.py <User Option>')
        sys.exit(0)
    user_option = sys.argv[1]

    if user_option == '1':
        file_name = "task3_dot_sim_matrix.csv"
        task = "dot"
    elif user_option == '2':
        file_name = "task3_pca_sim_matrix.csv"
        task = "pca"
    elif user_option == '3':
        file_name = "task3_svd_sim_matrix.csv"
        task = "svd"
    elif user_option == '4':
        file_name = "task3_nmf_sim_matrix.csv"
        task = "nmf"
    elif user_option == '5':
        file_name = "task3_lda_sim_matrix.csv"
        task = "lda"
    elif user_option == '6':
        file_name = "task3_Edit_Dist_sim_mat.csv"
        task = "edit"
    elif user_option == '7':
        file_name = "task3_DTW_sim_matrix.csv"
        task = "dtw"

    df = pd.read_csv(file_name, index_col=0)
    cols = list(df.columns.values)
    cols = [int(c) for c in cols]
    new_cols = [str(c) for c in sorted(cols)]
    new_df = df[new_cols]
    new_df = new_df.sort_index()
    new_df.to_csv('test.csv')
    heatmap = new_df.to_numpy()
    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(20,20))
    hp = sns.heatmap(heatmap, cmap='CMRmap', xticklabels=new_df.index, yticklabels=new_df.index)
    fig = hp.get_figure()
    fig.savefig('heatmap_' + task + '.png')

if __name__ == "__main__":
    main()
