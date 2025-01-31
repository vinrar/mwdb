import sys
import os


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('Run python phase3_preprocessing.py <Directory> <Resolution> <Window Length> <Shift Length>')
        sys.exit(0)
    directory = sys.argv[1]
    r = sys.argv[2]
    w = sys.argv[3]
    s = sys.argv[4]

    # print('Executing python phase3_task0a.py ' + directory + ' ' + r + ' ' + w + ' ' + s)
    os.system('python phase3_task0a.py ' + directory + ' ' + r + ' ' + w + ' ' + s)

    # print('Executing python phase3_task0b.py ' + directory)
    os.system('python phase3_task0b.py ' + directory)

    # print('Executing python create_vector_matrix.py ' + directory)
    os.system('python create_vector_matrix.py ' + directory)

    # print('Executing python Phase3_task1_pca_cos_sim.py' + )
    os.system('python Phase3_task1_pca_cos_sim.py')

