import numpy as np
import inverse_matrix as m
from colors import bcolors
from matrix_utility import print_matrix

def norm(mat):
    size = len(mat)
    max_row = 0
    for row in range(size):
        sum_row = 0
        for col in range(size):
            sum_row += abs(mat[row][col])
        if sum_row > max_row:
            max_row = sum_row
    return max_row


def condition_number(A):
    # Step 1: Calculate the max norm (infinity norm) of A
    norm_A = norm(A)

    # Step 2: Calculate the inverse of A
    A_inv = m.matrix_inverse(A)

    # Step 3: Calculate the max norm of the inverse of A
    norm_A_inv = norm(A_inv)

    # Step 4: Compute the condition number
    cond = norm_A * norm_A_inv

    #print(bcolors.OKBLUE, "A:", bcolors.ENDC)
    #print_matrix(A)

    #print(bcolors.OKBLUE, "inverse of A:", bcolors.ENDC)
    #print_matrix(A_inv)

    print(bcolors.OKBLUE, "Max Norm of A:", bcolors.ENDC, norm_A, "\n")

    #print(bcolors.OKBLUE, "max norm of the inverse of A:", bcolors.ENDC, norm_A_inv)

    return cond



"""
    print(" Date:19/2/2024 \n"
          " Group: Daniel Houri , 209445071 \n"
          "        Yakov Shtefan , 208060111 \n"
          "        Vladislav Rabinovich , 323602383 \n"
          "        Eve Hackmon, 209295914\n""
          " Git: https://github.com/YakovShtef/Bohan1.git \n"
          " Name: Yakov Shtefan, 208060111 \n")
"""
if __name__ == '__main__':
    A = np.array([[2, 1, 0],
                  [3, -1, 0],
                  [1, 4, -2]])
    cond = condition_number(A)

    #print(bcolors.OKGREEN, "\n condition number: ", cond, bcolors.ENDC)