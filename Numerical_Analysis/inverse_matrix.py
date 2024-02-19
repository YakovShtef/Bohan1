from colors import bcolors
from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix
import numpy as np
import condition_of_linear_equations as c

"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""

def swap_rows(matrix, row1, row2):
    temp_matrix = matrix
    N = len(temp_matrix)
    for i in range(N):
        matrix[row1][i], matrix[row2][i] = matrix[row2][i], matrix[row1][i]
    return temp_matrix


def print_first_three_elementary_matrices(n, elementary_matrices):
    print(bcolors.WARNING, "\nPrinting the first three elementary matrices:\n", bcolors.ENDC)
    for i in range(min(3, len(elementary_matrices))):
        print(f"Elementary matrix {i + 1}:\n{elementary_matrices[i]}\n")

def print_last_three_elementary_matrices(n, elementary_matrices):
    print(bcolors.WARNING, "\nPrinting the last three elementary matrices:\n", bcolors.ENDC)
    start_index = max(0, len(elementary_matrices) - 3)
    for i in range(start_index, len(elementary_matrices)):
        print(f"Elementary matrix {i + 1}:\n{elementary_matrices[i]}\n")


def matrix_inverse(matrix):
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    determinant = np.linalg.det(matrix)
    print("determinant check: ", determinant)
    if determinant == 0:
        raise ValueError("Matrix is singular (determinant is equal to zero) , cannot find its inverse.")

    n = matrix.shape[0]
    identity = np.identity(n)
    count = 0
    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            j = i + 1
            while matrix[j][i] == 0:
                j += 1
            matrix = swap_rows(matrix, i, j)
            elementary_matrix = swap_rows(np.identity(n), i, j)
            identity = np.dot(elementary_matrix, identity)
            print(f"swap between row {i} and row {j}\n", matrix)


        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            count += 1
            scalar = 1 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")

            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            identity = np.dot(elementary_matrix, identity)
            #print(f"current inverse :\n {identity}")
            print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)


        # Zero out the elements above and below the diagonal
        for j in range(i, n):
            if i != j:
                count += 1
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                identity = np.dot(elementary_matrix, identity)
                #print(f"current inverse :\n {identity}")
                print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------", bcolors.ENDC)
    for i in range(n-1, -1, -1):
        for j in range(n-1, -1, -1):
            if i > j:
                count += 1
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j + 1} = R{j + 1} + ({scalar}R{i + 1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                identity = np.dot(elementary_matrix, identity)
                # print(f"current inverse :\n {identity}")
                print(bcolors.OKGREEN,
                      "------------------------------------------------------------------------------------------------------------------",
                      bcolors.ENDC)

    return identity

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
    np.set_printoptions(suppress=True, precision=4)
    #the input matrix test text
    A = np.array([[4, 1, 3],
                  [3, 7, 0],
                  [2, 4, 9]])
    #the solution vector for matrix A
    B = np.array([0, 0, 0])

    try:
        A_inverse = matrix_inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print(bcolors.OKBLUE, "\nThe solution vector of matrix A: \n", np.dot(A_inverse, B))

        print("=====================================================================================================================", bcolors.ENDC)

    except ValueError as e:
        print(str(e))