import numpy as np
from colors import bcolors
from matrix_utility import swap_rows_elementary_matrix, row_addition_elementary_matrix

def lu(A):
    N = len(A)
    L = np.eye(N)  # Create an identity matrix of size N x N

    for i in range(N):

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = i
        v_max = A[pivot_row][i]
        for j in range(i + 1, N):
            if abs(A[j][i]) > abs(v_max):
                v_max = A[j][i]
                pivot_row = j

        # if a principal diagonal element is zero,it denotes that matrix is singular,
        # and will lead to a division-by-zero later.
        if A[pivot_row][i] == 0:
            raise ValueError("can't perform LU Decomposition")

        # Swap the current row with the pivot row
        if pivot_row != i:
            e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
            print(f"elementary matrix for swap between row {i} to row {pivot_row} :\n {e_matrix} \n")
            A = np.dot(e_matrix, A)
            print(f"The matrix after elementary operation :\n {A}")
            print(bcolors.OKGREEN, "---------------------------------------------------------------------------",bcolors.ENDC)


        for j in range(i + 1, N):
            #  Compute the multiplier
            if A[j][i] != 0:
                m = -A[j][i] / A[i][i]
                e_matrix = row_addition_elementary_matrix(N, j, i, m)
                e_inverse = np.linalg.inv(e_matrix)
                L = np.dot(L, e_inverse)
                print(f"L :\n {L}")
                A = np.dot(e_matrix, A)
                print(f"elementary matrix to zero the element in row {j} below the pivot in column {i} :\n {e_matrix} \n")
                print(f"The matrix after elementary operation :\n {A}")
                print(bcolors.OKGREEN, "---------------------------------------------------------------------------", bcolors.ENDC)
    U = A
    return L, U

def lu_solve(A_b, n):
    L, U = lu(A_b)
    print("Lower triangular matrix L:\n", L)
    print("Upper triangular matrix U:\n", U)
    LU = np.dot(L, U)
    b = np.zeros(n)
    for i in range(n):
        b[i] = LU[i][n]
    print("b\n", b)

    newU = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            newU[i][j] = U[i][j]
    print("newU\n", newU)

    print("L_U\n", np.dot(L, newU))
    inverseNewU = np.linalg.inv(newU)
    inverseL = np.linalg.inv(L)
    result = np.dot(inverseNewU,np.dot(inverseL, b))

    for x in result:
        print(bcolors.OKBLUE, "{:.6f}".format(x))

if __name__ == '__main__':
    A_b = np.array([[0, 1, 1, 1, -8],
                    [1, 1, 2, 1, -20],
                    [2, 2, 4, 0, -2],
                    [1, 2, 1, 1, 4]])
    n = len(A_b)
    lu_solve(A_b,n)