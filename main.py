import numpy as np


def gauss_elim(A, b):
    """
    Solve the system of linear equations Ax = b using Gaussian elimination.

    Parameters:
    A (numpy.ndarray): Coefficient matrix (n x n).
    b (numpy.ndarray): Right-hand side vector (n,1).

    Returns:
    numpy.ndarray: Solution vector x (n,1).
    """

    n = len(A[0])
    x = np.zeros((n,1))
    
    # Forward Pass
    for i in range(n):
        solve_pivot(A, b, n, i)

    # Matrix is now in REF
    # Back Substitution
    # x[n-1] = b[n-1]
    for i in range(n-1, -1, -1):
        val_to_sub = 0
        for k in range(i+1, n):
            val_to_sub -= A[i][k] * x[k]

        x[i] = b[i] + val_to_sub
        

    return x


def solve_pivot(A, b, n, i):
    """
    Create a pivot for i-th row and eliminate i-th column.

    Parameters:
    A (numpy.ndarray): Coefficient matrix (n x n).
    b (numpy.ndarray): Right-hand side vector (n,1).
    n (int): Size of the matrix.
    i (int): Current row index.
    """

    # Creating a pivot for the i-th row
    # Performing Ri = Ri * (1 / pivot_value)
    pivot_value = A[i][i]
    A[i] = A[i] / pivot_value
    b[i] = b[i] / pivot_value

    # Eliminating the i-th column
    for j in range(i+1, n):
        val = A[j][i]
        # perform Rj = Rj - val * Ri
        A[j] = A[j] - (val * A[i])
        b[j] = b[j] - (val * b[i])    

    return A, b

    
    




if __name__ == "__main__":
    # Example usage
    n = 2
    A = np.array([[1, 1], [1, -1]], dtype=float)
    b = np.array([5, 1], dtype=float).reshape(n,1)

    x = gauss_elim(A,b)
    print(x)