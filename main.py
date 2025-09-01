import numpy as np


def sanitize_input(A, b):
    """
    Sanitize and validate the input matrices A and b.

    Parameters:
    A (numpy.ndarray): Coefficient matrix.
    b (numpy.ndarray): Right-hand side vector.

    Returns:
    Sanitized matrices A and b.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square (n x n).")

    if A.shape[0] != b.shape[0]:
        raise ValueError("The number of rows in A must match the number of rows in b.")
    
    if np.linalg.det(A) == 0:
        raise ValueError("Matrix A must be non-singular.")

    return A, b


def gauss_elim(A, b):
    """
    Solve the system of linear equations Ax = b using Gaussian elimination.

    Parameters:
    A (numpy.ndarray): Coefficient matrix (n x n).
    b (numpy.ndarray): Right-hand side vector (n,1).

    Returns:
    numpy.ndarray: Solution vector x (n,1).
    """

    A, b = sanitize_input(A, b)
    n = len(A[0])
    x = np.zeros((n,1))
    
    # Reduced Echelon Form
    A, b = REF(A, b, n)

    # Back Substitution
    for i in range(n-1, -1, -1):
        val_to_sub = 0
        for k in range(i+1, n):
            val_to_sub -= A[i][k] * x[k]

        x[i] = b[i] + val_to_sub
    
    return x


def REF(A, b, n):
    """
    Create a pivot for i-th row and eliminate i-th column.

    Parameters:
    A (numpy.ndarray): Coefficient matrix (n x n).
    b (numpy.ndarray): Right-hand side vector (n,1).
    n (int): Size of the matrix.
    i (int): Current row index.
    """
    for i in range(n):
        # Creating a pivot for the i-th row
        # Performing Ri = Ri * (1 / pivot_value)
        pivot_value = A[i][i]

        
        if pivot_value == 0:
            for j in range(i+1, n):
                if A[j][i] != 0:
                    # Swapping rows i and j, where A[j][i] is non-zero
                    A[[i, j]] = A[[j, i]]
                    b[[i, j]] = b[[j, i]]
                    break
            
        
        pivot_value = A[i][i]
         # perform Ri = Ri / pivot_value
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
    A = [[2, -1, 1], [1, 3, 2], [1, -1, 2]]
    b = [3, 13, 5]

    x = gauss_elim(A,b)
    x_hat = np.linalg.solve(A, b)
    print("Our Solution: ", x.flatten())
    print("Numpy Solution: ", x_hat)