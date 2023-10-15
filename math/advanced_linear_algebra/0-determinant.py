#!/usr/bin/env python3
"""
Calculates the determinant of a matrix.
"""


def determinant(matrix):
    """Calculates the determinant of a matrix

    Args.
        matrix: A list of lists whose determinant should be calculated.
    Returns:
        The determinant of a matrix.
    """

    m_len = len(matrix)
    if m_len == 1 and not matrix[0]:
        return 1
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(x, list) for x in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == len(x) for x in matrix):
        raise ValueError("matrix must be a square matrix")
    if m_len == 1:
        return matrix[0][0]
    if m_len == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = []
    for i in range(m_len):
        if i == 0:
            det.append(matrix[0][i]
                       * calc([sub[i + 1:] for sub in matrix[i + 1:]]))
        elif i == m_len - 1:
            if i % 2 != 0:
                det.append(-matrix[0][i]
                           * calc([sub[:i] for sub in matrix[1:]]))
            else:
                det.append(matrix[0][i]
                           * calc([sub[:i] for sub in matrix[1:]]))

        else:
            if i % 2 != 0:
                det.append(-matrix[0][i]
                           * calc([sub[:i]
                                   + sub[i + 1:] for sub in matrix[1:]]))
            else:
                det.append(matrix[0][i]
                           * calc([sub[:i]
                                   + sub[i + 1:] for sub in matrix[1:]]))

    return sum(det)


if __name__ == '__main__':
  # Example usage:

  print(determinant([[1, 2], [3, 4]]))
  print(determinant([[1, 1], [1, 1]]))
  print(determinant([[5, 7, 9], [3, 1, 8], [6, 2, 4]]))
