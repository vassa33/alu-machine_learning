#!/usr/bin/env python3
"""
Calculates the determinant of a matrix.
"""


def determinant(matrix):
  """Calculates the determinant of a matrix.

  Args:
    matrix: A list of lists whose determinant should be calculated.

  Returns:
    The determinant of matrix.

  Raises:
    TypeError: If matrix is not a list of lists.
    ValueError: If matrix is not square.
  """

  if not isinstance(matrix, list):
    raise TypeError("matrix must be a list of lists")

  if len(matrix) == 0 or len(matrix[0]) == 0:
    return 1

  if len(matrix) != len(matrix[0]):
    raise ValueError("matrix must be a square matrix")

  if len(matrix) == 1:
    return matrix[0][0]

  # Calculate the determinant using the Laplace expansion along the first row.
  determinant = 0
  for i in range(len(matrix[0])):
    determinant += matrix[0][i] * (-1)**i * determinant(
        [row[1:] for row in matrix[1:] if row[0] != matrix[0][i]])

  return determinant


if __name__ == '__main__':
  # Example usage:

  print(determinant([[1, 2], [3, 4]]))
  print(determinant([[1, 1], [1, 1]]))
  print(determinant([[5, 7, 9], [3, 1, 8], [6, 2, 4]]))
