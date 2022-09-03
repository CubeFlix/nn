// matrix.go
// Matricies and matrix operations for the neural network.

package nn

import (
	"errors"
	"fmt"
	"math"
)


// Invalid matrix dimensions error function.
func invalidMatrixDimensionsError(rows int, cols int) error {
	return errors.New(fmt.Sprintf("Invalid matrix dimensions: %d, %d", rows, cols))
}

// Invalid matrix index error function.
func invalidMatrixIndexError(row int, col int) error {
	return errors.New(fmt.Sprintf("Invalid index for matrix: %d, %d", row, col))
}

// Invalid slice dimensions error function.
func invalidSliceDimensionsError(length int) error {
	return errors.New(fmt.Sprintf("Invalid slice dimensions: %d", length))
}


// Matrix type.
type Matrix struct {
	Rows int         // Number of rows. 
	Cols int         // Number of columns.
	M    [][]float64 // Matrix data.
}

// New matrix function.
func NewMatrix(rows int, cols int) (Matrix, error) {
	// Check the dimensions.
	if rows < 1 || cols < 1 {
		return Matrix{}, invalidMatrixDimensionsError(rows, cols)
	}

	// Create the matrix data.
	M := make([][]float64, rows)

	// Create a slice for each row.
	for i, _ := range M {
		M[i] = make([]float64, cols)
	}

	// Create a new matrix given the size.
	return Matrix{
		Rows: rows,
		Cols: cols,
		M: M,
	}, nil
}

// New matrix from slice function.
func NewMatrixFromSlice(slice [][]float64) (Matrix, error) {
	// Check the dimensions.
	rows := len(slice)
	if rows < 1 {
		return Matrix{}, invalidMatrixDimensionsError(rows, 0)
	}
	cols := len(slice[0])
	if cols < 1 {
		return Matrix{}, invalidMatrixDimensionsError(rows, cols)
	}

	// Create a new matrix given the size.
	return Matrix {
		Rows: rows,
		Cols: cols,
		M: slice,
	}, nil
}

// Matrix get function.
func (m *Matrix) Get(row int, col int) (float64, error) {
	// Get the value of matrix[row][col].

	// Check that the value exists.
	if row >= m.Rows || col >= m.Cols {
		return 0, invalidMatrixIndexError(row, col)
	}

	// Return the value.
	return m.M[row][col], nil
}

// Matrix set function.
func (m *Matrix) Set(row int, col int, value float64) error {
	// Set the value at matrix[row][col].

        // Check that the value exists.
        if row >= m.Rows || col >= m.Cols {
                return invalidMatrixIndexError(row, col)
        }

        // Set the value.
	m.M[row][col] = value

	return nil
}

// Matrix get row function.
func (m *Matrix) GetRow(row int) ([]float64, error) {
	// Get the values of row 'row'.

	// Check that the row exists.
	if row >= m.Rows {
		return nil, invalidMatrixIndexError(row, -1)
	}

	// Return the row.
	return m.M[row], nil
}

// Matrix get column function.
func (m *Matrix) GetColumn(col int) ([]float64, error) {
        // Get the values of column col.

        // Check that the column exists.
        if col >= m.Cols {
                return nil, invalidMatrixIndexError(-1, col)
        }

        // Create the slice and loop over the rows.
	s := make([]float64, m.Rows)
	for i, _ := range s {
		s[i] = m.M[i][col]
	}

	// Return the column.
        return s, nil
}

// Matrix insert row function.
func (m *Matrix) SetRow(row int, values []float64) error {
	// Set the row 'row' to values.

	// Check that the row exists.
	if row >= m.Rows {
		return invalidMatrixIndexError(row, -1)
	}

	// Check that the slice is the right length.
	if len(values) != m.Cols {
		return invalidSliceDimensionsError(len(values))
	}

	// Insert the row.
	m.M[row] = values

	return nil
}

// Matrix insert column function.
func (m *Matrix) SetColumn(col int, values []float64) error {
	// Set the column col to values.

	// Check that the column exists.
	if col >= m.Cols {
		return invalidMatrixIndexError(-1, col)
	}

	// Check that the slice is the right length.
	if len(values) != m.Rows {
		return invalidSliceDimensionsError(len(values))
	}

	// Loop over the rows and insert the column.
	for i := 0; i < m.Rows; i++ {
		m.M[i][col] = values[i]
	}

	return nil
}

// Matrix equality function.
func (m *Matrix) Equals(b Matrix) bool {
	// Check if the size and values of the matricies m and b are equal.

	// Check that the size is correct.
	if m.Rows != b.Rows || m.Cols != b.Cols {
		return false
	}

	// Check that the values are equal.
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if m.M[i][j] != b.M[i][j] {
				return false
			}
		}
	}

	// Return true.
	return true
}


// Matrix addition/subtraction functions.
func (m *Matrix) Add(b Matrix) (Matrix, error) {
	// Add matricies m and b and return the answer.

	// Check that the size is correct.
	if m.Rows != b.Rows || m.Cols != b.Cols {
		return Matrix{}, invalidMatrixDimensionsError(b.Rows, b.Cols)
	}

	// Create the new matrix.
	ans, _ := NewMatrix(m.Rows, m.Cols)

	// Add the matricies by looping over the values.
	for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
			ans.M[i][j] = m.M[i][j] + b.M[i][j]
		}
	}

	// Return the new matrix.
	return ans, nil
}

func (m *Matrix) Sub(b Matrix) (Matrix, error) {
        // Subtract matrix b from m and return the answer.

        // Check that the size is correct.
        if m.Rows != b.Rows || m.Cols != b.Cols {
                return Matrix{}, invalidMatrixDimensionsError(b.Rows, b.Cols)
        }

        // Create the new matrix.
        ans, _ := NewMatrix(m.Rows, m.Cols)

        // Subtract the matricies by looping over the values.
        for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        ans.M[i][j] = m.M[i][j] - b.M[i][j]
                }
        }

        // Return the new matrix.
        return ans, nil
}

// Matrix negation function.
func (m *Matrix) Neg() Matrix {
	// Negate the matrix m.

	// Create the new matrix.
	ans, _ := NewMatrix(m.Rows, m.Cols)

	// Negate the matrix by looping over all the values.
	for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
			ans.M[i][j] = -m.M[i][j]
		}
	}

	// Return the new matrix.
	return ans
}

// Matrix scalar functions.
func (m *Matrix) AddScalar(x float64) Matrix {
	// Create the new matrix.
        ans, _ := NewMatrix(m.Rows, m.Cols)

	// Add scalar x to every value.
	for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        ans.M[i][j] = m.M[i][j] + x
		}
        }

	// Return the new matrix.
	return ans
}

func (m *Matrix) MulScalar(x float64) Matrix {
	// Create the new matrix
	ans, _ := NewMatrix(m.Rows, m.Cols)

	// Add scalar x to every value.
        for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        ans.M[i][j] = m.M[i][j] * x
                }
        }

	// Return the new matrix.
        return ans
}

func (m *Matrix) PowScalar(x float64) Matrix {
	// Create the new matrix
        ans, _ := NewMatrix(m.Rows, m.Cols)

        // Add scalar x to every value.
        for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        ans.M[i][j] = math.Pow(m.M[i][j], x)
                }
        }

	// Return the new matrix.
        return ans
}

// Matrix dot product.
func (m *Matrix) Dot(b Matrix) (Matrix, error) {
	// Calculate the dot product of m and b.

	// Check that the dimensions are correct.
	if m.Cols != b.Rows {
		return Matrix{}, invalidMatrixDimensionsError(b.Rows, b.Cols)
	}

	// Create the new matrix.
	ans, _ := NewMatrix(m.Rows, b.Cols)

	// Loop over the matricies and calculate the dot product.
	for i := 0; i < m.Rows; i++ {
                for j := 0; j < b.Cols; j++ {
			// Matrix m columns multiplied by matrix b rows.
			sum := float64(0)
			for n := 0; n < m.Cols; n++ {
				sum += m.M[i][n] * b.M[n][j]
			}

			// Set the new value in the matrix.
			ans.M[i][j] = sum
                }
        }

	// Return the answer.
	return ans, nil
}

// Matrix transpose function.
func (m *Matrix) T() Matrix {
	// Create the newly-sized matrix.
	ans, _ := NewMatrix(m.Cols, m.Rows)

	// Set the values.
	for i := 0; i < m.Rows; i++ {
		col, _ := m.GetRow(i)
		ans.SetColumn(i, col)
	}

	// Return the matrix.
	return ans
}

// Matrix sum over axis. Maintains the original dimensions.
func (m *Matrix) Sum(axis int) Matrix {
	// Check that the axis is valid.
	if axis != 0 && axis != 1 {
		panic(fmt.Sprintf("Invalid for sum axis: %d", axis))
	}

	// Check the axis.
	if axis == 0 {
		// Create the new matrix.
		ans, _ := NewMatrix(1, m.Cols)

		// Calculate the sum over the columns.
		for i := 0; i < m.Cols; i++ {
			s := float64(0)
			for j := 0; j < m.Rows; j++ {
				s += m.M[j][i]
			}
			ans.M[0][i] = s
		}

		// Return the answer.
		return ans
	}

	if axis == 1 {
		// Create the new matrix.
		ans, _ := NewMatrix(1, m.Rows)

		// Calculate the sum over the rows.
		for i := 0; i < m.Rows; i++ {
			s := float64(0)
			for j := 0; j < m.Cols; j++ {
				s += m.M[i][j]
			}
			ans.M[0][i] = s
		}

		// Return the answer.
		return ans
	}
	return Matrix{}
}


func Clip(X Matrix) Matrix {
	for i := 0; i < X.Rows; i++ {
		for j := 0; j < X.Cols; j++ {
			if X.M[i][j] < 0 {
				X.M[i][j] = float64(1e-7)
			}
			X.M[i][j] = X.M[i][j]
		}
	}

	return X
}
