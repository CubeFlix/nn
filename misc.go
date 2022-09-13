// misc.go
// Miscellaneous helper functions.

package nn


import (
	"errors"
)


// Transform sparse matricies to one-hot matricies.
func SparseToOneHot(X Matrix, size int) (Matrix, error) {
	// Check that the size is valid.
	if size <= 0 {
		return Matrix{}, errors.New("nn.SparseToOneHot: Invalid size attribute.")
	}

	// Create the new matrix.
	ans, _ := NewMatrix(X.Rows, size)

	// Loop over the matrix, transforming each sparse vector into one-hot vectors.
	for i := 0; i < X.Rows; i++ {
		ans.M[i][int(X.M[i][0])] = 1
	}

	// Return the output matrix.
	return ans, nil
}


// Get the index of the maximum value for each row.
func RowMax(X Matrix) Matrix {
	// Create the new matrix.
	ans, _ := NewMatrix(X.Rows, 1)

	// Loop over the matrix, getting the largest value.
	for i := 0; i < X.Rows; i++ {
		var largest int
		var largestValue float64
		for j := 0; j < X.Cols; j++ {
			if X.M[i][j] > largestValue {
				largest = j
				largestValue = X.M[i][j]
			}
		}

		// Set the largest value.
		ans.M[i][0] = float64(largest)
	}

	// Return the output matrix.
	return ans
}
