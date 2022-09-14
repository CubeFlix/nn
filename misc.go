// misc.go
// Miscellaneous helper functions.

package nn


import (
	"errors"
	"time"
	"math/rand"
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


// Return the output values for binary categorization.
func OutputBinaryValues(X Matrix) Matrix {
	for i := 0; i < X.Rows; i++ {
		for j := 0; j < X.Cols; j++ {
			if X.M[i][j] < 0.5 {
				X.M[i][j] = 0
			} else {
				X.M[i][j] = 1
			}
		}
	}

	// Return the output matrix.
	return X
}


// Shuffle the X and Y matricies.
func ShuffleDataset(X, Y Matrix) (Matrix, Matrix) {
	// Seed the random number generator.
	rand.Seed(time.Now().UnixNano())

	// Shuffle the matricies.
	rand.Shuffle(X.Rows, func(i, j int) {
		X.M[i], X.M[j] = X.M[j], X.M[i]
		Y.M[i], Y.M[j] = Y.M[j], Y.M[i]
	})

	// Return the output matricies.
	return X, Y
}
