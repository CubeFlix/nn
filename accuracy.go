// accuracy.go
// Accuracy calculations.

package nn


import (
	"math"
)


// Calculate the regression accuracy.
func RegressionAccuracy(yHat, Y Matrix, percision float64) float64 {
	accuracy := 0

	// Loop over all the samples and count the correct ones.
	for i := 0; i < Y.Rows; i++ {
		for j := 0; j < Y.Cols; j++ {
			if math.Abs(yHat.M[i][j] - Y.M[i][j]) < percision {
				accuracy += 1
			}
		}
	}

	// Return the final accuracy.
	return float64(accuracy) / float64(Y.Rows * Y.Cols)
}


// Calculate the categorical accuracy.
func CategoricalAccuracy(yHat, Y Matrix) float64 {
	// Get the final outputs for yHat.
	outputs := RowMax(yHat)

	accuracy := 0

	// Loop over all the samples and count the correct ones.
	for i := 0; i < Y.Rows; i++ {
		if Y.M[i][int(outputs.M[i][0])] == 1 {
			accuracy += 1
		}
	}

	// Return the final accuracy.
	return float64(accuracy) / float64(Y.Rows)
}


// Calculate the binary categorical accuracy.
func BinaryCategoricalAccuracy(yHat, Y Matrix) float64 {
	// Get the final outputs for yHat.
	outputs := OutputBinaryValues(yHat)

	accuracy := 0

	// Loop over all the samples and count the correct ones.
	for i := 0; i < Y.Rows; i++ {
                for j := 0; j < Y.Cols; j++ {
                        if outputs.M[i][j] == Y.M[i][j] {
                                accuracy += 1
                        }
                }
        }

	// Return the final accuracy.
	return float64(accuracy) / float64(Y.Rows * Y.Cols)
}
