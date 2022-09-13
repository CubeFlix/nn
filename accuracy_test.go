// accuracy_test.go
// Testing for accuracy functions.

package nn

import (
	"testing"
)


func TestRegressionAccuracy(t *testing.T) {
	// Create the matricies.
	yHat, _ := NewMatrixFromSlice([][]float64{[]float64{0.1, 0.2}, []float64{0.2, 0.5}})
	Y, _ := NewMatrixFromSlice([][]float64{[]float64{0.105, 0.199}, []float64{0.3, 0.502}})

	// Calculate the accuracy.
	acc := RegressionAccuracy(yHat, Y, 0.01)

	if acc != 0.75 {
		t.Errorf("Invalid accuracy values.")
		return
	}
}


func TestCategoricalAccuracy(t *testing.T) {
        // Create the matricies.
        yHat, _ := NewMatrixFromSlice([][]float64{[]float64{0.1, 0.9}, []float64{0.8, 0.2}})
        Y, _ := NewMatrixFromSlice([][]float64{[]float64{0, 1}, []float64{1, 0}})

        // Calculate the accuracy.
        acc := CategoricalAccuracy(yHat, Y)

        if acc != 1 {
                t.Errorf("Invalid accuracy values.")
                return
        }
}
