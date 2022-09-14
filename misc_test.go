// misc_test.go
// Testing for miscellaneous functions.

package nn


import (
	"testing"
)


// Test conversion from sparse to one-hot matricies.
func TestSparseToOneHot(t *testing.T) {
	// Create a new matrix.
	m, _ := NewMatrixFromSlice([][]float64{[]float64{1}, []float64{3}, []float64{0}, []float64{2}})

	// Convert to one-hot matrix.
	o, err := SparseToOneHot(m, 5)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	// Check that the values are correct.
	if o.M[1][3] != 1 {
		t.Errorf("Invalid output values.")
		return
	}
}


// Test row max matrix function.
func TestRowMax(t *testing.T) {
	// Create a new matrix.
        m, _ := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}, []float64{3, -1, 0.5}, []float64{2, 5, 1}, []float64{0.1, 6, 7.5}})

	// Get the max values.
	o := RowMax(m)

	// Check that the values are correct.
	if o.M[2][0] != 1 {
		t.Errorf("Invalid output values.")
		return
	}
}


// Test the output binary values function.
func TestOutputBinaryValues(t *testing.T) {
	// Create a new matrix.
        m, _ := NewMatrixFromSlice([][]float64{[]float64{0.2}, []float64{0.1}, []float64{0.9}, []float64{0.78}})

	// Get the output values.
	o := OutputBinaryValues(m)

	// Check that the values are correct.
	if o.M[2][0] != 1 || o.M[0][0] != 0 {
		t.Errorf("Invalid output values.")
		return
	}
}


// Test shuffling matricies.
func TestShuffleDataset(t *testing.T) {
	// Create the matricies.
	X, _ := NewMatrixFromSlice([][]float64{[]float64{1}, []float64{3}, []float64{0}, []float64{2}})
	Y, _ := NewMatrixFromSlice([][]float64{[]float64{3}, []float64{4}, []float64{6}, []float64{7}})

	// Shuffle the matricies.
	X, Y = ShuffleDataset(X, Y)
}
