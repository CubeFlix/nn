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
