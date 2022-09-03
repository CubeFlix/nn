// matrix_test.go
// Testing for matrix.go.

package nn

import (
	"testing"
	"reflect"
	"math"
)


// Test basic matrix get/set functions.
func TestMatrix(t *testing.T) {
	// Create the matrix.
	m, err := NewMatrix(3, 3)

	// Set some values.
	m.Set(0, 0, 3.14)
	m.Set(1, 2, 6)
	m.Set(2, 0, 623)
	m.Set(1, 1, 0)

	// Get the values.
	val1, err := m.Get(0, 0)
	val2, err := m.Get(1, 2)
	val3, err := m.Get(2, 0)
	val4, err := m.Get(1, 1)

	// Note: only need to check the last value (0), as it might not show up on the errors.
	if err != nil {
		t.Error(err.Error())
		return
	}

	// Check if the values are correct.
	if val1 != 3.14 || val2 != 6 || val3 != 623 || val4 != 0 {
		t.Error("Matrix values are incorrect.")
	}
}

// Test create matrix from slice.
func TestMatrixFromSlice(t *testing.T) {
	// Create the matrix.
	m, err := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}})
	if err != nil {
		t.Error(err.Error())
		return
	}

	// Check if the values are correct.
	if v, _ := m.Get(0, 0); v != 1 {
		t.Error("Matrix values are incorrect.")
		return
	}

	if v, _ := m.GetRow(1); !reflect.DeepEqual(v, []float64{4, 5, 6}) {
		t.Error("Matrix values are incorrect.")
		return
	}
}

// Test insert and get rows and columns from matrix.
func TestMatrixRowsCols(t *testing.T) {
	// Create the matrix.
	m, err := NewMatrix(3, 3)

	// Set a row.
	err = m.SetRow(0, []float64{1, 2, 3})
	if err != nil {
		t.Error(err.Error())
                return
	}

	// Set a column.
	err = m.SetColumn(1, []float64{1, 2, 3})
        if err != nil {
                t.Error(err.Error())
                return
        }

	// Get a row.
	row, err := m.GetRow(1)
	if err != nil {
		t.Error(err.Error())
		return
	}

	// Check that the row equals 0, 2, 0
	if !reflect.DeepEqual(row, []float64{0, 2, 0}) {
		t.Error("Matrix values are incorrect.")
	}
}

// Test matrix operations.
func TestMatrixOperations(t *testing.T) {
	// Create the matricies.
	a, err := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}})
	b, err := NewMatrixFromSlice([][]float64{[]float64{2, 4, 6}, []float64{8, 10, 12}})
	c, err := NewMatrixFromSlice([][]float64{[]float64{-1, -1, -1}, []float64{-1, -1, -1}})

	// Add a and b.
	a, err = a.Add(b)
	if err != nil {
		t.Error(err.Error())
		return
	}

	// Negate c and subtract it.
	a, err = a.Sub(c.Neg())
	if err != nil {
		t.Error(err.Error())
		return
	}

	// Compare it to another matrix.
	d, err := NewMatrixFromSlice([][]float64{[]float64{2, 5, 8}, []float64{11, 14, 17}})
	if !a.Equals(d) {
		t.Error("Matrix values are incorrect.")
	}
}

// Test matrix scalar operations.
func TestMatrixScalarOperations(t *testing.T) {
	// Create the matrix.
	m, _ := NewMatrix(3, 3)

	// Add a scalar.
	m = m.AddScalar(10)

	// Multiply by 1/3.
	m = m.MulScalar(1/3)

	// Square the matrix.
	m = m.PowScalar(2)

	// Compare a value of the matrix to a 10/3^2.
	if x, _ := m.Get(0, 0); x - math.Pow(10/3, 2) > 0.1 {
		t.Error("Matrix values are incorrect.")
	}
}

// Test matrix dot product.
func TestMatrixDotProduct(t *testing.T) {
	// Create the matricies.
	a, err := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}})
        b, err := NewMatrixFromSlice([][]float64{[]float64{1, 0, 0}, []float64{0, 1, 0}, []float64{0, 0, 1}})

	// Calculate the dot product.
	d, err := a.Dot(b)
	if err != nil {
		t.Error(err.Error())
		return
	}

	// Compare the new matrix to a.
	if !d.Equals(a) {
		t.Error("Matrix values are incorrect.")
	}
}

// Test matrix transpose.
func TestMatrixTranspose(t *testing.T) {
	// Create the matrix.
	a, _ := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}})

	// Get the transpose.
	tr := a.T()

	// Check the dimensions and the values.
	if tr.Rows != 3 || tr.Cols != 2 {
		t.Error("Matrix dimensions are incorrect.")
		return
	}
	if x, _ := tr.Get(2, 1); x != 6 {
		t.Error("Matrix values are incorrect.")
	}
}

// Test matrix sum.
func TestMatrixSum(t *testing.T) {
	// Create the matrix.
	a, _ := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}})

	// Calculate the sums over both axes.
	ax0 := a.Sum(0)
	ax1 := a.Sum(1)

	// Check the sums.
	if sum, _ := NewMatrixFromSlice([][]float64{[]float64{5, 7, 9}}); !ax0.Equals(sum) {
		t.Error("Matrix values are incorrect.")
	}
	if sum, _ := NewMatrixFromSlice([][]float64{[]float64{6, 15}}); !ax1.Equals(sum) {
                t.Error("Matrix values are incorrect.")
        }
}
