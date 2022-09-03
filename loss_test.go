// loss_test.go
// Testing for loss.go.

package nn

import (
	"testing"
)

// Test MSE loss function.
func TestMSELoss(t *testing.T) {
	// Create the MSE loss object.
	loss, _ := NewMeanSquaredLoss(5)

	// Create the matricies.
	yhat, _ := NewMatrixFromSlice([][]float64{[]float64{1.1, 1.9, 3.2, 4.01, 4.97}})
	y, _ := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3, 4, 5}})

	// Calculate the forward pass.
	j, err := loss.Forward(yhat, y)
	if err != nil {
		t.Error(err.Error())
		return
	}
	t.Logf("%f", j)

	// Calculate the backward pass.
	dInputs, err := loss.Backward(yhat, y)
	if err != nil {
		t.Error(err.Error())
		return
	}
	t.Logf("%v", dInputs)
}
