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

// Test absolute loss function.
func TestAbsLoss(t *testing.T) {
        // Create the absolute loss object.
        loss, _ := NewMeanAbsoluteLoss(5)

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

// Test cross-entropy loss function.
func TestCrossEntropyLoss(t *testing.T) {
        // Create the cross-entropy loss object.
        loss, _ := NewCrossEntropyLoss(5)

        // Create the matricies.
        yhat, _ := NewMatrixFromSlice([][]float64{[]float64{0.03, 0.03, 0.02, 0.02, 0.9}, []float64{0.02, 0.8, 0.08, 0.5, 0.5}})
        y, _ := NewMatrixFromSlice([][]float64{[]float64{0, 0, 0, 0, 1}, []float64{0, 1, 0, 0, 0}})

        // Calculate the forward pass.
        j, err := loss.Forward(yhat, y)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", j)

        // Calculate the backward pass.
        dInputs, err := loss.Backward(yhat, y)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", dInputs)
}

// Test binary cross-entropy loss function.
func TestBinaryCrossEntropyLoss(t *testing.T) {
        // Create the binary cross-entropy loss object.
        loss, _ := NewBinaryCrossEntropyLoss()

        // Create the matricies.
        yhat, _ := NewMatrixFromSlice([][]float64{[]float64{0.9}, []float64{0.3}})
        y, _ := NewMatrixFromSlice([][]float64{[]float64{0}, []float64{2}})

        // Calculate the forward pass.
        j, err := loss.Forward(yhat, y)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", j)

        // Calculate the backward pass.
        dInputs, err := loss.Backward(yhat, y)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", dInputs)
}
