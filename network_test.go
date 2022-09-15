// network_test.go
// Testing for network.go.

package nn

import (
        "testing"
)


// Test basic neural network layer forward and backward passes.
func TestHiddenLayer(t *testing.T) {
	// Create the layer.
	l, _ := NewLayer(3, 5)

	// Initialize the layer.
	l.Init()

	// Perform the forward pass with three test cases.
	x, err := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}, []float64{7, 8, 9}})
	if err != nil {
		t.Error(err.Error())
		return
	}
	out, err := l.Forward(x)
	if err != nil {
		t.Error(err.Error())
		return
	}
	t.Logf("%v", out)

	// Perform the backward pass.
	dValues, err := NewMatrixFromSlice([][]float64{[]float64{1, 1, 1, 1, 1}, []float64{2, 2, 2, 2, 2}, []float64{3, 3, 3, 3, 3}})
	dWeights, dBiases, dInputs, err := l.Backward(x, dValues)
	if err != nil {
		t.Error(err.Error())
		return
	}
	t.Logf("%v, %v, %v", dWeights, dBiases, dInputs)
}

// Test linear neural network hidden layer forward and backward passes.
func TestLinearLayer(t *testing.T) {
	// Create the layer.
        l, _ := NewLinearLayer(3, 5)

        // Initialize the layer.
        l.Init()

        // Perform the forward pass.
        x, err := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}})
        if err != nil {
                t.Error(err.Error())
                return
        }
        out, err := l.Forward(x)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", out)

        // Perform the backward pass.
        dValues, err := NewMatrixFromSlice([][]float64{[]float64{0.3, 0.1, 0.2, 0.2, 0.2}})
        dWeights, dBiases, dInputs, err := l.Backward(x, dValues)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v, %v, %v", dWeights, dBiases, dInputs)
}

// Test sigmoid neural network hidden layer forward and backward passes.
func TestSigmoidLayer(t *testing.T) {
        // Create the layer.
        l, _ := NewSigmoidLayer(3, 5)

        // Initialize the layer.
        l.Init()

        // Perform the forward pass.
        x, err := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}})
        if err != nil {
                t.Error(err.Error())
                return
        }
        out, err := l.Forward(x)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", out)

        // Perform the backward pass.
        dValues, err := NewMatrixFromSlice([][]float64{[]float64{0.3, 0.1, 0.2, 0.2, 0.2}})
        dWeights, dBiases, dInputs, err := l.Backward(x, dValues)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v, %v, %v", dWeights, dBiases, dInputs)
}

// Test leaky neural network hidden layer forward and backward passes.
func TestLeakyLayer(t *testing.T) {
        // Create the layer.
        l, _ := NewLeakyLayer(3, 5, 0.1)

        // Initialize the layer.
        l.Init()

        // Perform the forward pass.
        x, err := NewMatrixFromSlice([][]float64{[]float64{1, -1, 2}})
        if err != nil {
                t.Error(err.Error())
                return
        }
        out, err := l.Forward(x)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", out)

        // Perform the backward pass.
        dValues, err := NewMatrixFromSlice([][]float64{[]float64{0.3, 0.1, 0.2, 0.2, 0.2}})
        dWeights, dBiases, dInputs, err := l.Backward(x, dValues)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v, %v, %v", dWeights, dBiases, dInputs)
}

// Test softmax neural network hidden layer forward and backward passes.
func TestSoftmaxLayer(t *testing.T) {
        // Create the layer.
        l, _ := NewSoftmaxLayer(3, 5)

        // Initialize the layer.
        l.Init()

        // Perform the forward pass.
        x, err := NewMatrixFromSlice([][]float64{[]float64{1, -1, 2}})
        if err != nil {
                t.Error(err.Error())
                return
        }
        out, err := l.Forward(x)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", out)

        // Perform the backward pass.
        dValues, err := NewMatrixFromSlice([][]float64{[]float64{0.3, 0.1, 0.2, 0.2, 0.2}})
        dWeights, dBiases, dInputs, err := l.Backward(x, dValues)
        if err != nil {
                t.Error(err.Error())
		return
        }
        t.Logf("%v, %v, %v", dWeights, dBiases, dInputs)
}

// Test softmax neural network hidden layer with categorical cross-entropy loss forward and backward passes.
func TestSoftmaxCrossEntropyLayer(t *testing.T) {
        // Create the layer and the loss.
        l, _ := NewSoftmaxLayer(3, 5)
	jl, _ := NewCrossEntropyLoss(5)

        // Initialize the layer.
        l.Init()

        // Perform the forward pass.
        x, err := NewMatrixFromSlice([][]float64{[]float64{1, -1, 2}})
        if err != nil {
                t.Error(err.Error())
                return
        }
	y, err := NewMatrixFromSlice([][]float64{[]float64{0, 0, 0, 1, 0}})
        if err != nil {
                t.Error(err.Error())
                return
        }
        out, err := l.Forward(x)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", out)

	// Calculate the loss.
	loss, err := jl.Forward(out, y)
	if err != nil {
		t.Error(err.Error())
		return
	}
	t.Logf("%v", loss)

        // Perform the backward pass.
        dWeights, dBiases, dInputs, err := l.BackwardCrossEntropy(x, y, out)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v, %v, %v", dWeights, dBiases, dInputs)
}

// Test dropout neural network hidden layer forward and backward passes.
func TestDropoutLayer(t *testing.T) {
        // Create the layer.
        l, _ := NewDropoutLayer(3, 5, 0.1)

        // Initialize the layer.
        l.Init()

        // Perform the forward pass.
        x, err := NewMatrixFromSlice([][]float64{[]float64{1, -1, 2}})
        if err != nil {
                t.Error(err.Error())
                return
        }
        out, err := l.Forward(x)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v", out)

        // Perform the backward pass.
        dValues, err := NewMatrixFromSlice([][]float64{[]float64{0.3, 0.1, 0.2, 0.2, 0.2}})
        dWeights, dBiases, dInputs, err := l.Backward(x, dValues)
        if err != nil {
                t.Error(err.Error())
                return
        }
        t.Logf("%v, %v, %v", dWeights, dBiases, dInputs)
}
