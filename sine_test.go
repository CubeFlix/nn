// sine_test.go
// Sine wave full network testing for nn.

package nn

import (
	"testing"
	"math"
	"time"
)

func TestSine(t *testing.T) {
	// Create the layers.
	l, _ := NewLayer(1, 64)
	l2, _ := NewLayer(64, 64)
	l3, _ := NewLinearLayer(64, 1)
	loss, _ := NewMeanSquaredLoss(1)
	l.Init()
	l2.Init()
	l3.Init()
	t.Logf("%v", l.Weights)

	// Learning rate.
	learningRate := 0.000001

	// Get start time.
	starttime := time.Now()
	t.Logf(starttime.String())

	// Create the input data.
	X, _ := NewMatrix(1000, 1)
	Y, _ := NewMatrix(1000, 1)
	for i := 0; i < 1000; i++ {
		_ = X.Set(i, 0, float64(i)/1000)
		_ = Y.Set(i, 0, math.Sin(float64(i)/1000))
	}

	// Get time.
	t.Logf(time.Now().String())

	for i := 0; i < 1000; i++ {
	gradientsWeightsL1 := Matrix{}
	gradientsBiasesL1 := Matrix{}
	gradientsWeightsL2 := Matrix{}
	gradientsBiasesL2 := Matrix{}
	gradientsWeightsL3 := Matrix{}
        gradientsBiasesL3 := Matrix{}
	// Forward and backward passes.

	// Forward layer 1.
	out1, err := l.Forward(X)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	// Forward layer 2.
	out2, err := l2.Forward(out1)
	if err != nil {
		t.Errorf(err.Error())
                return
	}

	// Forward layer 3.
	out3, err := l3.Forward(out2)
        if err != nil {
                t.Errorf(err.Error())
                return
        }

	// Loss forward pass.
	j, err := loss.Forward(out3, Y)
        if err != nil {
                t.Errorf(err.Error())
                return
        }

	// Backward passes.
	// Loss backward pass.
	dValues, err := loss.Backward(out3, Y)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	if i % 100 == 0 {
		t.Logf("%f", j)
		// t.Logf("%v", out2)
	}

	// Layer 3 backward pass.
	dWeights, dBiases, dValues, err := l3.Backward(out2, dValues)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	gradientsWeightsL3 = dWeights
	gradientsBiasesL3 = dBiases

	// Layer 2 backward pass.
	dWeights, dBiases, dValues, err = l2.Backward(out1, dValues)
        if err != nil {
                t.Errorf(err.Error())
                return
        }
        gradientsWeightsL2 = dWeights
        gradientsBiasesL2 = dBiases

	// Layer 1 backward pass.
	dWeights, dBiases, dValues, err = l.Backward(X, dValues)
        if err != nil {
                t.Errorf(err.Error())
                return
        }
        gradientsWeightsL1 = dWeights
        gradientsBiasesL1 = dBiases

	// Update the the weights and biases based on the gradients.
	lWeights, err := l.Weights.Sub(gradientsWeightsL1.MulScalar(learningRate))
	l.Weights = &lWeights
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	lBiases, err := l.Biases.Sub(gradientsBiasesL1.MulScalar(learningRate))
	l.Biases = &lBiases
	if err != nil {
               t.Errorf(err.Error())
               return
        }
	l2Weights, err := l2.Weights.Sub(gradientsWeightsL2.MulScalar(learningRate))
	l2.Weights = &l2Weights
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	l2Biases, err := l2.Biases.Sub(gradientsBiasesL2.MulScalar(learningRate))
	l2.Biases = &l2Biases
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	l3Weights, err := l3.Weights.Sub(gradientsWeightsL3.MulScalar(learningRate))
        l3.Weights = &l3Weights
	if err != nil {
                t.Errorf(err.Error())
                return
        }
	l3Biases, err := l3.Biases.Sub(gradientsBiasesL3.MulScalar(learningRate))
	l3.Biases = &l3Biases
	if err != nil {
                t.Errorf(err.Error())
                return
        }

	if i == 999 {
		t.Logf("%v\n%v\n%v %v", X, out3, Y, j)
	}

	} // Epoch loop
}
