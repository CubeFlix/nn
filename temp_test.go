// temp_test.go
// Temporary testing for nn.

package nn

import (
	"testing"
	"math"
	"time"
)

func TestTemp(t *testing.T) {
	// Create the layers.
	l, _ := NewLayer(1, 64)
	l2, _ := NewLayer(64, 64)
	l3, _ := NewLinearLayer(64, 1)
	loss, _ := NewMeanSquaredLoss(1)
	l.Init()
	l2.Init()
	l3.Init()
	t.Logf("%v", l.Weights)

	// Get start time.
	starttime := time.Now()
	t.Logf(starttime.String())

	// Create the input data.
	X, err := NewMatrix(100, 1)
	if err != nil {
		t.Errorf(err)
		return
	}
	Y, err := NewMatrix(100, 1)
	if err != nil {
		t.Errorf(err)
		return
	}
	for i := 0; i < 100; i++ {
		_ := X.Set(i, 1, float64(i)/50)
		_ := Y.Set(i, 1, math.Sin(float64(i)/50))
	}

	// Get time.
	t.Logf(time.Now().String())

	for i := 0; i < 100; i++ {
	gradientsWeightsL1, _ := Matrix{}
	gradientsBiasesL1, _ := Matrix{}
	gradientsWeightsL2, _ := Matrix{}
	gradientsBiasesL2, _ := Matrix{}
	gradientsWeightsL3, _ := Matrix{}
        gradientsBiasesL3, _ := Matrix{}
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
	j, err := loss.Forward(out3, Y[i])
        if err != nil {
                t.Errorf(err.Error())
                return
        }
	t.Logf("%f %f", out3, j)

	// Backward passes.
	// Loss backward pass.
	dValues, err := loss.Backward(out3, Y)
	if err != nil {
		t.Errorf(err.Error())
		return
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
	}

	// Define learning rate.
	learningRate := 0.001

	// Update the the weights and biases based on the gradients.
	for i := 0; i < 64; i++ {
		lWeights := gradientsWeights[i*3+2]
		lBiases := gradientsBiases[i*3+2]
		lWeights, err := l.Weights.Sub(lWeights.MulScalar(learningRate))
		l.Weights = &lWeights
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		lBiases, err = l.Biases.Sub(lBiases.MulScalar(learningRate))
		l.Biases = &lBiases
		if err != nil {
                        t.Errorf(err.Error())
                        return
                }
		l2Weights := gradientsWeights[i*3+1]
		l2Biases := gradientsBiases[i*3+1]
		l2Weights, err = l2.Weights.Sub(l2Weights.MulScalar(learningRate))
		l2.Weights = &l2Weights
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		l2Biases, err = l2.Biases.Sub(l2Biases.MulScalar(learningRate))
		l2.Biases = &l2Biases
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		l3Weights := gradientsWeights[i*3]
                l3Biases := gradientsBiases[i*3]
		l3Weights, err = l3.Weights.Sub(l3Weights.MulScalar(learningRate))
                l3.Weights = &l3Weights
		if err != nil {
                        t.Errorf(err.Error())
                        return
                }
		l3Biases, err = l3.Biases.Sub(l3Biases.MulScalar(learningRate))
		l3.Biases = &l3Biases
		if err != nil {
                        t.Errorf(err.Error())
                        return
                }
	}
	}
}
