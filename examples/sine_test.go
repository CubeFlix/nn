// sine_test.go
// Sine wave full network testing for nn.
// Note: requires some fine-tuning

package nn

import (
	"testing"
	"math"
	"time"
	"math/rand"
)

func ptr(val Matrix, _ error) *Matrix {
    return &val
}

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
	learningRate := 0.00001

	// Num of epochs.
	epochs := 1000

	// Get start time.
	starttime := time.Now()
	t.Logf(starttime.String())

	// Create the input data.
	samples := 250
	X, _ := NewMatrix(samples, 1)
	Y, _ := NewMatrix(samples, 1)
	for i := 0; i < samples; i++ {
		_ = X.Set(i, 0, float64(i)/float64(samples))
		_ = Y.Set(i, 0, (math.Sin(float64(i)/float64(samples))))
	}
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(samples, func(i, j int) { X.M[i], X.M[j] = X.M[j], X.M[i]; Y.M[i], Y.M[j] = Y.M[j], Y.M[i] })

	// Get time.
	t.Logf(time.Now().String())

	gradientsWeightsL1 := Matrix{}
	gradientsBiasesL1 := Matrix{}
	gradientsWeightsL2 := Matrix{}
	gradientsBiasesL2 := Matrix{}
	gradientsWeightsL3 := Matrix{}
        gradientsBiasesL3 := Matrix{}
	for i := 0; i < epochs; i++ {
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

	if i == epochs - 1 {
                t.Logf("%v\n%v\n%v %v", l.Weights, out3, Y, j)
        }
	if i % 100 == 0 {
                t.Logf("%f", j)
		// Calculate accuracy.
		s := 0
		for n := 0; n < samples; n++ {
			if math.Abs(out3.M[n][0] - Y.M[n][0]) < 0.01{
				s += 1
			}
		}
		t.Logf("%f", float64(s)/float64(samples))
        }

	// Backward passes.
	// Loss backward pass.
	dValues, err := loss.Backward(out3, Y)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	// Scale the gradients down.
	// dValues = dValues.MulScalar(0.005)

	// Layer 3 backward pass.
	dWeights, dBiases, dValues, err := l3.Backward(out2, dValues)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	gradientsWeightsL3 = dWeights
	gradientsBiasesL3 = dBiases

	// Scale the gradients down.
	// dValues = dValues.MulScalar(0.001)

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
	l.Weights = ptr(l.Weights.Sub(gradientsWeightsL1.MulScalar(learningRate)))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	l.Biases = ptr(l.Biases.Sub(gradientsBiasesL1.MulScalar(learningRate)))
	if err != nil {
               t.Errorf(err.Error())
               return
        }
	l2.Weights = ptr(l2.Weights.Sub(gradientsWeightsL2.MulScalar(learningRate)))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	l2.Biases = ptr(l2.Biases.Sub(gradientsBiasesL2.MulScalar(learningRate)))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	l3.Weights = ptr(l3.Weights.Sub(gradientsWeightsL3.MulScalar(learningRate)))
	if err != nil {
                t.Errorf(err.Error())
                return
        }
	l3.Biases = ptr(l3.Biases.Sub(gradientsBiasesL3.MulScalar(learningRate)))
	if err != nil {
                t.Errorf(err.Error())
                return
        }

	} // Epoch loop
}
