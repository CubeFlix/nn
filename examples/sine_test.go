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

	// Create the optimizer objects.
	opt1, _ := NewAdamOptimizer(0.001, 0, 1e-7, 0.9, 0.999)
        opt2, _ := NewAdamOptimizer(0.001, 0, 1e-7, 0.9, 0.999)
        opt3, _ := NewAdamOptimizer(0.001, 0, 1e-7, 0.9, 0.999)

	// Num of epochs.
	epochs := 2

	// Get start time.
	starttime := time.Now()
	t.Logf(starttime.String())

	// Create the input data.
	samples := 250
	x, _ := NewMatrix(samples, 1)
	y, _ := NewMatrix(samples, 1)
	for i := 0; i < samples; i++ {
		_ = x.Set(i, 0, float64(i)/float64(samples))
		_ = y.Set(i, 0, (math.Sin(float64(i)/float64(samples))))
	}
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(samples, func(i, j int) { x.M[i], x.M[j] = x.M[j], x.M[i]; y.M[i], y.M[j] = y.M[j], y.M[i] })

	// Get time.
	t.Logf(time.Now().String())

	gradientsWeightsL1 := Matrix{}
	gradientsBiasesL1 := Matrix{}
	gradientsWeightsL2 := Matrix{}
	gradientsBiasesL2 := Matrix{}
	gradientsWeightsL3 := Matrix{}
        gradientsBiasesL3 := Matrix{}
	for i := 0; i < epochs; i++ {
	for start := 0; start < samples; start+=25 {
        X, _ := NewMatrixFromSlice(x.M[start:start+25])
	Y, _ := NewMatrixFromSlice(y.M[start:start+25])
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
		for n := 0; n < 24; n++ {
			if math.Abs(out3.M[n][0] - Y.M[n][0]) < 0.01{
				s += 1
			}
		}
		t.Logf("%f", float64(s)/float64(24))
        }

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

	// Update the the weights and biases based on the gradients.
	opt3.Update(l3.Weights, l3.Biases, gradientsWeightsL3, gradientsBiasesL3)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	opt2.Update(l2.Weights, l2.Biases, gradientsWeightsL2, gradientsBiasesL2)
        if err != nil {
                t.Errorf(err.Error())
                return
        }
	opt1.Update(l.Weights, l.Biases, gradientsWeightsL1, gradientsBiasesL1)
        if err != nil {
                t.Errorf(err.Error())
                return
        }
	} // Batch loop
	} // Epoch loop
}
