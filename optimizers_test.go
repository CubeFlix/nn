// optimizers_test.go
// Tests for optimizers.

package nn

import (
	"testing"
	"time"
	"math/rand"
)

func TestSGDOptimizer(t *testing.T) {
	// Create the optimizers.
	opt1, _ := NewSGDOptimizer(5e-2, 0, 0)
	opt2, _ := NewSGDOptimizer(5e-2, 0, 0)
	opt3, _ := NewSGDOptimizer(5e-2, 0, 0)

	// Create the layers
	l1, _ := NewLayer(1, 8)
	l1.Init()
	l2, _ := NewLayer(8, 8)
	l2.Init()
	l3, _ := NewSoftmaxLayer(8, 2)
        l3.Init()

	// Create the loss.
	loss, _ := NewCrossEntropyLoss(2)

	// Create the data.
	samples := 100
	x, _ := NewMatrix(samples, 1)
	y, _ := NewMatrix(samples, 2)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < samples; i++ {
		x.M[i][0] = rand.Float64()
	}
	for i := 0; i < samples; i++ {
		if x.M[i][0] < 0.5 {
			y.M[i][0] = 1
		} else {
			y.M[i][1] = 1
		}
	}

	// Begin training.
	epochs := 500
	for i := 0; i < epochs; i++ {
		// start := rand.Intn(100-25)
		X, _ := NewMatrixFromSlice(x.M[0:100])
		Y, _ := NewMatrixFromSlice(y.M[0:100])
		// Forward pass.
		out1, err := l1.Forward(X)
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		out2, err := l2.Forward(out1)
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		out3, err := l3.Forward(out2)
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		j, err := loss.Forward(out3, Y)
		if err != nil {
			t.Errorf(err.Error())
			return
		}

		// Backward pass.
		dWeights3, dBiases3, dValues, err := l3.BackwardCrossEntropy(out2, Y, out3)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
		dWeights2, dBiases2, dValues, err := l2.Backward(out1, dValues)
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		dWeights1, dBiases1, dValues, err := l1.Backward(X, dValues)
		if err != nil {
                        t.Errorf(err.Error())
                        return
                }

		// Update the values with the optimizers.
		err = opt3.Update(l3.Weights, l3.Biases, dWeights3, dBiases3)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
		err = opt2.Update(l2.Weights, l2.Biases, dWeights2, dBiases2)
		if err != nil {
                        t.Errorf(err.Error())
                        return
                }
		err = opt1.Update(l1.Weights, l1.Biases, dWeights1, dBiases1)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }

		if i % 100 == 0 {
			avgJ := float64(0)
			for n := 0; n < samples; n++ {
				avgJ += j.M[n][0] / float64(samples)
			}
			t.Logf("%v", avgJ)
		}
		if i == epochs - 1 {
			t.Logf("%v", out3)
		}
	}
}

func TestAdamOptimizer(t *testing.T) {
        // Create the optimizers.
        opt1, _ := NewAdamOptimizer(0.001, 0, 1e-7, 0.9, 0.999)
        opt2, _ := NewAdamOptimizer(0.001, 0, 1e-7, 0.9, 0.999)
        opt3, _ := NewAdamOptimizer(0.001, 0, 1e-7, 0.9, 0.999)

        // Create the layers
        l1, _ := NewLayer(1, 8)
        l1.Init()
        l2, _ := NewLayer(8, 8)
        l2.Init()
        l3, _ := NewSoftmaxLayer(8, 2)
        l3.Init()

        // Create the loss.
        loss, _ := NewCrossEntropyLoss(2)

        // Create the data.
        samples := 100
        x, _ := NewMatrix(samples, 1)
        y, _ := NewMatrix(samples, 2)
        rand.Seed(time.Now().UnixNano())
        for i := 0; i < samples; i++ {
                x.M[i][0] = rand.Float64()
        }
        for i := 0; i < samples; i++ {
                if x.M[i][0] < 0.5 {
                        y.M[i][0] = 1
                } else {
                        y.M[i][1] = 1
                }
        }

        // Begin training.
        epochs := 500
        for i := 0; i < epochs; i++ {
                // start := rand.Intn(100-25)
                X, _ := NewMatrixFromSlice(x.M[0:100])
                Y, _ := NewMatrixFromSlice(y.M[0:100])
                // Forward pass.
                out1, err := l1.Forward(X)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
                out2, err := l2.Forward(out1)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
                out3, err := l3.Forward(out2)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
                j, err := loss.Forward(out3, Y)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }

                // Backward pass.
                dWeights3, dBiases3, dValues, err := l3.BackwardCrossEntropy(out2, Y, out3)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
		dWeights2, dBiases2, dValues, err := l2.Backward(out1, dValues)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
		dWeights1, dBiases1, dValues, err := l1.Backward(X, dValues)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }

                // Update the values with the optimizers.
                err = opt3.Update(l3.Weights, l3.Biases, dWeights3, dBiases3)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
                err = opt2.Update(l2.Weights, l2.Biases, dWeights2, dBiases2)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }
                err = opt1.Update(l1.Weights, l1.Biases, dWeights1, dBiases1)
                if err != nil {
                        t.Errorf(err.Error())
                        return
                }

		if i % 100 == 0 {
			avgJ := float64(0)
			for n := 0; n < samples; n++ {
				avgJ += j.M[n][0] / float64(samples)
			}
			t.Logf("%v", avgJ)
		}

                if i == epochs - 1 {
                        t.Logf("%v", out3)
                }
        }
}
