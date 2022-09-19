// spiral_test.go
// Spiral dataset testing.

package nn

import (
	"testing"
	"math/rand"
	"math"
)


// Spiral data function taken from https://github.com/Sentdex/NNfSiX/blob/master/Go/p005-ReLU-Activation.go
func NewSpiralData(numberOfPoints, numberOfClasses int) (Matrix, Matrix) {
	X, _ := NewMatrix(numberOfPoints*numberOfClasses, 2)
	Y, _ := NewMatrix(numberOfPoints*numberOfClasses, numberOfClasses)

	for c := 0; c < numberOfClasses; c++ {
		radius := linspace(0, 1, numberOfPoints)
		t := linspace(float64(c*4), float64((c+1)*4), numberOfPoints)
		for i := range t {
			t[i] += 0.2 * rand.NormFloat64()
		}
		for i := 0; i < numberOfPoints; i++ {
			X.M[c*numberOfPoints+i][0] = radius[i]*math.Sin(t[i]*2.5)
			X.M[c*numberOfPoints+i][1] = radius[i]*math.Cos(t[i]*2.5)
			Y.M[c*numberOfPoints+i][c] = 1
		}
	}

	return X, Y
}

func linspace(start, end float64, num int) []float64 {
	result := make([]float64, num)
	step := (end - start) / float64(num-1)
	for i := range result {
		result[i] = start + float64(i)*step
	}
	return result
}

func TestSpiral(t *testing.T) {
	rand.Seed(0)
	// Init the logging.
        err := InitLogger(true, true, "spiral.log")
        if err != nil {
                t.Errorf(err.Error())
                return
        }

	// Create the spiral data.
	X, Y := NewSpiralData(100, 2)

	// Shuffle the dataset.
	X, Y = ShuffleDataset(X, Y)

	// Create the layers.
	l1, _ := NewLayer(2, 64)
	l3, _ := NewSoftmaxLayer(64, 2)

	// Create the loss and optimizer.
	loss, _ := NewCrossEntropyLoss(2)
	optimizer, _ := NewAdamOptimizer(0.01, 0, 1e-7, 0.9, 0.999)

	// Create the model and finish it.
	model := NewModel()
	_ = model.AddLayer(&l1)
	_ = model.AddLayer(&l3)
	model.InitLayers()
	model.Finalize(&loss, &optimizer, CategoricalAccuracyType, 0)

	// Fit the model.
	model.Fit(X, Y, 10000, 0, Matrix{}, Matrix{}, 100)
}
