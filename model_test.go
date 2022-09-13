// model_test.go
// Tests basic model functionality.

package nn

import (
	"testing"
	"time"
	"math/rand"
)


// Test the model object.
func TestModel(t *testing.T) {
	// Create a new model object.
	m := NewModel()

	// Add some layers.
	l1, _ := NewLayer(3, 5)
	l2, _ := NewLayer(5, 5)
	l3, _ := NewLinearLayer(5, 1)
	m.AddLayer(&l1)
	m.AddLayer(&l2)
	m.AddLayer(&l3)

	// Finalize the network.
	loss, _ := NewMeanSquaredLoss(1)
	optimizer, _ := NewAdamOptimizer(0.01, 0, 1e-7, 0.9, 0.999)
	m.Finalize(&loss, &optimizer, RegressionAccuracyType, 0.01)
	m.InitLayers()

	// Run a forward and backward pass.
	X, _ := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}})
	Y, _ := NewMatrixFromSlice([][]float64{[]float64{0.5}})
	out, err := m.Forward(X)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	_, err = m.Backward(out, Y)
	if err != nil {
                t.Errorf(err.Error())
                return
        }

	// Predict values.
	_, err = m.Predict(X)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
}


// Test training the model object.
func TestTrainModel(t *testing.T) {
	// Init the logging.
	err := InitLogger(true, true, "log.log")
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	// Create the layers.
        l1, _ := NewLayer(1, 8)
        l2, _ := NewLayer(8, 8)
        l3, _ := NewSoftmaxLayer(8, 2)

        // Create the loss.
        loss, _ := NewCrossEntropyLoss(2)

        // Create an optimizer.
	optimizer, _ := NewAdamOptimizer(0.001, 0, 1e-7, 0.9, 0.999)

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

	// Create the model.
	m := NewModel()
	m.AddLayer(&l1)
	m.AddLayer(&l2)
	m.AddLayer(&l3)
	m.Finalize(&loss, &optimizer, CategoricalAccuracyType, 0)
        m.InitLayers()

	// Fit the model.
	err = m.Fit(x, y, 500, 100, Matrix{}, Matrix{}, 100)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
}
