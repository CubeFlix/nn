// model_data_test.go
// Model data test.

package nn

import (
        "testing"
        "bytes"
)


func TestModelData(t *testing.T) {
	// Init the logging.
        err := InitLogger(true, true, "log.log")
        if err != nil {
                t.Errorf(err.Error())
                return
        }

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

	// Export the model to a buffer.
	data := NewSavedModelData(m)
        var buf = new(bytes.Buffer)
        data.Serialize(buf)

	// Load the model.
        model, err := LoadModel(buf)
        if err != nil {
                t.Errorf(err.Error())
                return
        }

	// Get the loss.
	X, _ := NewMatrixFromSlice([][]float64{[]float64{1, 2, 3}})
	Y, _ := NewMatrixFromSlice([][]float64{[]float64{0.5}})
	j, err := model.CalculateLoss(X, Y)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	t.Logf("%f", j)
}
