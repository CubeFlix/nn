// iris_test.go
// Iris dataset (Fisher) testing.

package nn

import (
	"testing"
	"encoding/csv"
	"strconv"
	"os"
)


func TestIris(t *testing.T) {
	// Init the logging.
        err := InitLogger(true, true, "iris.log")
        if err != nil {
                t.Errorf(err.Error())
                return
        }

	// Load the dataset.
	f, err := os.Open("iris.data")
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	defer f.Close()

	// Load the CSV file.
	csvReader := csv.NewReader(f)
	records, err2 := csvReader.ReadAll()
	if err != nil {
		t.Errorf(err2.Error())
		return
	}

	// Create the matricies.
	X, _ := NewMatrix(len(records), 4)
	Y, _ := NewMatrix(len(records), 3)

	// Loop over the records and set the values.
	for i := 0; i < len(records); i++ {
		X.M[i][0], _ = strconv.ParseFloat(records[i][0], 8)
		X.M[i][1], _ = strconv.ParseFloat(records[i][1], 8)
		X.M[i][2], _ = strconv.ParseFloat(records[i][2], 8)
		X.M[i][3], _ = strconv.ParseFloat(records[i][3], 8)

		if records[i][4] == "Iris-setosa" {
			Y.M[i][0] = 1
		} else if records[i][4] == "Iris-versicolor" {
			Y.M[i][1] = 1
		} else {
			Y.M[i][2] = 1
		}
	}

	// Shuffle the dataset.
	X, Y = ShuffleDataset(X, Y)

	// Create the layers.
	l1, _ := NewLayer(4, 16)
	l2, _ := NewLayer(16, 16)
	l3, _ := NewSoftmaxLayer(16, 3)

	// Create the loss and optimizer.
	loss, _ := NewCrossEntropyLoss(3)
	optimizer, _ := NewAdamOptimizer(0.01, 0, 1e-7, 0.9, 0.999)

	// Create the model and finish it.
	model := NewModel()
	_ = model.AddLayer(&l1)
	_ = model.AddLayer(&l2)
	_ = model.AddLayer(&l3)
	model.InitLayers()
	model.Finalize(&loss, &optimizer, CategoricalAccuracyType, 0)

	// Fit the model.
	model.Fit(X, Y, 500, 0, Matrix{}, Matrix{}, 100)
}
