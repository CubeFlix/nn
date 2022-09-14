// model.go
// Neural network model code.

package nn

import (
	"errors"
	"math"
)


// Loss type type definition.
type LossType int8

// Loss types and codes.
const (
        MeanSquaredLossType        LossType = 0
        MeanAbsoluteLossType                = 1
        CrossEntropyLossType                = 2
        BinaryCrossEntropyLossType          = 3
)


// Optimizer type type definition.
type OptimizerType int8

// Optimizer types and codes.
const (
        SGDOptimizerType  OptimizerType = 0
        AdamOptimizerType               = 1
)


// Accuracy type type definition.
type AccuracyType int8

// Accuracy types and codes.
const (
        RegressionAccuracyType        AccuracyType = 0
	CategoricalAccuracyType                    = 1
	BinaryCategoricalAccuracyType              = 2
)


// Gradients struct.
type Gradients struct {
	DWeights Matrix
	DBiases  Matrix
}


// Neural network model struct.
type Model struct {
	ModelSize         int
	InputSize         int
	OutputSize        int
	LossType          LossType
	Loss              Loss
	AccuracyType      AccuracyType
	AccuracyPercision float64            // Only applicable for regression.
	OptimizerType     OptimizerType
	OptimizerValues   map[string]float64
	Layers            []Layer
	Optimizers        []Optimizer
}

// Create a new model object.
func NewModel() Model {
	return Model{
		Layers: []Layer{},
	}
}

// Add a layer to the model.
func (m *Model) AddLayer(l Layer) error {
	// Get the layer data.
	_, _, values := l.getValues()

	// If it is the first layer, set the input size.
	if m.ModelSize == 0 {
		m.InputSize = int(values["inputs"])
	} else {
		// If not, check that its input size lines up with the current output size.
		if m.OutputSize != int(values["inputs"]) {
			return errors.New("nn.Model: Layer's input size does not match up with previous layer's input size.")
		}
	}

	// Add the layer.
	m.Layers = append(m.Layers, l)

	// Set the new model size and output size.
	m.ModelSize += 1
	m.OutputSize = int(values["outputs"])

	return nil
}

// Finalize the model with the loss and optimizer data.
func (m *Model) Finalize(loss Loss, optimizer Optimizer, accuracyType AccuracyType, accuracyPercision float64) error {
	// Set the loss.
	values := loss.getValues()
	if m.OutputSize != int(values["size"]) {
		return errors.New("nn.Model: Loss size does not match up with output size.")
	}
	m.LossType = LossType(values["type"])
	m.Loss = loss

	// Set the optimizer.
	values = optimizer.getValues()
	m.OptimizerType = OptimizerType(values["type"])
	delete(values, "type")
	m.OptimizerValues = values

	// Create all the optimizers.
	m.Optimizers = []Optimizer{}
	for i := 0; i < m.ModelSize; i++ {
		o, err := NewOptimizerFromType(m.OptimizerType, m.OptimizerValues)
		if err != nil {
			return err
		}
		m.Optimizers = append(m.Optimizers, o)
	}

	// Set the accuracy.
	m.AccuracyType = accuracyType
	if accuracyPercision < 0 {
		return errors.New("nn.Model: Accuracy percision cannot be less than zero.")
	}
	m.AccuracyPercision = accuracyPercision

	return nil
}

// Initialize all the layers.
func (m *Model) InitLayers() {
	for i := 0; i < m.ModelSize; i++ {
		m.Layers[i].Init()
	}
}

// Forward pass. Returns a list of outputs from each layer, including the inputs.
func (m *Model) Forward(X Matrix) ([]Matrix, error) {
	outputs := []Matrix{X}
	output := X

	// Loop over all the layers and perform their forward pass.
	for i := 0; i < m.ModelSize; i++ {
		out, err := m.Layers[i].Forward(output)
		output = out
		if err != nil {
			return []Matrix{}, err
		}
		outputs = append(outputs, output)
	}

	// Return the output matrix.
	return outputs, nil
}

// Backward pass. Takes in outputs from the forward pass, along with the true values. Returns a list of gradients.
func (m *Model) Backward(outputs []Matrix, Y Matrix) ([]Gradients, error) {
	// Create the list of gradients.
	gradients := []Gradients{}

	// Backward pass over loss.
	var dValues Matrix
	if _, ok := m.Layers[m.ModelSize - 1].(*SoftmaxLayer); ok && m.LossType == CrossEntropyLossType {
		// Use more efficient cross entropy backward pass.
		l, _ := m.Layers[m.ModelSize - 1].(*SoftmaxLayer)
		dWeights, dBiases, dInputs, err := l.BackwardCrossEntropy(outputs[m.ModelSize - 1], Y, outputs[m.ModelSize])
		dValues = dInputs
		if err != nil {
			return []Gradients{}, err
		}
		gradients = append(gradients, Gradients{dWeights, dBiases})
	} else {
		// Standard loss backward pass.
		dInputs, err := m.Loss.Backward(outputs[m.ModelSize], Y)
		dValues = dInputs
		if err != nil {
                        return []Gradients{}, err
                }
	}

	// Loop over the layers and perform their backward pass.
	for i := m.ModelSize - 1; i >= 0; i-- {
		if _, ok := m.Layers[i].(*SoftmaxLayer); ok && m.LossType == CrossEntropyLossType && i == m.ModelSize - 1 {
			continue
		}
		dWeights, dBiases, dInputs, err := m.Layers[i].Backward(outputs[i], dValues)
		dValues = dInputs
		gradients = append(gradients, Gradients{dWeights, dBiases})
		if err != nil {
                        return []Gradients{}, err
                }
	}

	// Return the gradients.
	return gradients, nil
}

// Fit the network. If batchSize is zero, the model will not use batching. If yVal is empty, the model will not use validation. If logEvery is zero, the model will not be verbose.
func (m *Model) Fit(X, Y Matrix, epochs, batchSize int, xVal, yVal Matrix, logEvery int) error {
	// See if we will have to use validation.
	useValidation := (yVal.Rows != 0)

	// Calculate the number of batch steps. If not using batching, the number of steps will be 1 and the size will be number of samples.
	useBatching := (batchSize != 0)
	batchSteps := 1
	if useBatching {
		batchSteps = Y.Rows / batchSize
		if batchSteps * batchSize < Y.Rows {
			batchSteps += 1
		}
	} else {
		batchSize = Y.Rows
	}

	// Main training loop.
	for epoch := 0; epoch < epochs; epoch++ {
		// Batch training loop.
		for batchStep := 0; batchStep < batchSteps; batchStep++ {
			// Get the batch X and Y matricies.
			batchX, _ := NewMatrixFromSlice(X.M[batchStep * batchSize : int(math.Min(float64((batchStep + 1) * batchSize), float64(Y.Rows)))])
			batchY, _ := NewMatrixFromSlice(Y.M[batchStep * batchSize : int(math.Min(float64((batchStep + 1) * batchSize), float64(Y.Rows)))])

			// Perform the forward pass.
			outputs, err := m.Forward(batchX)
			if err != nil {
				ErrorLogger.Printf("Failed to perform forward pass: %s", err.Error())
				return err
			}

			// Perform the backward pass.
			gradients, err := m.Backward(outputs, batchY)
			if err != nil {
                                ErrorLogger.Printf("Failed to perform forward pass: %s", err.Error())
                                return err
                        }

			// Update the weights and biases using the optimizers.
			for layer := 0; layer < m.ModelSize; layer++ {
				weights, biases, _ := m.Layers[layer].getValues()
				err := m.Optimizers[layer].Update(weights, biases, gradients[m.ModelSize - layer - 1].DWeights, gradients[m.ModelSize - layer - 1].DBiases)
				if err != nil {
					ErrorLogger.Printf("Failed to update using optimizer: %s", err.Error())
					return err
				}
			}
		}

		// Log output.
		if logEvery != 0 && epoch % logEvery == 0 {
			// Calculate the loss and accuracy.
			loss, err := m.CalculateLoss(X, Y)
			if err != nil {
				ErrorLogger.Printf("Failed to calculate loss: %s", err.Error())
				return err
			}
			accuracy, err2 := m.CalculateAccuracy(X, Y)
			if err2 != nil {
				ErrorLogger.Printf("Failed to calculate accuracy: %s", err2.Error())
				return err2
			}

			// Log the final output.
			InfoLogger.Printf("Epoch: %d, Loss: %f, Accuracy: %f", epoch, loss, accuracy)
		}

		// Log validation output.
		if useValidation && logEvery != 0 && epoch % logEvery == 0 {
			// Calculate the loss and accuracy.
                        loss, err := m.CalculateLoss(xVal, yVal)
                        if err != nil {
                                ErrorLogger.Printf("Failed to calculate validation loss: %s", err.Error())
                                return err
                        }
                        accuracy, err2 := m.CalculateAccuracy(xVal, yVal)
                        if err2 != nil {
                                ErrorLogger.Printf("Failed to calculate validation accuracy: %s", err2.Error())
                                return err2
                        }

                        // Log the final output.
                        InfoLogger.Printf("Epoch: %d, Validation Loss: %f, Validation Accuracy: %f", epoch, loss, accuracy)
		}
	}

	return nil
}

// Calculate the average loss for the model, given X and Y.
func (m *Model) CalculateLoss(X, Y Matrix) (float64, error) {
	// Perform the forward pass.
	outputs, err := m.Forward(X)
	if err != nil {
		return 0, err
	}

	// Perform the loss pass.
	j, err := m.Loss.Forward(outputs[m.ModelSize], Y)
	if err != nil {
                return 0, err
        }

	return j, nil
}

// Calculate the accuracy of the model.  
func (m *Model) CalculateAccuracy(X, Y Matrix) (float64, error) {
	// Perform the forward pass.
        outputs, err := m.Forward(X)
        if err != nil {
                return 0, err
        }

	// Determine which type of accuracy to calculate.
	if m.AccuracyType == RegressionAccuracyType {
		return RegressionAccuracy(outputs[m.ModelSize], Y, m.AccuracyPercision), nil
	} else if m.AccuracyType == CategoricalAccuracyType {
		return CategoricalAccuracy(outputs[m.ModelSize], Y), nil
	} else if m.AccuracyType == BinaryCategoricalAccuracyType {
		return BinaryCategoricalAccuracy(outputs[m.ModelSize], Y), nil
	}
	return 0, errors.New("nn.Model: Invalid accuracy type.")
}

// Predict the output of the model.
func (m *Model) Predict(X Matrix) (Matrix, error) {
	// Perform the forward pass.
        outputs, err := m.Forward(X)
        if err != nil {
                return Matrix{}, err
        }

        // Determine how to return the final values.
        if m.AccuracyType == RegressionAccuracyType {
                return outputs[m.ModelSize], nil
        } else if m.AccuracyType == CategoricalAccuracyType {
                return RowMax(outputs[m.ModelSize]), nil
        } else if m.AccuracyType == BinaryCategoricalAccuracyType {
		return OutputBinaryValues(outputs[m.ModelSize]), nil
	}
	return Matrix{}, errors.New("nn.Model: Invalid accuracy type.")
}

