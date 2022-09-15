// network.go
// Main neural network layer code.

package nn

import (
	"fmt"
	"time"
	"math"
	"math/rand"
	"errors"
)


// Layer interface.
type Layer interface {
	Init()
	Forward(Matrix)                               (Matrix, error)
	Backward(Matrix, Matrix)                      (Matrix, Matrix, Matrix, error)
	getValues()                                   (*Matrix, *Matrix, map[string]float64)
	setValues(Matrix, Matrix, map[string]float64)
}


// Invalid layer dimensions error.
func invalidLayerDimensionsError(inputSize, outputSize int) error {
	return errors.New(fmt.Sprintf("nn.Layer: Invalid layer dimensions: %d, %d", inputSize, outputSize))
}


// Main hidden neural network layer struct.
type HiddenLayer struct {
	InputSize  int
	OutputSize int
	Weights    *Matrix
	Biases     *Matrix
	reluInputs Matrix
}

// Create a new hidden layer.
func NewLayer(inputSize, outputSize int) (HiddenLayer, error) {
	// Check that the input and output sizes are valid.
	if inputSize < 1 || outputSize < 1 {
		return HiddenLayer{}, invalidLayerDimensionsError(inputSize, outputSize)
	}

	// Create the new matricies.
	weights, _ := NewMatrix(inputSize, outputSize)
	biases, _ := NewMatrix(1, outputSize)

	// Create and return the new hidden layer.
	return HiddenLayer{
		InputSize:  inputSize,
		OutputSize: outputSize,
		Weights:    &weights,
		Biases:     &biases,
	}, nil
}

// Get the values for the layer.
func (l *HiddenLayer) getValues() (*Matrix, *Matrix, map[string]float64) {
	values := map[string]float64{"inputs": float64(l.InputSize), "outputs": float64(l.OutputSize), "type": float64(HiddenLayerType)}
	return l.Weights, l.Biases, values
}

// Set the values for the layer.
func (l *HiddenLayer) setValues(weights, biases Matrix, values map[string]float64) {
	l.InputSize = int(values["inputs"])
	l.OutputSize = int(values["outputs"])
	l.Weights = &weights
	l.Biases = &biases
}

// Initialize the hidden layer values.
func (l *HiddenLayer) Init() {
	// Using He weight initialization. Calculate the std for the weights based on the number of inputs.
	std := math.Sqrt(float64(2) / float64(l.InputSize))

	// Create the random number generator.
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Randomize the weights.
	for i := 0; i < l.InputSize; i++ {
		for j := 0; j < l.OutputSize; j++ {
			// Create a random value for the weight and multiply it by the std.
			l.Weights.M[i][j] = r.Float64() * std
		}
	}
}

// Hidden layer forward pass.
func (l *HiddenLayer) Forward(x Matrix) (Matrix, error) {
	// Check that the input matrix is valid.
	if x.Cols != l.InputSize {
		return Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
	}

	// Complete the feedforward process (Y = relu(XW + B)).
	out, err := x.Dot(*l.Weights)
	if err != nil {
		return Matrix{}, err
	}

	for i := 0; i < out.Rows; i++ {
		for j := 0; j < out.Cols; j++ {
			out.M[i][j] += l.Biases.M[0][j]
		}
	}

	// Save the RELU inputs.
	l.reluInputs = out

	// Apply RELU activation for the hidden layer.
	out = RELU(out)

	// Return the matrix.
	return out, nil
}

// Hidden layer backward pass. Arguments are the input matrix and the gradients from the next layer. Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *HiddenLayer) Backward(x Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
	// Check that the input and output matricies are valid.
	if x.Cols != l.InputSize {
		return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
	}
	if dValues.Cols != l.OutputSize {
		return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
	}

	// Calculate the gradients on the RELU activation function.
	dValues = RELUPrime(dValues, l.reluInputs)

	// Complete the backpropagation process and calculate the gradients.
	it := x.T()
	wt := l.Weights.T()
	dWeights, err := it.Dot(dValues)
	if err != nil {
		return Matrix{}, Matrix{}, Matrix{}, err
	}

	dBiases := dValues.Sum(0)

	dInputs, err := dValues.Dot(wt)

	return dWeights, dBiases, dInputs, nil
}


// Main linear neural network output layer struct.
type LinearLayer struct {
        InputSize  int
        OutputSize int
        Weights    *Matrix
        Biases     *Matrix
}

// Create a new linear layer.
func NewLinearLayer(inputSize, outputSize int) (LinearLayer, error) {
        // Check that the input and output sizes are valid.
        if inputSize < 1 || outputSize < 1 {
                return LinearLayer{}, invalidLayerDimensionsError(inputSize, outputSize)
        }

        // Create the new matricies.
        weights, _ := NewMatrix(inputSize, outputSize)
        biases, _ := NewMatrix(1, outputSize)

        // Create and return the new hidden layer.
        return LinearLayer{
                InputSize:  inputSize,
                OutputSize: outputSize,
                Weights:    &weights,
                Biases:     &biases,
        }, nil
}

// Get the values for the layer.
func (l *LinearLayer) getValues() (*Matrix, *Matrix, map[string]float64) {
	values := map[string]float64{"inputs": float64(l.InputSize), "outputs": float64(l.OutputSize), "type": float64(LinearLayerType)}
        return l.Weights, l.Biases, values
}

// Set the values for the layer.
func (l *LinearLayer) setValues(weights, biases Matrix, values map[string]float64) {
        l.InputSize = int(values["inputs"])
        l.OutputSize = int(values["outputs"])
        l.Weights = &weights
        l.Biases = &biases
}

// Initialize the linear layer values.
func (l *LinearLayer) Init() {
        // Using He weight initialization. Calculate the std for the weights based on the number of inputs.
        std := math.Sqrt(float64(2) / float64(l.InputSize))

        // Create the random number generator.
        r := rand.New(rand.NewSource(time.Now().UnixNano()))

        // Randomize the weights.
        for i := 0; i < l.InputSize; i++ {
                for j := 0; j < l.OutputSize; j++ {
                        // Create a random value for the weight and multiply it by the std.
                        l.Weights.M[i][j] = r.Float64() * std
                }
        }
}

// Linear layer forward pass.
func (l *LinearLayer) Forward(x Matrix) (Matrix, error) {
        // Check that the input matrix is valid.
        if x.Cols != l.InputSize {
                return Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }

        // Complete the feedforward process (Y = X + B).
        out, err := x.Dot(*l.Weights)
        if err != nil {
                return Matrix{}, err
        }

        for i := 0; i < out.Rows; i++ {
                for j := 0; j < out.Cols; j++ {
                        out.M[i][j] += l.Biases.M[0][j]
                }
        }

        // Return the matrix.
        return out, nil
}

// Linear layer backward pass. Arguments are the input matrix and the gradients from the next layer (AKA loss). Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *LinearLayer) Backward(x Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
        // Check that the input and output matricies are valid.
        if x.Cols != l.InputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }
        if dValues.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
        }

        // Complete the backpropagation process and calculate the gradients.
        it := x.T()
        wt := l.Weights.T()
        dWeights, err := it.Dot(dValues)
        if err != nil {
                return Matrix{}, Matrix{}, Matrix{}, err
        }

        dBiases := dValues.Sum(0)

        dInputs, err := dValues.Dot(wt)

        return dWeights, dBiases, dInputs, nil
}


// Main sigmoid neural network layer struct.
type SigmoidLayer struct {
        InputSize  int
        OutputSize int
        Weights    *Matrix
        Biases     *Matrix
}

// Create a new sigmoid layer.
func NewSigmoidLayer(inputSize, outputSize int) (SigmoidLayer, error) {
        // Check that the input and output sizes are valid.
        if inputSize < 1 || outputSize < 1 {
                return SigmoidLayer{}, invalidLayerDimensionsError(inputSize, outputSize)
        }

        // Create the new matricies.
        weights, _ := NewMatrix(inputSize, outputSize)
        biases, _ := NewMatrix(1, outputSize)

        // Create and return the new sigmoid layer.
        return SigmoidLayer{
                InputSize:  inputSize,
                OutputSize: outputSize,
                Weights:    &weights,
                Biases:     &biases,
        }, nil
}

// Get the values for the layer.
func (l *SigmoidLayer) getValues() (*Matrix, *Matrix, map[string]float64) {
	values := map[string]float64{"inputs": float64(l.InputSize), "outputs": float64(l.OutputSize), "type": float64(SigmoidLayerType)}
        return l.Weights, l.Biases, values
}

// Set the values for the layer.
func (l *SigmoidLayer) setValues(weights, biases Matrix, values map[string]float64) {
        l.InputSize = int(values["inputs"])
        l.OutputSize = int(values["outputs"])
        l.Weights = &weights
        l.Biases = &biases
}

// Initialize the sigmoid layer values.
func (l *SigmoidLayer) Init() {
        // Using He weight initialization. Calculate the std for the weights based on the number of inputs.
        std := math.Sqrt(float64(2) / float64(l.InputSize))

        // Create the random number generator.
        r := rand.New(rand.NewSource(time.Now().UnixNano()))

        // Randomize the weights.
        for i := 0; i < l.InputSize; i++ {
                for j := 0; j < l.OutputSize; j++ {
                        // Create a random value for the weight and multiply it by the std.
                        l.Weights.M[i][j] = r.Float64() * std
                }
        }
}

// Sigmoid layer forward pass.
func (l *SigmoidLayer) Forward(x Matrix) (Matrix, error) {
        // Check that the input matrix is valid.
        if x.Cols != l.InputSize {
                return Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }

        // Complete the feedforward process (Y = X + B).
        out, err := x.Dot(*l.Weights)
        if err != nil {
                return Matrix{}, err
        }

        for i := 0; i < out.Rows; i++ {
                for j := 0; j < out.Cols; j++ {
                        out.M[i][j] += l.Biases.M[0][j]
                }
        }

	// Add the sigmoid activation function.
	out = Sigmoid(out)

        // Return the matrix.
        return out, nil
}

// Sigmoid layer backward pass. Arguments are the input matrix and the gradients from the next layer. Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *SigmoidLayer) Backward(x Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
        // Check that the input and output matricies are valid.
        if x.Cols != l.InputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }
        if dValues.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
        }

        // Calculate the gradients on the sigmoid activation function.
        dValues = SigmoidPrime(x, dValues)

        // Complete the backpropagation process and calculate the gradients.
        it := x.T()
        wt := l.Weights.T()
        dWeights, err := it.Dot(dValues)
        if err != nil {
                return Matrix{}, Matrix{}, Matrix{}, err
        }

        dBiases := dValues.Sum(0)

        dInputs, err := dValues.Dot(wt)

        return dWeights, dBiases, dInputs, nil
}


// Main leaky RELU neural network layer struct.
type LeakyLayer struct {
        InputSize  int
        OutputSize int
        Weights    *Matrix
        Biases     *Matrix
	Slope      float64
	reluInputs Matrix
}

// Create a new leaky layer.
func NewLeakyLayer(inputSize, outputSize int, slope float64) (LeakyLayer, error) {
        // Check that the input and output sizes are valid.
        if inputSize < 1 || outputSize < 1 {
                return LeakyLayer{}, invalidLayerDimensionsError(inputSize, outputSize)
        }

	// Check that the slope value is valid
	if slope < 0 {
		return LeakyLayer{}, errors.New("nn.LeakyLayer: Invalid LeakyRELU slope.")
	}

        // Create the new matricies.
        weights, _ := NewMatrix(inputSize, outputSize)
        biases, _ := NewMatrix(1, outputSize)

        // Create and return the new leaky layer.
        return LeakyLayer{
                InputSize:  inputSize,
                OutputSize: outputSize,
                Weights:    &weights,
                Biases:     &biases,
		Slope:      slope,
        }, nil
}

// Get the values for the layer.
func (l *LeakyLayer) getValues() (*Matrix, *Matrix, map[string]float64) {
	values := map[string]float64{"inputs": float64(l.InputSize), "outputs": float64(l.OutputSize), "slope": l.Slope, "type": float64(LeakyLayerType)}
        return l.Weights, l.Biases, values
}

// Set the values for the layer.
func (l *LeakyLayer) setValues(weights, biases Matrix, values map[string]float64) {
        l.InputSize = int(values["inputs"])
        l.OutputSize = int(values["outputs"])
	l.Slope = values["slope"]
        l.Weights = &weights
        l.Biases = &biases
}

// Initialize the leaky layer values.
func (l *LeakyLayer) Init() {
        // Using He weight initialization. Calculate the std for the weights based on the number of inputs.
        std := math.Sqrt(float64(2) / float64(l.InputSize))

        // Create the random number generator.
        r := rand.New(rand.NewSource(time.Now().UnixNano()))

        // Randomize the weights.
        for i := 0; i < l.InputSize; i++ {
                for j := 0; j < l.OutputSize; j++ {
                        // Create a random value for the weight and multiply it by the std.
                        l.Weights.M[i][j] = r.Float64() * std
                }
        }
}

// Leaky layer forward pass.
func (l *LeakyLayer) Forward(x Matrix) (Matrix, error) {
        // Check that the input matrix is valid.
        if x.Cols != l.InputSize {
                return Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }

        // Complete the feedforward process (Y = relu(XW + B)).
        out, err := x.Dot(*l.Weights)
        if err != nil {
                return Matrix{}, err
        }

        for i := 0; i < out.Rows; i++ {
                for j := 0; j < out.Cols; j++ {
                        out.M[i][j] += l.Biases.M[0][j]
                }
        }

	// Save the RELU inputs.
	l.reluInputs = out

        // Apply leaky RELU activation for the hidden layer.
        out = LeakyRELU(out, l.Slope)

        // Return the matrix.
        return out, nil
}

// Leaky layer backward pass. Arguments are the input matrix and the gradients from the next layer. Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *LeakyLayer) Backward(x Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
        // Check that the input and output matricies are valid.
        if x.Cols != l.InputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }
        if dValues.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
        }

        // Calculate the gradients on the leaky RELU activation function.
	dValues = LeakyRELUPrime(dValues, l.reluInputs, l.Slope)

        // Complete the backpropagation process and calculate the gradients.
        it := x.T()
        wt := l.Weights.T()
        dWeights, err := it.Dot(dValues)
        if err != nil {
                return Matrix{}, Matrix{}, Matrix{}, err
        }

        dBiases := dValues.Sum(0)

        dInputs, err := dValues.Dot(wt)

        return dWeights, dBiases, dInputs, nil
}


// Main softmax neural network layer struct.
type SoftmaxLayer struct {
        InputSize  int
        OutputSize int
        Weights    *Matrix
        Biases     *Matrix
	outputs    Matrix
}

// Create a new softmax layer.
func NewSoftmaxLayer(inputSize, outputSize int) (SoftmaxLayer, error) {
        // Check that the input and output sizes are valid.
        if inputSize < 1 || outputSize < 1 {
                return SoftmaxLayer{}, invalidLayerDimensionsError(inputSize, outputSize)
        }

        // Create the new matricies.
        weights, _ := NewMatrix(inputSize, outputSize)
        biases, _ := NewMatrix(1, outputSize)

        // Create and return the new softmax layer.
        return SoftmaxLayer{
                InputSize:  inputSize,
                OutputSize: outputSize,
                Weights:    &weights,
                Biases:     &biases,
        }, nil
}

// Get the values for the layer.
func (l *SoftmaxLayer) getValues() (*Matrix, *Matrix, map[string]float64) {
	values := map[string]float64{"inputs": float64(l.InputSize), "outputs": float64(l.OutputSize), "type": float64(SoftmaxLayerType)}
        return l.Weights, l.Biases, values
}

// Set the values for the layer.
func (l *SoftmaxLayer) setValues(weights, biases Matrix, values map[string]float64) {
        l.InputSize = int(values["inputs"])
        l.OutputSize = int(values["outputs"])
        l.Weights = &weights
        l.Biases = &biases
}

// Initialize the softmax layer values.
func (l *SoftmaxLayer) Init() {
        // Using He weight initialization. Calculate the std for the weights based on the number of inputs.
        std := math.Sqrt(float64(2) / float64(l.InputSize))

        // Create the random number generator.
        r := rand.New(rand.NewSource(time.Now().UnixNano()))

        // Randomize the weights.
        for i := 0; i < l.InputSize; i++ {
                for j := 0; j < l.OutputSize; j++ {
                        // Create a random value for the weight and multiply it by the std.
                        l.Weights.M[i][j] = r.Float64() * std
                }
        }
}

// Softmax layer forward pass.
func (l *SoftmaxLayer) Forward(x Matrix) (Matrix, error) {
        // Check that the input matrix is valid.
        if x.Cols != l.InputSize {
                return Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }

        // Complete the feedforward process (Y = softmax(XW + B)).
        out, err := x.Dot(*l.Weights)
        if err != nil {
                return Matrix{}, err
        }

        for i := 0; i < out.Rows; i++ {
                for j := 0; j < out.Cols; j++ {
                        out.M[i][j] += l.Biases.M[0][j]
                }
        }

        // Apply softmax activation for the softmax layer.
	out = Softmax(out)

        // Save the outputs.
	l.outputs = out

	// Return the matrix.
        return out, nil
}

// Softmax layer backward pass. Arguments are the input matrix and the gradients from the next layer. Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *SoftmaxLayer) Backward(x Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
        // Check that the input and output matricies are valid.
	if x.Cols != l.InputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }
        if l.outputs.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(l.outputs.Rows, l.outputs.Cols)
        }
        if dValues.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
        }
	if l.outputs.Rows != dValues.Rows {
		return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(l.outputs.Rows, dValues.Rows)
	}

        // Calculate the gradients on the softmax activation function.
	dValuesPrev := dValues
	dValues, _ = NewMatrix(dValues.Rows, dValues.Cols)

	for i := 0; i < dValues.Rows; i++ {
		// Calculate the Jacobian matrix of the OUTPUT.
		// Create a diagonal matrix from a single row of outputs.
		diag, _ := NewMatrix(l.outputs.Cols, l.outputs.Cols)
		for j := 0; j < l.outputs.Cols; j++ {
			diag.M[j][j] = l.outputs.M[i][j]
		}

		// Dot the output row with itself, transformed.
		outputRow, _ := NewMatrixFromSlice([][]float64{l.outputs.M[i]})
		outputRow = outputRow.T()
		outputDot, err := outputRow.Dot(outputRow.T())
		if err != nil {
			return Matrix{}, Matrix{}, Matrix{}, err
		}

		// Subtract the dot matrix from the diagonal matrix.
		jacobian, err := diag.Sub(outputDot)
		if err != nil {
			return Matrix{}, Matrix{}, Matrix{}, err
		}

		dValuesRow, _ := NewMatrixFromSlice([][]float64{dValuesPrev.M[i]})
		row, err := jacobian.Dot(dValuesRow.T())
		if err != nil {
			return Matrix{}, Matrix{}, Matrix{}, err
		}

		dValues.SetRow(i, row.T().M[0])
	}

	// Complete the backpropagation process and calculate the gradients.
        it := x.T()
        wt := l.Weights.T()
        dWeights, err := it.Dot(dValues)
        if err != nil {
                return Matrix{}, Matrix{}, Matrix{}, err
        }

        dBiases := dValues.Sum(0)

        dInputs, err := dValues.Dot(wt)

        return dWeights, dBiases, dInputs, nil
}

// Softmax + categorial cross-entropy loss layer backward pass. Arguments are the input, correct output and output matricies. Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *SoftmaxLayer) BackwardCrossEntropy(x Matrix, y Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
        // Check that the input and output matricies are valid.
	if y.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(y.Rows, y.Cols)
        }
        if dValues.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
        }
        if y.Rows != dValues.Rows {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(y.Rows, dValues.Rows)
        }

	newValues, _ := NewMatrix(dValues.Rows, dValues.Cols)
	for i := 0; i < dValues.Rows; i++ {
		for j := 0; j < dValues.Cols; j++ {
			if y.M[i][j] == 1 {
				newValues.M[i][j] = (dValues.M[i][j] - 1)/float64(dValues.Rows)
			} else {
				newValues.M[i][j] = dValues.M[i][j]/float64(dValues.Rows)
			}
		}
	}

	// Complete the backpropagation process and calculate the gradients.
        it := x.T()
        wt := l.Weights.T()
        dWeights, err := it.Dot(newValues)
	if err != nil {
                return Matrix{}, Matrix{}, Matrix{}, err
        }

        dBiases := newValues.Sum(0)

        dInputs, err := newValues.Dot(wt)

        return dWeights, dBiases, dInputs, nil
}


// Main dropout neural network layer struct.
type DropoutLayer struct {
        InputSize  int
        OutputSize int
        Weights    *Matrix
        Biases     *Matrix
	Dropout    float64
        reluInputs Matrix
	binaryMask Matrix
}

// Create a new dropout layer.
func NewDropoutLayer(inputSize, outputSize int, dropout float64) (DropoutLayer, error) {
        // Check that the input and output sizes are valid.
        if inputSize < 1 || outputSize < 1 {
                return DropoutLayer{}, invalidLayerDimensionsError(inputSize, outputSize)
        }

	// Check that the dropout value is correct.
	if dropout > 1 || dropout < 0 {
		return DropoutLayer{}, errors.New("Invalid dropout value.")
	}

        // Create the new matricies.
        weights, _ := NewMatrix(inputSize, outputSize)
        biases, _ := NewMatrix(1, outputSize)

        // Create and return the new hidden layer.
        return DropoutLayer{
                InputSize:  inputSize,
                OutputSize: outputSize,
                Weights:    &weights,
                Biases:     &biases,
		Dropout:    dropout,
	}, nil
}

// Get the values for the layer.
func (l *DropoutLayer) getValues() (*Matrix, *Matrix, map[string]float64) {
	values := map[string]float64{"inputs": float64(l.InputSize), "outputs": float64(l.OutputSize), "type": float64(DropoutLayerType), "dropout": l.Dropout}
        return l.Weights, l.Biases, values
}

// Set the values for the layer.
func (l *DropoutLayer) setValues(weights, biases Matrix, values map[string]float64) {
        l.InputSize = int(values["inputs"])
        l.OutputSize = int(values["outputs"])
        l.Dropout = values["dropout"]
	l.Weights = &weights
        l.Biases = &biases
}

// Initialize the dropout layer values.
func (l *DropoutLayer) Init() {
        // Using He weight initialization. Calculate the std for the weights based on the number of inputs.
        std := math.Sqrt(float64(2) / float64(l.InputSize))

        // Create the random number generator.
        r := rand.New(rand.NewSource(time.Now().UnixNano()))

        // Randomize the weights.
        for i := 0; i < l.InputSize; i++ {
                for j := 0; j < l.OutputSize; j++ {
                        // Create a random value for the weight and multiply it by the std.
                        l.Weights.M[i][j] = r.Float64() * std
                }
        }
}

// Dropout layer forward pass.
func (l *DropoutLayer) Forward(x Matrix) (Matrix, error) {
        // Check that the input matrix is valid.
        if x.Cols != l.InputSize {
                return Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }

        // Complete the feedforward process (Y = dropout(relu(XW + B))).
        out, err := x.Dot(*l.Weights)
        if err != nil {
                return Matrix{}, err
        }

        for i := 0; i < out.Rows; i++ {
                for j := 0; j < out.Cols; j++ {
                        out.M[i][j] += l.Biases.M[0][j]
                }
        }

        // Save the RELU inputs.
        l.reluInputs = out

        // Apply RELU activation for the hidden layer.
        out = RELU(out)

	// Apply a scaled binomial distribution matrix to the 
	l.binaryMask, _ = NewMatrix(out.Rows, out.Cols)
	binomial := binomial{N: 1, P: 1 - l.Dropout}
	binomial.NewSource()
	for i := 0; i < out.Rows; i++ {
		for j := 0; j < out.Cols; j++ {
			l.binaryMask.M[i][j] = binomial.Rand()
			out.M[i][j] *= l.binaryMask.M[i][j] / (1 - l.Dropout)
		}
	}

        // Return the matrix.
        return out, nil
}

// Dropout layer forward pass without dropout.
func (l *DropoutLayer) ForwardNoDropout(x Matrix) (Matrix, error) {
        // Check that the input matrix is valid.
        if x.Cols != l.InputSize {
                return Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }

        // Complete the feedforward process (Y = dropout(relu(XW + B))).
        out, err := x.Dot(*l.Weights)
        if err != nil {
                return Matrix{}, err
        }

        for i := 0; i < out.Rows; i++ {
                for j := 0; j < out.Cols; j++ {
                        out.M[i][j] += l.Biases.M[0][j]
                }
        }

        // Save the RELU inputs.
        l.reluInputs = out

        // Apply RELU activation for the hidden layer.
        out = RELU(out)

	// Return the matrix.
	return out, nil
}

// Dropout layer backward pass. Arguments are the input matrix and the gradients from the next layer. Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *DropoutLayer) Backward(x Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
        // Check that the input and output matricies are valid.
        if x.Cols != l.InputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }
        if dValues.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
        }

	// Calculate the gradients on the dropout.
	for i := 0; i < dValues.Rows; i++ {
                for j := 0; j < dValues.Cols; j++ {
			dValues.M[i][j] *= l.binaryMask.M[i][j]
		}
	}

        // Calculate the gradients on the RELU activation function.
        dValues = RELUPrime(dValues, l.reluInputs)

        // Complete the backpropagation process and calculate the gradients.
        it := x.T()
        wt := l.Weights.T()
        dWeights, err := it.Dot(dValues)
        if err != nil {
                return Matrix{}, Matrix{}, Matrix{}, err
        }

        dBiases := dValues.Sum(0)

        dInputs, err := dValues.Dot(wt)

        return dWeights, dBiases, dInputs, nil
}
