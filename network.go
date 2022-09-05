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


// Invalid layer dimensions error.
func invalidLayerDimensionsError(inputSize, outputSize int) error {
	return errors.New(fmt.Sprintf("Invalid layer dimensions: %d, %d", inputSize, outputSize))
}


// Main hidden neural network layer struct.
type HiddenLayer struct {
	InputSize  int
	OutputSize int
	Weights    *Matrix
	Biases     *Matrix
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

	// Apply RELU activation for the hidden layer.
	RELU(out)

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
	RELUPrime(dValues)

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
	Sigmoid(out)

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
        SigmoidPrime(x)

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
}

// Create a new leaky layer.
func NewLeakyLayer(inputSize, outputSize int, slope float64) (LeakyLayer, error) {
        // Check that the input and output sizes are valid.
        if inputSize < 1 || outputSize < 1 {
                return LeakyLayer{}, invalidLayerDimensionsError(inputSize, outputSize)
        }

	// Check that the slope value is valid
	if slope < 0 {
		return LeakyLayer{}, errors.New("Invalid LeakyRELU slope.")
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

        // Apply leaky RELU activation for the hidden layer.
        LeakyRELU(out, l.Slope)

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
	LeakyRELUPrime(dValues, l.Slope)

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
	Softmax(out)

        // Return the matrix.
        return out, nil
}

// Softmax layer backward pass. Arguments are the INPUT and OUTPUT matricies and the gradients from the next layer. Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *SoftmaxLayer) Backward(x Matrix, yhat Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
        // Check that the input and output matricies are valid.
	if x.Cols != l.InputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
        }
        if yhat.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(yhat.Rows, yhat.Cols)
        }
        if dValues.Cols != l.OutputSize {
                return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
        }
	if yhat.Rows != dValues.Rows {
		return Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(yhat.Rows, dValues.Rows)
	}

        // Calculate the gradients on the softmax activation function.
	dValuesPrev := dValues
	dValues, _ = NewMatrix(dValues.Rows, dValues.Cols)

	for i := 0; i < dValues.Rows; i++ {
		// Calculate the Jacobian matrix of the OUTPUT.
		// Create a diagonal matrix from a single row of outputs.
		diag, _ := NewMatrix(yhat.Cols, yhat.Cols)
		for j := 0; j < yhat.Cols; j++ {
			diag.M[j][j] = yhat.M[i][j]
		}

		// Dot the output row with itself, transformed.
		outputRow, _ := NewMatrixFromSlice([][]float64{yhat.M[i]})
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
// Softmax+loss layer
