// network.go
// Main neural network layer code.

// NOTE: due to a arcchitectual failure on the author's part, lots of layer code is repeated unnecessarily

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
	dValues = RELUPrime(dValues)

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
        dValues = SigmoidPrime(dValues)

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
