// exp_heavy.go
// Experimental "Heavy" quadratic neural networks.

package nn

import (
	"time"
	"math"
	"math/rand"
)


// Main heavy neural network layer struct.
type HeavyLayer struct {
	InputSize  int
	OutputSize int
	Weights    *Matrix
	Heavies    *Matrix
	Biases     *Matrix
	reluInputs Matrix
}

// Create a new heavy layer.
func NewHeavyLayer(inputSize, outputSize int) (HeavyLayer, error) {
	// Check that the input and output sizes are valid.
	if inputSize < 1 || outputSize < 1 {
		return HeavyLayer{}, invalidLayerDimensionsError(inputSize, outputSize)
	}

	// Create the new matricies.
	weights, _ := NewMatrix(inputSize, outputSize)
	heavies, _ := NewMatrix(inputSize, outputSize)
	biases, _ := NewMatrix(1, outputSize)

	// Create and return the new heavy layer.
	return HeavyLayer{
		InputSize:  inputSize,
		OutputSize: outputSize,
		Weights:    &weights,
		Heavies:    &heavies,
		Biases:     &biases,
	}, nil
}

// // Get the values for the layer.
// func (l *HeavyLayer) getValues() (*Matrix, *Matrix, map[string]float64) {
// 	values := map[string]float64{"inputs": float64(l.InputSize), "outputs": float64(l.OutputSize)}
// 	return l.Weights, l.Biases, values
// }
// 
// // Set the values for the layer.
// func (l *HeavyLayer) setValues(weights, biases Matrix, values map[string]float64) {
// 	l.InputSize = int(values["inputs"])
// 	l.OutputSize = int(values["outputs"])
// 	l.Weights = &weights
// 	l.Biases = &biases
// }

// Initialize the heavy layer values.
func (l *HeavyLayer) Init() {
	// Using He weight initialization. Calculate the std for the weights based on the number of inputs.
	std := math.Sqrt(float64(2) / float64(l.InputSize))

	// Create the random number generator.
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Randomize the weights.
	for i := 0; i < l.InputSize; i++ {
		for j := 0; j < l.OutputSize; j++ {
			// Create a random value for the weight and multiply it by the std.
			l.Weights.M[i][j] = r.NormFloat64() * std
			l.Heavies.M[i][j] = r.NormFloat64() * std
		}
	}
}

// heavy layer forward pass.
func (l *HeavyLayer) Forward(x Matrix) (Matrix, error) {
	// Check that the input matrix is valid.
	if x.Cols != l.InputSize {
		return Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
	}

	// Complete the feedforward process (Y = relu(XW + X^2H + B)).
	out, err := x.Dot(*l.Weights)
	if err != nil {
		return Matrix{}, err
	}

	xPow := x.PowScalar(2)
	heavyOut, err := xPow.Dot(*l.Heavies)

	for i := 0; i < out.Rows; i++ {
		for j := 0; j < out.Cols; j++ {
			out.M[i][j] += heavyOut.M[i][j] + l.Biases.M[0][j]
		}
	}

	// Save the RELU inputs.
	l.reluInputs = out

	// Apply RELU activation for the heavy layer.
	out = RELU(out)

	// Return the matrix.
	return out, nil
}

// heavy layer backward pass. Arguments are the input matrix and the gradients from the next layer. Ouputs the gradients for the weights, biases, and inputs, respectively.
func (l *HeavyLayer) Backward(x Matrix, dValues Matrix) (Matrix, Matrix, Matrix, error) {
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
