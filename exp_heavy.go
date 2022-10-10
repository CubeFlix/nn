// exp_heavy.go
// Experimental "Heavy" quadratic neural networks.

package nn

import (
	"time"
	"math"
	"math/rand"
	"errors"
	"fmt"
)


// Main heavy neural network layer struct.
type HeavyLayer struct {
	InputSize  int
	OutputSize int
	Weights    *Matrix
	Heavies    *Matrix
	Biases     *Matrix
	reluInputs Matrix
	xSquared   Matrix
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

// Heavy layer forward pass.
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
	l.xSquared = xPow

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

// Heavy layer backward pass. Arguments are the input matrix and the gradients from the next layer. Outputs the gradients for the heavies, weights, biases, and inputs, respectively.
func (l *HeavyLayer) Backward(x Matrix, dValues Matrix) (Matrix, Matrix, Matrix, Matrix, error) {
	// Check that the input and output matricies are valid.
	if x.Cols != l.InputSize {
		return Matrix{}, Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(x.Rows, x.Cols)
	}
	if dValues.Cols != l.OutputSize {
		return Matrix{}, Matrix{}, Matrix{}, Matrix{}, invalidMatrixDimensionsError(dValues.Rows, dValues.Cols)
	}

	// Calculate the gradients on the RELU activation function.
	dValues = RELUPrime(dValues, l.reluInputs)

	// Complete the backpropagation process and calculate the gradients.
	// let . denote element-wise product
	// dY/dH = X^2 * dPrev
	// dY/dX = dPrev * (2H^T . X + W^T)
	it := x.T()
	xst := l.xSquared.T()
	wt := l.Weights.T()
	ht := l.Heavies.T()
	dWeights, err := it.Dot(dValues)
	if err != nil {
		return Matrix{}, Matrix{}, Matrix{}, Matrix{}, err
	}

	dHeavies, err := xst.Dot(dValues)
	if err != nil {
		return Matrix{}, Matrix{}, Matrix{}, Matrix{}, err
	}

	dBiases := dValues.Sum(0)

	twoXH, _ := NewMatrix(l.OutputSize, l.InputSize)
	for i := 0; i < l.OutputSize; i++ {
		for j := 0; j < l.InputSize; j++ {
			twoXH.M[i][j] = ht.M[i][j] * x.M[i][j] * 2 + wt.M[i][j]
		}
	}
	dInputs, err := dValues.Dot(twoXH)
	if err != nil {
		return Matrix{}, Matrix{}, Matrix{}, Matrix{}, err
	}

	return dHeavies, dWeights, dBiases, dInputs, nil
}


// Heavy version of Adam.


// Adam optimizer object. Can handle a single layer.
type HeavyAdamOptimizer struct {
        LearningRate    float64
        Decay           float64
        Epsilon         float64
	Beta1           float64
	Beta2           float64
        currentRate     float64
        iterations      int
		heavyMomentums Matrix
        weightMomentums Matrix
        biasMomentums   Matrix
		heavyCache     Matrix
	weightCache     Matrix
	biasCache       Matrix
        useDecay        bool
        useMomentum     bool
}

// Get the optimizer values.
func (optimizer *HeavyAdamOptimizer) getValues() (map[string]float64) {
        return map[string]float64{
                "learningRate": optimizer.LearningRate,
                "decay":        optimizer.Decay,
                "epsilon":      optimizer.Epsilon,
		"beta1":        optimizer.Beta1,
		"beta2":        optimizer.Beta2,
		"type":         float64(AdamOptimizerType),
	}
}

// Set the optimizer values.
func (optimizer *HeavyAdamOptimizer) setValues(values map[string]float64) {
        optimizer.LearningRate = values["learningRate"]
        optimizer.Decay = values["decay"]
        optimizer.Epsilon = values["epsilon"]
	optimizer.Beta1 = values["beta1"]
	optimizer.Beta2 = values["beta2"]
}

// Create a new Adam optimizer object.
func NewHeavyAdamOptimizer(learningRate, decay, epsilon, beta1, beta2 float64) (HeavyAdamOptimizer, error) {
        // Check that the optimizer values are valid.
        if learningRate <= 0 {
                return HeavyAdamOptimizer{}, errors.New(fmt.Sprintf("nn.HeavyAdamOptimizer: Invalid learning rate: %f", learningRate))
        }
        if decay < 0 {
                return HeavyAdamOptimizer{}, errors.New(fmt.Sprintf("nn.HeavyAdamOptimizer: Invalid decay: %f", decay))
        }

        // Create the new Adam optimizer object.
        return HeavyAdamOptimizer{
                LearningRate: learningRate,
                Decay:        decay,
                Epsilon:      epsilon,
		Beta1:        beta1,
		Beta2:        beta2,
		currentRate:  learningRate,
                iterations:   0,
                useDecay:     (decay != 0),
        }, nil
}

// Update the weights and biases for the layer.
func (optimizer *HeavyAdamOptimizer) Update(heavies *Matrix, weights *Matrix, biases *Matrix, dHeavies Matrix, dWeights Matrix, dBiases Matrix) error {
	// Calculate the new learning rate.
	if optimizer.useDecay {
                optimizer.currentRate = optimizer.LearningRate * (float64(1) / (float64(1) + optimizer.Decay * float64(optimizer.iterations)))
        }

	if optimizer.weightMomentums.Rows == 0 {
                // Weight and bias momentums are nil.
				optimizer.heavyMomentums, _ = NewMatrix(heavies.Rows, heavies.Cols)
                optimizer.weightMomentums, _ = NewMatrix(weights.Rows, weights.Cols)
                optimizer.biasMomentums, _ = NewMatrix(biases.Rows, biases.Cols)
				optimizer.heavyCache, _ = NewMatrix(heavies.Rows, heavies.Cols)
		optimizer.weightCache, _ = NewMatrix(weights.Rows, weights.Cols)
		optimizer.biasCache, _ = NewMatrix(biases.Rows, biases.Cols)
        }

	// Calculate the new weight and bias momentums.
	scaledHeavyMomentums := optimizer.heavyMomentums.MulScalar(optimizer.Beta1)
	scaledHeavyGradients := dHeavies.MulScalar(float64(1) - optimizer.Beta1)
	err := errors.New("")
	optimizer.heavyMomentums, err = scaledHeavyMomentums.Add(scaledHeavyGradients)
	if err != nil {
		return err
	}

	scaledWeightMomentums := optimizer.weightMomentums.MulScalar(optimizer.Beta1)
	scaledWeightGradients := dWeights.MulScalar(float64(1) - optimizer.Beta1)
	optimizer.weightMomentums, err = scaledWeightMomentums.Add(scaledWeightGradients)
	if err != nil {
		return err
	}

	scaledBiasMomentums := optimizer.biasMomentums.MulScalar(optimizer.Beta1)
        scaledBiasGradients := dBiases.MulScalar(float64(1) - optimizer.Beta1)
	optimizer.biasMomentums, err = scaledBiasMomentums.Add(scaledBiasGradients)
	if err != nil {
                return err
        }

	// Calculate the current weight and bias momentums.
	correctedHeavyMomentums := optimizer.heavyMomentums.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))
	correctedWeightMomentums := optimizer.weightMomentums.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))
	correctedBiasMomentums := optimizer.biasMomentums.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))

	// Calculate the new weight and bias caches.
	scaledHeavyMomentums = optimizer.heavyCache.MulScalar(optimizer.Beta2)
	scaledHeavyGradients = dHeavies.PowScalar(2)
        scaledHeavyGradients = scaledHeavyGradients.MulScalar(float64(1) - optimizer.Beta2)
	optimizer.heavyCache, err = scaledHeavyMomentums.Add(scaledHeavyGradients)
	if err != nil {
                return err
        }

	scaledWeightMomentums = optimizer.weightCache.MulScalar(optimizer.Beta2)
	scaledWeightGradients = dWeights.PowScalar(2)
        scaledWeightGradients = scaledWeightGradients.MulScalar(float64(1) - optimizer.Beta2)
	optimizer.weightCache, err = scaledWeightMomentums.Add(scaledWeightGradients)
	if err != nil {
                return err
        }

        scaledBiasMomentums = optimizer.biasCache.MulScalar(optimizer.Beta2)
        scaledBiasGradients = dBiases.PowScalar(2)
        scaledBiasGradients = scaledBiasGradients.MulScalar(float64(1) - optimizer.Beta2)
        optimizer.biasCache, err = scaledBiasMomentums.Add(scaledBiasGradients)
	if err != nil {
                return err
        }

	// Calculate the corrected weight and bias caches.
	correctedHeavyCaches := optimizer.heavyCache.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))
	correctedWeightCaches := optimizer.weightCache.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))
	correctedBiasCaches := optimizer.biasCache.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))

	// Update the weights and biases matricies.
	for i := 0; i < heavies.Rows; i++ {
		for j := 0; j < heavies.Cols; j++ {
	heavies.M[i][j] += -optimizer.currentRate * correctedHeavyMomentums.M[i][j] / (math.Sqrt(correctedHeavyCaches.M[i][j]) + optimizer.Epsilon)
		}
}

        for i := 0; i < weights.Rows; i++ {
                for j := 0; j < weights.Cols; j++ {
			weights.M[i][j] += -optimizer.currentRate * correctedWeightMomentums.M[i][j] / (math.Sqrt(correctedWeightCaches.M[i][j]) + optimizer.Epsilon)
                }
        }

        for i := 0; i < biases.Rows; i++ {
		for j := 0; j < biases.Cols; j++ {
                        biases.M[i][j] += -optimizer.currentRate * correctedBiasMomentums.M[i][j] / (math.Sqrt(correctedBiasCaches.M[i][j]) + optimizer.Epsilon)
                }
        }

	optimizer.iterations += 1

	return nil
}
