// optimizers.go
// Basic optimizers for networks.

package nn

import (
	"fmt"
	"errors"
	"math"
)

// Stochastic gradient descent optimizer object. Can handle a single layer.
type SGDOptimizer struct {
	LearningRate    float64
	Decay           float64
	Momentum        float64
	currentRate     float64
	iterations      int
	weightMomentums Matrix
	biasMomentums   Matrix
	useDecay        bool
	useMomentum     bool
}

// Create a new SGD optimizer object.
func NewSGDOptimizer(learningRate, decay, momentum float64) (SGDOptimizer, error) {
	// Check that the optimizer values are valid.
	if learningRate <= 0 {
		return SGDOptimizer{}, errors.New(fmt.Sprintf("nn.SGDOptimizer: Invalid learning rate: %f", learningRate))
	}
	if decay < 0 {
		return SGDOptimizer{}, errors.New(fmt.Sprintf("nn.SGDOptimizer: Invalid decay: %f", decay))
	}
	if momentum < 0 {
		return SGDOptimizer{}, errors.New(fmt.Sprintf("nn.SGDOptimizer: Invalid momentum: %f", momentum))
	}

	// Create the new SGD optimizer object.
	return SGDOptimizer{
		LearningRate: learningRate,
		Decay:        decay,
		Momentum:     momentum,
		currentRate:  learningRate,
		iterations:   0,
		useDecay:     (decay != 0),
		useMomentum:  (momentum != 0),
	}, nil
}

// Update the weights and biases for the layer.
func (optimizer *SGDOptimizer) Update(weights *Matrix, biases *Matrix, dWeights Matrix, dBiases Matrix) error {
	// Calculate the new learning rate.
	if optimizer.useDecay {
		optimizer.currentRate = optimizer.LearningRate * (float64(1) / (float64(1) + optimizer.Decay * float64(optimizer.iterations)))
	}

	// Update the weight and bias values.
	if optimizer.useMomentum {
		if optimizer.weightMomentums.Rows == 0 {
			// Weight and bias momentums are nil.
			optimizer.weightMomentums, _ = NewMatrix(weights.Rows, weights.Cols)
			optimizer.biasMomentums, _ = NewMatrix(biases.Rows, biases.Cols)
		}

		// Calculate weight updates with momentum.
		scaledMomentumsWeights := optimizer.weightMomentums.MulScalar(optimizer.Momentum)
		scaledGradientsWeights := dWeights.MulScalar(optimizer.currentRate)
		err := errors.New("")
		optimizer.weightMomentums, err = scaledMomentumsWeights.Sub(scaledGradientsWeights)
		if err != nil {
			return err
		}

		// Calculate bias updates with momentum.
		scaledMomentumsBiases := optimizer.biasMomentums.MulScalar(optimizer.Momentum)
		scaledGradientsBiases := dBiases.MulScalar(optimizer.currentRate)
		optimizer.biasMomentums, err = scaledMomentumsBiases.Sub(scaledGradientsBiases)
		if err != nil {
			return err
		}

		// Update the weights and biases matricies.
		for i := 0; i < weights.Rows; i++ {
			for j := 0; j < weights.Cols; j++ {
				weights.M[i][j] += optimizer.weightMomentums.M[i][j]
			}
		}

		for i := 0; i < biases.Rows; i++ {
                        for j := 0; j < biases.Cols; j++ {
                                biases.M[i][j] += optimizer.biasMomentums.M[i][j]
                        }
                }
	} else {
		// Calculate weight and bias updates.
		weightUpdates := dWeights.MulScalar(optimizer.currentRate)
		biasUpdates := dBiases.MulScalar(optimizer.currentRate)

		// Update the weights and biases matricies.
                for i := 0; i < weights.Rows; i++ {
                        for j := 0; j < weights.Cols; j++ {
                                weights.M[i][j] -= weightUpdates.M[i][j]
                        }
                }

                for i := 0; i < biases.Rows; i++ {
                        for j := 0; j < biases.Cols; j++ {
                                biases.M[i][j] -= biasUpdates.M[i][j]
                        }
                }
	}

	optimizer.iterations += 1

	return nil
}

// Adam optimizer object. Can handle a single layer.
type AdamOptimizer struct {
        LearningRate    float64
        Decay           float64
        Epsilon         float64
	Beta1           float64
	Beta2           float64
        currentRate     float64
        iterations      int
        weightMomentums Matrix
        biasMomentums   Matrix
	weightCache     Matrix
	biasCache       Matrix
        useDecay        bool
        useMomentum     bool
}

// Create a new Adam optimizer object.
func NewAdamOptimizer(learningRate, decay, epsilon, beta1, beta2 float64) (AdamOptimizer, error) {
        // Check that the optimizer values are valid.
        if learningRate <= 0 {
                return AdamOptimizer{}, errors.New(fmt.Sprintf("nn.AdamOptimizer: Invalid learning rate: %f", learningRate))
        }
        if decay < 0 {
                return AdamOptimizer{}, errors.New(fmt.Sprintf("nn.AdamOptimizer: Invalid decay: %f", decay))
        }

        // Create the new Adam optimizer object.
        return AdamOptimizer{
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
func (optimizer *AdamOptimizer) Update(weights *Matrix, biases *Matrix, dWeights Matrix, dBiases Matrix) error {
	// Calculate the new learning rate.
	if optimizer.useDecay {
                optimizer.currentRate = optimizer.LearningRate * (float64(1) / (float64(1) + optimizer.Decay * float64(optimizer.iterations)))
        }

	if optimizer.weightMomentums.Rows == 0 {
                // Weight and bias momentums are nil.
                optimizer.weightMomentums, _ = NewMatrix(weights.Rows, weights.Cols)
                optimizer.biasMomentums, _ = NewMatrix(biases.Rows, biases.Cols)
		optimizer.weightCache, _ = NewMatrix(weights.Rows, weights.Cols)
		optimizer.biasCache, _ = NewMatrix(biases.Rows, biases.Cols)
        }

	// Calculate the new weight and bias momentums.
	scaledWeightMomentums := optimizer.weightMomentums.MulScalar(optimizer.Beta1)
	scaledWeightGradients := dWeights.MulScalar(float64(1) - optimizer.Beta1)
	err := errors.New("")
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
	correctedWeightMomentums := optimizer.weightMomentums.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))
	correctedBiasMomentums := optimizer.biasMomentums.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))

	// Calculate the new weight and bias caches.
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
	correctedWeightCaches := optimizer.weightCache.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))
	correctedBiasCaches := optimizer.biasCache.MulScalar(float64(1) / (float64(1) - math.Pow(optimizer.Beta1, float64(optimizer.iterations + 1))))

	// Update the weights and biases matricies.
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
