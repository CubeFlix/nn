// optimizers.go
// Basic optimizers for networks.

package nn

// Stochastic gradient descent optimizer object. Can handle a single layer.
type SGDOptimizer struct {
	LearningRate    float64
	Decay           float64
	Momentum        float64
	currentMomentum float64
	iterations      int
}

// Update the weights and biases for the layer.
func (optimizer *SGDOptimizer) Update(weights *Matrix, biases *Matrix) {
	
}
