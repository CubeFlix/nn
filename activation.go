// nn.go
// Activation functions for the neural network.

package nn

import (
	"math"
)


// RELU activation function.
func RELU(m Matrix) Matrix {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			// Calculate the RELU of the value.
			m.M[i][j] = math.Max(m.M[i][j], 0)
		}
	}

	// Return the final matrix.
	return m
}

// RELU gradient function.
func RELUPrime(m Matrix) Matrix {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			// Calculate the derivatives for RELU (heaviside step function)
			if m.M[i][j] < 0 {
				m.M[i][j] = 0
			} else {
				m.M[i][j] = 1
			}
		}
	}

	// Return the final matrix.
	return m
}


// Sigmoid activation function.
func Sigmoid(m Matrix) Matrix {
	for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        // Calculate the sigmoid of the value.
                        m.M[i][j] = float64(1)/(1 + math.Exp(-m.M[i][j]))
                }
        }

        // Return the final matrix.
        return m
}

// Sigmoid gradient function.
func SigmoidPrime(m Matrix) Matrix {
	m = Sigmoid(m)

	for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        // Calculate the derivatives for sigmoid (g(x)(1-g(x)))
                        m.M[i][j] = m.M[i][j] * (1-m.M[i][j])
                }
        }

        // Return the final matrix.
        return m
}
