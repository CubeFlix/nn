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
func RELUPrime(m Matrix, x Matrix) Matrix {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			// Calculate the derivatives for RELU (heaviside step function)
			if x.M[i][j] <= 0 {
				m.M[i][j] = 0
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
func SigmoidPrime(m Matrix, dValues Matrix) Matrix {
	m = Sigmoid(m)

	for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        // Calculate the derivatives for sigmoid (g(x)(1-g(x)))
                        m.M[i][j] = dValues.M[i][j] * m.M[i][j] * (1-m.M[i][j])
                }
        }

        // Return the final matrix.
        return m
}

// Leaky RELU activation function.
func LeakyRELU(m Matrix, slope float64) Matrix {
        for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        // Calculate the RELU of the value.
			if m.M[i][j] < 0 {
				m.M[i][j] = m.M[i][j] * slope
			}
                }
        }

        // Return the final matrix.
        return m
}


// Leaky RELU gradient function.
func LeakyRELUPrime(m, x Matrix, slope float64) Matrix {
        for i := 0; i < m.Rows; i++ {
                for j := 0; j < m.Cols; j++ {
                        // Calculate the derivatives for RELU (heaviside step function)
                        if x.M[i][j] < 0 {
                                m.M[i][j] = m.M[i][j] * slope
                        }
                }
        }

        // Return the final matrix.
        return m
}

// Softmax activation function.
func Softmax(m Matrix) Matrix {
	for i := 0; i < m.Rows; i++ {
		// Calculate the sum of the row and divide by it.
		sum := float64(0)
		for j := 0; j < m.Cols; j++ {
			m.M[i][j] = math.Exp(m.M[i][j])
			sum += m.M[i][j]
		}
		for j := 0; j < m.Cols; j++ {
			m.M[i][j] = m.M[i][j]/sum
		}
	}

	// Return the final matrix.
	return m
}
