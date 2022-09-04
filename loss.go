// loss.go
// Main neural network loss function code.

package nn

import (
	"math"
	"errors"
	"fmt"
)


func invalidLossSize(size int) error {
	return errors.New(fmt.Sprintf("Invalid loss input size: %d", size))
}


// Mean squared error loss struct.
type MeanSquaredLoss struct {
	Size int
}

// New mean squared loss function.
func NewMeanSquaredLoss(size int) (MeanSquaredLoss, error) {
	if size < 1 {
		// Invalid size.
		return MeanSquaredLoss{}, invalidLossSize(size)
	}

	// Return the new mean squared loss struct.
	return MeanSquaredLoss{size}, nil
}

// Mean squared loss forward pass function.
func (loss *MeanSquaredLoss) Forward(yhat Matrix, y Matrix) (float64, error) {
	// Check that all the dimensions match up.
	if yhat.Cols != loss.Size || y.Cols != loss.Size {
		// Find which matrix has incorrect dimensions and return an error.
		if yhat.Cols != loss.Size {
			return 0, invalidMatrixDimensionsError(yhat.Rows, yhat.Cols)
		} else if y.Cols != loss.Size {
			return 0, invalidMatrixDimensionsError(y.Rows, y.Cols)
		}
	}

	// Calculate the mean squared error (J = Σ[(yhat-y)^2]).
	sub, err := yhat.Sub(y)
	if err != nil {
		return 0, err
	}

	for i := 0; i < sub.Cols; i++ {
		sub.M[0][i] = math.Pow(sub.M[0][i], 2)
	}

	out := sub.Sum(1).M[0][0] / float64(sub.Cols)

	return out, nil
}

// Mean squared loss backward pass function. Outputs the gradients of the inputs. 
func (loss *MeanSquaredLoss) Backward(yhat Matrix, y Matrix) (Matrix, error) {
	// Check that all the dimensions match up.
	if yhat.Cols != loss.Size || y.Cols != loss.Size {
                // Find which matrix has incorrect dimensions and return an error.
                if yhat.Cols != loss.Size {
                        return Matrix{}, invalidMatrixDimensionsError(yhat.Rows, yhat.Cols)
                } else if y.Cols != loss.Size {
                        return Matrix{}, invalidMatrixDimensionsError(y.Rows, y.Cols)
                }
        }

	// Calculate the gradient of the mean squared error function.
	dInputs, err := yhat.Sub(y)
	if err != nil {
		return Matrix{}, err
	}
	dInputs = dInputs.MulScalar(float64(2) / float64(loss.Size))

	// Return the final gradient.
	return dInputs, nil
}


// Mean absolute error loss struct.
type MeanAbsoluteLoss struct {
        Size int
}

// New mean absolute loss function.
func NewMeanAbsoluteLoss(size int) (MeanAbsoluteLoss, error) {
        if size < 1 {
                // Invalid size.
                return MeanAbsoluteLoss{}, invalidLossSize(size)
        }

        // Return the new mean absolute loss struct.
        return MeanAbsoluteLoss{size}, nil
}

// Mean absolute loss forward pass function.
func (loss *MeanAbsoluteLoss) Forward(yhat Matrix, y Matrix) (float64, error) {
        // Check that all the dimensions match up.
        if yhat.Cols != loss.Size || y.Cols != loss.Size {
                // Find which matrix has incorrect dimensions and return an error.
                if yhat.Cols != loss.Size {
                        return 0, invalidMatrixDimensionsError(yhat.Rows, yhat.Cols)
                } else if y.Cols != loss.Size {
                        return 0, invalidMatrixDimensionsError(y.Rows, y.Cols)
                }
        }

        // Calculate the mean absolute error (J = Σ[|(yhat-y)|]).
        sub, err := yhat.Sub(y)
        if err != nil {
                return 0, err
        }

        for i := 0; i < sub.Cols; i++ {
                sub.M[0][i] = math.Abs(sub.M[0][i])
        }

        out := sub.Sum(1).M[0][0] / float64(sub.Cols)

        return out, nil
}

// Mean absolute loss backward pass function. Outputs the gradients of the inputs.
func (loss *MeanAbsoluteLoss) Backward(yhat Matrix, y Matrix) (Matrix, error) {
        // Check that all the dimensions match up.
        if yhat.Cols != loss.Size || y.Cols != loss.Size {
                // Find which matrix has incorrect dimensions and return an error.
                if yhat.Cols != loss.Size {
                        return Matrix{}, invalidMatrixDimensionsError(yhat.Rows, yhat.Cols)
                } else if y.Cols != loss.Size {
                        return Matrix{}, invalidMatrixDimensionsError(y.Rows, y.Cols)
                }
        }

        // Calculate the gradient of the mean absolute error function.
        dInputs, err := yhat.Sub(y)
        if err != nil {
                return Matrix{}, err
        }
	for i := 0; i < dInputs.Rows; i++ {
		for j := 0; j < dInputs.Cols; j++ {
			if dInputs.M[i][j] > 0 {
				dInputs.M[i][j] = float64(1/dInputs.Cols)
			} else if dInputs.M[i][j] < 0 {
				dInputs.M[i][j] = float64(-1/dInputs.Cols)
			} else {
				dInputs.M[i][j] = 0
			}
		}
	}

        // Return the final gradient.
        return dInputs, nil
}

// Cross-entropy loss struct.
type CrossEntropyLoss struct {
        Size int
}

// New cross-entropy loss function.
func NewCrossEntropyLoss(size int) (CrossEntropyLoss, error) {
        if size < 1 {
                // Invalid size.
                return CrossEntropyLoss{}, invalidLossSize(size)
        }

        // Return the new cross-entropy loss struct.
        return CrossEntropyLoss{size}, nil
}

// Cross-entropy loss forward pass function.
func (loss *CrossEntropyLoss) Forward(yhat Matrix, y Matrix) (float64, error) {
        // Check that all the dimensions match up.
        if yhat.Cols != loss.Size || y.Cols != loss.Size {
                // Find which matrix has incorrect dimensions and return an error.
                if yhat.Cols != loss.Size {
                        return 0, invalidMatrixDimensionsError(yhat.Rows, yhat.Cols)
                } else if y.Cols != loss.Size {
                        return 0, invalidMatrixDimensionsError(y.Rows, y.Cols)
                }
        }

        // Calculate the cross-entropy loss (J = -log(Σ[clip(yhat) * y])).
        _, err := yhat.Sub(y)
        if err != nil {
                return 0, err
        }

        //for i := 0; i < sub.Cols; i++ {
        //        sub.M[0][i] = math.Pow(sub.M[0][i], 2)
        //}

        //out := sub.Sum(1).M[0][0]

        return 0, nil
}

// Cross-entropy loss backward pass function.
// Note:
func (loss *CrossEntropyLoss) Backward(yhat Matrix, y Matrix) (Matrix, error) {
	return Matrix{}, nil
}


