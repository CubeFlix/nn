// layer_data.go
// Saving and loading layers as data.

package nn


import (
	"bytes"
	"encoding/binary"
	"errors"
)


// Layer type type definition.
type LayerType int8

// Layer types and codes.
const (
	HiddenLayerType  LayerType = 0
	LinearLayerType            = 1
	SigmoidLayerType           = 2
	LeakyLayerType             = 3
	SoftmaxLayerType           = 4
)


// Saved layer data struct.
type SavedLayerData struct {
	Type    LayerType
	Inputs  int
	Outputs int
	Slope   float64
	Weights Matrix
	Biases  Matrix
}

// Create a new savedLayerData object from a layer.
func NewSavedLayerData(layer Layer) SavedLayerData {
	// Get the values from the layer interface.
	weights, biases, values := layer.getValues()

	// Return the new saved layer data object
	return SavedLayerData {
		Type:    LayerType(values["type"]),
		Inputs:  int(values["inputs"]),
		Outputs: int(values["outputs"]),
		Slope:   values["slope"],
		Weights: *weights,
		Biases:  *biases,
	}
}

// Serialize the layer into a buffer.
func (l *SavedLayerData) SerializeLayer(buf *bytes.Buffer) error {
	// Check that the matrix sizes are correct.
	if l.Weights.Rows != l.Inputs || l.Weights.Cols != l.Outputs {
		return invalidLayerDimensionsError(l.Inputs, l.Outputs)
	}
	if l.Biases.Rows != 1 || l.Biases.Cols != l.Outputs {
                return invalidLayerDimensionsError(l.Inputs, l.Outputs)
        }

	// Write the magic bytes.
	buf.WriteString("LA")

	// Write the layer type to the buffer.
	err := binary.Write(buf, binary.LittleEndian, l.Type)
	if err != nil {
		return err
	}

	// Write the input and output sizes into the buffer.
	err = binary.Write(buf, binary.LittleEndian, int32(l.Inputs))
	if err != nil {
                return err
        }
	err = binary.Write(buf, binary.LittleEndian, int32(l.Outputs))
	if err != nil {
                return err
        }

	// Write the optional slope into the buffer.
	err = binary.Write(buf, binary.LittleEndian, l.Slope)
	if err != nil {
                return err
        }

	// Write the weights into the buffer.
	for i := 0; i < l.Weights.Rows; i++ {
		for j := 0; j < l.Weights.Cols; j++ {
			err = binary.Write(buf, binary.LittleEndian, l.Weights.M[i][j])
			if err != nil {
				return err
			}
		}
	}

	// Write the biases into the buffer.
	for i := 0; i < l.Biases.Rows; i++ {
                for j := 0; j < l.Biases.Cols; j++ {
                        err = binary.Write(buf, binary.LittleEndian, l.Biases.M[i][j])
                        if err != nil {
                                return err
                        }
                }
        }

	return nil
}


// Load a layer buffer into a saved layer data object.
func loadLayerBuffer(buf *bytes.Buffer) (SavedLayerData, error) {
	// Read the magic bytes.
	magic := make([]byte, 2)
	_, err := buf.Read(magic)
	if err != nil {
		return SavedLayerData{}, err
	}
	if string(magic) != "LA" {
		// Invalid magic bytes.
		return SavedLayerData{}, errors.New("nn.LoadLayer: Invalid magic bytes. Check that the data is not corrupted.")
	}

	// Read the layer type.
	var layerType int8
	err = binary.Read(buf, binary.LittleEndian, &layerType)
	if err != nil {
		return SavedLayerData{}, err
	}

	// Read the input and output sizes.
	var inputSize, outputSize int32
	err = binary.Read(buf, binary.LittleEndian, &inputSize)
        if err != nil {
                return SavedLayerData{}, err
        }
	err = binary.Read(buf, binary.LittleEndian, &outputSize)
        if err != nil {
                return SavedLayerData{}, err
        }

	// Read the optional slope value.
	var slope float64
        err = binary.Read(buf, binary.LittleEndian, &slope)
        if err != nil {
                return SavedLayerData{}, err
        }

	// Read the weight matrix.
	weights, err := NewMatrix(int(inputSize), int(outputSize))
	if err != nil {
		return SavedLayerData{}, err
	}
	for i := 0; i < int(inputSize); i++ {
                for j := 0; j < int(outputSize); j++ {
                        err = binary.Read(buf, binary.LittleEndian, &weights.M[i][j])
                        if err != nil {
                                return SavedLayerData{}, err
                        }
                }
        }

	// Read the bias matrix.
        biases, err := NewMatrix(1, int(outputSize))
        if err != nil {
                return SavedLayerData{}, err
        }
        for j := 0; j < int(outputSize); j++ {
                err = binary.Read(buf, binary.LittleEndian, &biases.M[0][j])
                if err != nil {
                        return SavedLayerData{}, err
                }
        }

	// Return the new saved layer data object.
	return SavedLayerData{
		Type:    LayerType(layerType),
		Inputs:  int(inputSize),
		Outputs: int(outputSize),
		Slope:   slope,
		Weights: weights,
		Biases:  biases,
	}, nil
}

// Load a layer as a buffer and return a layer interface object.
func LoadLayer(buf *bytes.Buffer) (Layer, error) {
	// Load the buffer as a saved layer data object.
	savedLayerData, err := loadLayerBuffer(buf)
	if err != nil {
		return nil, err
	}

	// Switch over the type value and return the proper layer.
	switch savedLayerData.Type {
		case HiddenLayerType:
			return &HiddenLayer{
				InputSize:  savedLayerData.Inputs,
				OutputSize: savedLayerData.Outputs,
				Weights:    &savedLayerData.Weights,
				Biases:     &savedLayerData.Biases,
			}, nil
		case LinearLayerType:
                        return &LinearLayer{
                                InputSize:  savedLayerData.Inputs,
                                OutputSize: savedLayerData.Outputs,
                                Weights:    &savedLayerData.Weights,
                                Biases:     &savedLayerData.Biases,
                        }, nil
		case SigmoidLayerType:
                        return &SigmoidLayer{
                                InputSize:  savedLayerData.Inputs,
                                OutputSize: savedLayerData.Outputs,
                                Weights:    &savedLayerData.Weights,
                                Biases:     &savedLayerData.Biases,
                        }, nil
		case LeakyLayerType:
                        return &LeakyLayer{
                                InputSize:  savedLayerData.Inputs,
                                OutputSize: savedLayerData.Outputs,
				Slope:      savedLayerData.Slope,
                                Weights:    &savedLayerData.Weights,
                                Biases:     &savedLayerData.Biases,
                        }, nil
		case SoftmaxLayerType:
                        return &SoftmaxLayer{
                                InputSize:  savedLayerData.Inputs,
                                OutputSize: savedLayerData.Outputs,
                                Weights:    &savedLayerData.Weights,
                                Biases:     &savedLayerData.Biases,
                        }, nil
		default:
			return nil, errors.New("nn.LoadLayer: Invalid layer type value.")
	}
}
