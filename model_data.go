// model_data.go
// Saving and loading models as data.

package nn


import (
        "bytes"
        "encoding/binary"
	"errors"
	"os"
)


// Saved model data struct.
type SavedModelData struct {
        Version           string
	ModelSize         int
	InputSize         int
	OutputSize        int
	LossType          LossType
	AccuracyType      AccuracyType
	AccuracyPercision float64
	OptimizerType     OptimizerType
	OptimizerValues   map[string]float64
	Layers            []SavedLayerData
}

// Create a new SavedModelData object from a layer.
func NewSavedModelData(model Model) SavedModelData {
        // Get the saved layer data objects.
	layers := []SavedLayerData{}
	for i := 0; i < model.ModelSize; i++ {
		layers = append(layers, NewSavedLayerData(model.Layers[i]))
	}

        // Return the new saved layer data object
        return SavedModelData{
                Version:           VERSION,
		ModelSize:         model.ModelSize,
		InputSize:         model.InputSize,
		OutputSize:        model.OutputSize,
		LossType:          model.LossType,
		AccuracyType:      model.AccuracyType,
		AccuracyPercision: model.AccuracyPercision,
		OptimizerType:     model.OptimizerType,
		OptimizerValues:   model.OptimizerValues,
		Layers:            layers,
        }
}

// Serialize the model into a buffer. NOTE: Model optimizer caches will not be saved or loaded.
func (m *SavedModelData) Serialize(buf *bytes.Buffer) error {
	// Write the magic bytes.
        buf.WriteString("NNML")

	// Write the version to the buffer.
        buf.WriteString(m.Version)

	// Write the model size to the buffer.
	err := binary.Write(buf, binary.LittleEndian, int8(m.ModelSize))
        if err != nil {
                return err
        }

	// Write the input and output sizes into the buffer.
        err = binary.Write(buf, binary.LittleEndian, int32(m.InputSize))
        if err != nil {
                return err
        }
        err = binary.Write(buf, binary.LittleEndian, int32(m.OutputSize))
        if err != nil {
                return err
        }

	// Write the loss type to the buffer.
        err = binary.Write(buf, binary.LittleEndian, int8(m.LossType))
        if err != nil {
                return err
        }

	// Write the accuracy type and percision to the buffer.
        err = binary.Write(buf, binary.LittleEndian, int8(m.AccuracyType))
        if err != nil {
                return err
        }
        err = binary.Write(buf, binary.LittleEndian, m.AccuracyPercision)
        if err != nil {
                return err
        }

	// Write the optimizer type to the buffer.
	err = binary.Write(buf, binary.LittleEndian, int8(m.OptimizerType))
        if err != nil {
                return err
        }

	// Write the optimizer values.
	err = binary.Write(buf, binary.LittleEndian, int8(len(m.OptimizerValues)))
        if err != nil {
                return err
        }

	// Loop over the optimizer values and write each key-value pair.
	for k, v := range m.OptimizerValues {
		// Write the length of the key.
		err = binary.Write(buf, binary.LittleEndian, int8(len(k)))
	        if err != nil {
	                return err
		}

		// Write the key.
		buf.WriteString(k)

		// Write the value.
		err = binary.Write(buf, binary.LittleEndian, v)
                if err != nil {
                        return err
                }
	}

	// Write each layer to the buffer.
	for i := 0; i < m.ModelSize; i++ {
		err = m.Layers[i].SerializeLayer(buf)
		if err != nil {
			return err
		}
	}

	return nil
}


// Return a loss object.
func loadLoss(lossType LossType, size int) (Loss, error) {
	switch lossType {
                case MeanSquaredLossType:
                        return &MeanSquaredLoss{
                                Size: size,
                        }, nil
		case MeanAbsoluteLossType:
                        return &MeanAbsoluteLoss{
                                Size: size,
                        }, nil
		case CrossEntropyLossType:
                        return &CrossEntropyLoss{
                                Size: size,
                        }, nil
		case BinaryCrossEntropyLossType:
                        return &BinaryCrossEntropyLoss{
                                Size: size,
                        }, nil
		default:
			return nil, errors.New("nn.LoadModel: Invalid loss type.")
	}
}


// Return a optimizer object.
func loadOptimizer(optimizerType OptimizerType, values map[string]float64) (Optimizer, error) {
	switch optimizerType {
		case SGDOptimizerType:
			o := SGDOptimizer{}
			o.setValues(values)
			return &o, nil
		case AdamOptimizerType:
			o := AdamOptimizer{}
			o.setValues(values)
			return &o, nil
		default:
			return nil, errors.New("nn.LoadModel: Invalid optimizer type.")
	}
}


// Load a model buffer into a saved model data object. The layers will be saved into a seperate slice.
func loadModelBuffer(buf *bytes.Buffer) (SavedModelData, []Layer, error) {
        // Read the magic bytes.
        magic := make([]byte, 4)
        _, err := buf.Read(magic)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }
        if string(magic) != "NNML" {
                // Invalid magic bytes.
                return SavedModelData{}, []Layer{}, errors.New("nn.LoadModel: Invalid magic bytes. Check that the data is not corrupted.")
        }

	// Read the model version.
        version := make([]byte, 5)
        _, err = buf.Read(version)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }
        if string(version) != VERSION {
                // Different version info.
		WarningLogger.Printf("Model version %s may be incompatable with nn version %s.", string(version), VERSION)
        }

        // Read the model size.
        var modelSize int8
        err = binary.Read(buf, binary.LittleEndian, &modelSize)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }

	// Read the input and output sizes.
	var inputSize, outputSize int32
        err = binary.Read(buf, binary.LittleEndian, &inputSize)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }
	err = binary.Read(buf, binary.LittleEndian, &outputSize)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }

	// Read the loss type.
	var lossType int8
	err = binary.Read(buf, binary.LittleEndian, &lossType)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }

	// Read the accuracy type and percision.
        var accuracyType int8
        err = binary.Read(buf, binary.LittleEndian, &accuracyType)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }
	var accuracyPercision float64
        err = binary.Read(buf, binary.LittleEndian, &accuracyPercision)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }

	// Read the optimizer type.
	var optimizerType int8
        err = binary.Read(buf, binary.LittleEndian, &optimizerType)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }

	// Read the optimizer values.
	optimizerValues := make(map[string]float64)

	// Read the length of the values.
	var lenOptimizerValues int8
	err = binary.Read(buf, binary.LittleEndian, &lenOptimizerValues)
        if err != nil {
                return SavedModelData{}, []Layer{}, err
        }

	// Loop over all the values.
	for i := 0; i < int(lenOptimizerValues); i++ {
		// Read the key length.
		var lenKey int8
	        err = binary.Read(buf, binary.LittleEndian, &lenKey)
	        if err != nil {
	                return SavedModelData{}, []Layer{}, err
		}

		// Read the key.
		key := make([]byte, int(lenKey))
	        _, err := buf.Read(key)
	        if err != nil {
	                return SavedModelData{}, []Layer{}, err
	        }

		// Read the value.
		var value float64
                err = binary.Read(buf, binary.LittleEndian, &value)
                if err != nil {
                        return SavedModelData{}, []Layer{}, err
                }

		// Set the value.
		optimizerValues[string(key)] = value
	}

	// Read all the layers.
	layers := []Layer{}

	// Loop over all the layers.
	for i := 0; i < int(modelSize); i++ {
		// Load the layer.
		layer, err := LoadLayer(buf)
		if err != nil {
			return SavedModelData{}, []Layer{}, err
		}

		// Add the layer.
		layers = append(layers, layer)
	}

	// Return the new saved model data object.
        return SavedModelData{
		Version:           string(version),
		ModelSize:         int(modelSize),
		InputSize:         int(inputSize),
		OutputSize:        int(outputSize),
		LossType:          LossType(lossType),
		AccuracyType:      AccuracyType(accuracyType),
		AccuracyPercision: accuracyPercision,
		OptimizerType:     OptimizerType(optimizerType),
		OptimizerValues:   optimizerValues,
        }, layers, nil
}

// Load a model as a buffer and return a model interface object. NOTE: Model optimizer caches will not be saved or loaded.
func LoadModel(buf *bytes.Buffer) (Model, error) {
        // Load the buffer as a saved model data object.
        savedModelData, layers, err := loadModelBuffer(buf)
        if err != nil {
                return Model{}, err
        }

	// Create the new model object.
	model := NewModel()

	// Add the layers.
	for i := 0; i < savedModelData.ModelSize; i++ {
		err := model.AddLayer(layers[i])
		if err != nil {
			return Model{}, err
		}
	}

	// Create a loss and optimizer object.
	loss, err := loadLoss(savedModelData.LossType, savedModelData.OutputSize)
	if err != nil {
		return Model{}, err
	}
	optimizer, err := loadOptimizer(savedModelData.OptimizerType, savedModelData.OptimizerValues)
	if err != nil {
		return Model{}, err
	}

	// Finalize the model.
	err = model.Finalize(loss, optimizer, savedModelData.AccuracyType, savedModelData.AccuracyPercision)
	if err != nil {
		return Model{}, err
	}

	// Return the finished model.
	return model, nil
}


// Save a model to a file.
func SaveFile(model *Model, filename string) error {
	// Get the saved model data.
	data := NewSavedModelData(*model)

	// Open the file.
        file, err := os.Create(filename)
        if err != nil {
                return err
        }
        defer file.Close()

	// Save the model data to a buffer.
        var buffer bytes.Buffer
        err = data.Serialize(&buffer)
        if err != nil {
                return err
        }

	// Write the buffer to the file.
        _, err = file.Write(buffer.Bytes())
        if err != nil {
                return err
        }

	return nil
}

// Load a model from a file.
func LoadFile(filename string) (Model, error) {
        // Open the file.
        file, err := os.Open(filename)
        if err != nil {
                return Model{}, err
        }
        defer file.Close()

	// Get the file size and create a new buffer.
	stat, err := file.Stat()
	if err != nil {
		return Model{}, err
	}
	buffer := make([]byte, stat.Size())

        // Read the file into the buffer.
	_, err = file.Read(buffer)
	if err != nil {
		return Model{}, err
	}

	// Create a bytes.Buffer object and load it.
	buf := bytes.NewBuffer(buffer)
	return LoadModel(buf)
}
