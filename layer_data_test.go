// layer_data_test.go

package nn

import (
	"testing"
	"bytes"
	"fmt"
)


func TestLayerData(t *testing.T) {
	// Create a layer.
	l, _ := NewLayer(3, 5)
	l.Init()

	// Save the layer.
	data := NewSavedLayerData(&l)
	var buf = new(bytes.Buffer)
	data.SerializeLayer(buf)

	// Load the layer.
	layer, err := LoadLayer(buf)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	fmt.Printf("%v", layer.(*HiddenLayer).Weights)
}
