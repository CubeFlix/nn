// layer_data_test.go
// Layer data test.

package nn

import (
	"testing"
	"bytes"
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
	t.Logf("%v", layer.(*HiddenLayer).Weights)
}
