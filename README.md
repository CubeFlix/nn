# nn

A simple test neural network in Go.

The project is split into several files, each one responsible for a specific part of a neural network. `network.go` contains the main code for the layers. `loss.go` contains the code for different loss functions that can be used with the layers. `activation.go` does not contain any layers, but rather simple activation functions that act on matricies. `optimizers.go` implements basic optimizers. `layer_data.go` implements layer saving and loading functionality. Finally, `nn` uses a custom matrix system, implemented in `matrix.go`. The respective `_test.go` files contain simple tests for each file.

Currently (as of 9/3/22) the only full neural network test for `nn` is in `sine_test.go`, which models regression for a sine wave.

This project is INCOMPLETE.
-----

- [ ] test out categorization (iris dataset)
- [ ] test out categorization (spiral dataset, nnfs)
- [x] add softmax classification
- [x] add binary cross-entropy loss
- [x] add and test optimizers (~~sgd~~, ~~Adam~~, RMSProp)
- [ ] full model object
- [ ] add dropout
- [ ] add regularization
- [ ] try out 'Heavy' network classes (you know which one)
- [ ] try out LSTM and RNNs
