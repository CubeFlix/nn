# nn

A simple neural network in Go.

`nn` allows users to easily and quickly create simple, yet powerful neural networks in Go. `nn` does not require any non-default packages, and runs on most systems. It uses a custom matrix engine and implements simple layers, activation and loss functions and optimizers, all easily packaged within a `Model` object. Simple examples can be found within the `examples` folder. `nn` is fully customizable and can be expanded to fit most modern networks. `nn` is reasonably fast and does not support GPU training.

The project is split into several files, each one responsible for a specific part of the network. However, the three main files are `network.go`, which contains the main code for the layers and activation functions. `matrix.go` contains the code for the custom matrix system. Finally, `model.go` contains code for `Model` objects and bundles the entire project together.

Currently (as of 9/3/22) the only full neural network test for `nn` is in `sine_test.go`, which models regression for a sine wave.

This project is INCOMPLETE, but in a usable state.
-----

- [x] binary cross-entropy accuracy
- [ ] test out categorization (iris dataset)
- [ ] test out categorization (spiral dataset, nnfs)
- [x] saving and loading models
- [x] add softmax classification
- [x] add binary cross-entropy loss
- [x] add and test optimizers (~~sgd~~, ~~Adam~~, RMSProp)
- [x] full model object
- [ ] add dropout
- [ ] add regularization
- [ ] try out 'Heavy' network classes (you know which one)
- [ ] try out LSTM and RNNs
