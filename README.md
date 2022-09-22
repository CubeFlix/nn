# nn

A simple neural network in Go.

`nn` allows users to easily and quickly create simple, yet powerful neural networks in Go. `nn` does not require any non-default packages, and runs on most systems. It uses a custom matrix engine and implements simple layers, activation and loss functions and optimizers, all easily packaged within a `Model` object. Simple examples can be found within the `examples` folder. `nn` is fully customizable and can be expanded to fit most modern networks. `nn` is reasonably fast and does not support GPU training.

The project is split into several files, each one responsible for a specific part of the network. However, the three main files are `network.go`, which contains the main code for the layers and activation functions. `matrix.go` contains the code for the custom matrix system. Finally, `model.go` contains code for `Model` objects and bundles the entire project together.

Currently (as of 9/19/22) the three full neural network examples are "sine_test.go", "spiral_test.go" and "iris_test.go".

This project is COMPLETE.
-----

- [x] binary categorical accuracy
- [x] test out categorization (iris dataset)
- [x] test out categorization (spiral dataset, nnfs, dropout)
- [x] saving and loading models
- [x] add softmax classification
- [x] add binary cross-entropy loss
- [x] add and test optimizers (~~sgd~~, ~~Adam~~, RMSProp)
- [x] full model object
- [x] add dropout
- [ ] try out 'Heavy' network classes (you know which one)
- [ ] try out LSTM and RNNs
