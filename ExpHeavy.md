# Heavy Layers

Heavy layers are quadratic neural network layers that work in conjunction with standard Dense layers. While a standard feedforward hidden layer might look like this: ![equation](https://latex.codecogs.com/svg.image?\mathbf{\hat{Y}}&space;=&space;\sigma(\mathbf{XW}&space;&plus;&space;\vec{b})&space;), a quadratic "Heavy" layer would include a third, squared term with its own weight matrix.

## Forward Pass

A simple forward pass on a Heavy layer would function like this: ![equation](https://latex.codecogs.com/svg.image?\mathbf{\hat{Y}}&space;=&space;\sigma{}(\mathbf{X^2}\mathbf{H}&space;&plus;&space;\mathbf{XW&space;&plus;&space;\vec{b}})). The new **H** matrix would be of size inputs * outputs.

## Backward Pass

For the backward pass on a Heavy layer, the bias gradient stays the same, along with the weight gradient. The two new gradients that need to be calculated are the gradients on the inputs and the Heavy values. The gradients on the Heavy values are simple enough: ![equation](https://latex.codecogs.com/svg.image?\frac{\partial\mathbf{\hat{Y}}}{\partial\mathbf{H}}&space;=&space;\mathbf{X^2}&space;\cdot&space;\frac{\partial\mathbf{\hat{Y}}}{\left&space;[&space;\mathbf{X^2H&space;&plus;&space;XW&space;&plus;&space;\vec{b}}&space;\right&space;]}&space;) The gradients on the inputs prove tricker, but are still very similar to the standard backprop calculations: ![equation](https://latex.codecogs.com/svg.image?\frac{\partial\mathbf{\hat{Y}}}{\partial\mathbf{X}}&space;=&space;\frac{\partial\mathbf{\hat{Y}}}{\left&space;[&space;\mathbf{X^2H&space;&plus;&space;XW&space;&plus;&space;\vec{b}}&space;\right&space;]}&space;\cdot&space;\left&space;[&space;\mathbf{W^T&space;&plus;&space;2H^T&space;\circ&space;X}&space;\right&space;])