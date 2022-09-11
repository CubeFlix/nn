# Layer Data Implementation

Layer data buffer implementation details.

Each layer will be encoded as such:

| Name and value               | Size    | Type   |
| ---------------------------- | ------- | int    |
| Layer type                   | 1 byte  | int    |
| Input size                   | 4 bytes | int    |
| Output size                  | 4 bytes | int    |
| Slope (leakyRELU, optional)  | 8 bytes | float  |
| Weight values (rows by cols) | N bytes | floats |
| Bias values (rows by cols)   | N bytes | floats |

