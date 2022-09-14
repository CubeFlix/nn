# Model Data Implementation

Model data buffer implementation details.

Each model will be encoded as such:

| Name and value               | Size    | Type   |
| ---------------------------- | ------- | ------ |
| Magic bytes                  | 4 bytes | string |
| Version                      | 5 bytes | string |
| Model size                   | 1 byte  | int    |
| Input size                   | 4 bytes | int    |
| Output size                  | 4 bytes | int    |
| Loss type                    | 1 byte  | int    |
| Accuracy type                | 1 byte  | int    |
| Accuracy percision           | 8 bytes | float  |
| Optimizer type               | 1 byte  | int    |
| Optimizer values             | N bytes | custom |
| Layers                       | N bytes | custom |

Optimizer values will be encoded as such:

| Name and value               | Size    | Type   |
| ---------------------------- | ------- | ------ |
| Length                       | 1 byte  | int    |
| Length of key 1              | 1 byte  | int    |
| Key 1                        | N bytes | int    |
| Value 1                      | 8 bytes | float  |
| ...                          | ...     | ...    |

