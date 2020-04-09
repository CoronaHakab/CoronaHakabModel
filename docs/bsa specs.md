# The BSA file format
Taking inspiration from many (little used) sparse matrix formats. The BSA (Binary Sparse Array) file format is used to store **square, sparse** matrices of arbitrary size.
## Restrictions:
* A matrix's size cannot exceed 2^32 - 1
# Format
## header
* 2 bytes: magic string: SA [83, 65]
* 1 byte: version specifier: currently 0
* ? bytes: user-data (for user usage, will continue until a byte is 0)
* 1 byte: flags (currently unused, 0)
* 4 bytes: matrix side size (called N from here on)
## rows
* 4*N bytes: for each row, the respective int32 will describe how many non-default values it has.
## columns
* 4*X bytes: Each row will have ints describing the index of the columns of non-default values. The column indices will be sorted.
## Values
* 1 byte: data type (see below for valid data types)
* ? bytes: user-data (for user usage, will continue until a byte is 0)
* ? bytes: default value (for this version only 0 is accepted)
* for each row and column described, a number of bytes as described by the data type are used to describe the value of the cell.

# Valid data values
* 1: 8-bit integers
* 2: 16-bit integers
* 3: 32-bit integers
* 4: 64-bit integers
* 11: 32-bit floating values
* 12: 64-bit floating values
* 251: arbitrary 8-bit value (used defined)
* 252: arbitrary 16-bit value (used defined)
* 253: arbitrary 32-bit value (used defined)
* 254: arbitrary 64-bit value (used defined)

during development, only 32-bit floating values will be accepted

# Example
the integer matrix
```
0 1 3 0 8
1 0 2 0 0
2 0 3 1 1
0 0 0 0 0
1 2 4 0 0
```
will be saved as the following sequential bytes:
```
83, 65 - magic
0 - version
0 - UD
0 - flags
0, 0, 0, 5 - size
0,0,0,3, 0,0,0,2, 0,0,0,4, 0,0,0,0, 0,0,0,3 - row lengths
0,0,0,1, 0,0,0,2, 0,0,0,8
0,0,0,0, 0,0,0,3
0,0,0,0, 0,0,0,2, 0,0,0,3, 0,0,0,4

0,0,0,0, 0,0,0,1, 0,0,0,2 - columns
1 - integer data type
0 - UD
0 - default value
1, 3, 8
1, 2
2, 3, 1, 1

1, 2, 4 - values
```

# Size
size of a matrix of size N with T non-default values, an data type of length B is:
* header: 9
* rows: 4*N
* columns: 4*T
* values: 2 + B + B*T

in total: `11 + B + 4*N + 4*T + B*T`
