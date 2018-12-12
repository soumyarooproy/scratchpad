# Tutorial for adding a new tensor type in PyTorch

This tutorial walks the reader through the steps required to add a new tensor type in PyTorch. The approach described here uses [C++ extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html).

## Prerequites

Please familiarize yourself with the PyTorch [codebase structure](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md).

**Also, note that, at the time of writing this document, major implementation changes were underway and, therefore, the details below are most likely not going to work as they are.**

## Steps

### Basic Idea



**Note:** Add a case for `MyCustomFloat` type in `aten/src/ATen/DLConvertor.cpp` to avoid compiler warnings. Just start off by throwing an exception like the complex types in there

### Convert from `FloatTensor`

Implement `s_copy_()` for scalar type.

### Convert to `FloatTensor`

??

### Print Tensor

Implement `_local_dense_scalar()` for scalar type.

#### 1. Convert the tensor to `FloatTensor`

`torch/_tensor_str.py` houses the tensor printing code, which is invoked from the `Tensor` class defined in `torch/tensor.py`.

#### 2. Recommended

Look into `is_floating_point()` and `isFloatingPoint()` methods

### Transpose

Get it for free!

### Copy Tensor

```
t = torch.empty(2, 2, dtype=torch.dojofloat)
t_copy = t
```

### Arithmetic Ops

Need to define `promoteTypes`

### Broadcasting Ops

