# Tutorial for adding a new tensor type in PyTorch (on the CPU)

This tutorial walks the reader through the steps required to add a new tensor type (`bfloat16` used in this write-up) in PyTorch (only on the CPU). The approach described here uses [C++ extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html).

The goal of this document is to provide hardware/system researchers/engineers who are new to PyTorch a reference quide for some of the lower-level implementation details and APIs in PyTorch so that they can extend PyTorch on custom AI hardware/accelerators.

**Note that the steps/suggestions made in this write-up are merely for prototyping and a significant effort is expected to get it working in production.**

## Prerequisites

1. Please familiarize yourself with the PyTorch [codebase structure](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md).
2. This [complex type tensor c++ extension test](https://github.com/pytorch/pytorch/blob/master/test/cpp_extensions/complex_registration_extension.cpp).

**Also, note that, at the time of writing this document, fairly significant PyTorch implementation changes were underway and, therefore, the details below are most likely not going to work exactly as is.**

## Steps

### Basic Approach

The approach that I took in this exercise (based on which I am writing this document) is to make minimally invasive changes to the PyTorch repo and have most of the business logic reside as part of C++ extension code. This enables quick turnaround times for experimentation.

### PyTorch Changes

Following are the primary changes needed in PyTorch repo to enable a `bfloat16` tensor type with very limited functionality:
1. Define `bfloat16` data type
1. Register new type with ATen and Caffe2
1. Add hooks for C++ extension
1. Expose `torch.bfloat16` data type in python

Later in this document, I also outline more changes that are required to add more functionality, such as copying/converting tensors and printing them, to this new tensor type.

#### 1. Define `bfloat16` Scalar Type

I added a minimal `bfloat16` definition, which basically allows converting back and forth from `float` types. [TensorFlow's `bfloat16`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/lib/bfloat16) is a good reference for a CPU emulation library.

- [`c10/BFloat16.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-8a987bb88e24e1f4a301df4713f783f1)
- [`aten/src/ATen/core/BFloat16.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-de8e37b5f778862142d3a98f68cdec40)

#### 2. Register `bfloat16` Type with ATen and Caffe2

First, add an `enum` type for `bfloat16` on a `CPU` device in ATen:
- [`aten/src/ATen/core/Type.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-7d11e981ef25d5bdd9c1b2f6411766f0)
- [`aten/src/ATen/templates/Type.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-09397284acffdfa65bc13710f6879cb4)

Do the same in Caffe2:
- [`c10/util/typeid.cpp`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-1373c2f6d7c97af88540dc8ec1f95841)

Next, add `Scalar` object constructor from `bfloat16` and also conversion from `Scalar` to `bfloat16`:
- [`c10/core/Scalar.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-aabe1c134055c8dcefad830c1c7ae957)

Next, add casting support for Tensor objects:
- [`aten/src/ATen/core/TensorMethods.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-766e5a8823dafb1986062d9dd8093796)
- [`aten/src/ATen/templates/TensorMethods.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-0e8bf6f5024b32c240a4c1f0b4d8fd71)

Finally, add a few functions that IIANM are to sync between ATen and Caffe2 worlds:
- [`c10/core/ScalarType.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-803e59d78f99f0d3cf9db6153e800dca)

#### 3. Add C++ Extension Hooks

These set of changes were a little confusing to me. The reason for that, I gather, is the fact that even though any new implementation should be part of `c10` code, they also need to support some legacy torch APIs and some existing caffe2 APIs due to backward compatibility reasons. This should get cleaned up in the future.

For this step, I followed the [Complex Hooks Interface](https://github.com/pytorch/pytorch/aten/src/ATen/detail/ComplexHooksInterface.h) and created a similar interface for `bfloat16` and enabled deferred (or lazy) registration of the type in the `CPU` device context.

The files added/modified for this are as follows:

- [`aten/src/ATen/detail/BFloat16HooksInterface.cpp`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-c5f5a31a6657a988932a2e3c5ab93688)
- [`aten/src/ATen/detail/BFloat16HooksInterface.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-6d39b0a1c1f7e0d1bdb06475090a1ec3)
- [`aten/src/ATen/Context.cpp`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-442cbbb4fd838ed005dcf0c92bea1020)
- [`aten/src/ATen/Context.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-28f4a2d0ea28112f2657b70c5e5e9b03)
- [`aten/src/ATen/core/LegacyTypeDispatch.h`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-f0dfeba69dac93bde8a36595318a273d)

#### 4. Expose `bfloat16` Datatype in Python

This is a tiny and self-explanatory change:
- [`torch/csrc/utils/tensor_dtypes.cpp`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-3f4c348238c99c09097e3719772b5af8)

#### 5. Other Changes

Handle `bfloat16` scalar type in [`aten/src/ATen/DLConvertor.cpp`](https://github.com/soumyarooproy/pytorch/pull/1/files#diff-6d515c35e38e894fd2253074921ed411) to avoid compiler warnings.

### C++ Extension for `bfloat16` Tensor

With the PyTorch changes described above, simply copying [complex type tensor c++ extension source](https://github.com/pytorch/pytorch/blob/master/test/cpp_extensions/complex_registration_extension.cpp) and replacing the occurrences of complex types with `bfloat16` types will enable creation of empty `bfloat16` tensors.

``` python
>>> b = torch.empty(2,2,dtype=torch.bfloat16)
>>> b.dtype
torch.bfloat16
>>> b.size()
torch.Size([2, 2])
>>> b
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: _copy__cpu not implemented for 'CPUBFloat16Type'
```

However, attempting to print a `bfloat16` tensor results in an error. Similarly, attempting to convert to or from a `float` tensor will also result in errors. In the rest of this tutorial, I describe how to add more functionality to `bfloat16` tensor type.

#### 1. Convert `torch.float` Tensor to `torch.bfloat16` Tensor

The straighforward way to achieve this is to override the `s_copy_()` method for `CPUBFloat16Type`. This bypasses the dynamic dispatch copy logic resident in `native/Copy.[h,cpp]`, which at the time of writing this document was located in `aten/src/ATen`.

```c++
Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override {
    if (src.type().scalarType() != at::ScalarType::Float)
        THError("s_copy_() for BFloat16 implemented only for float");

    using self_T = at::BFloat16;
    using src_T = float;
    at::CPU_tensor_apply2<self_T, src_T>(
      self, src, [](self_T& self_val, const src_T& src_val) {
        self_val = src_val;
      });
    return self;
}
```

The above change will make following statements operational:

```python
>>> f = torch.randn(1, 2)
>>> f.dtype
torch.float32
>>> b = f.to(dtype=torch.bfloat16)
>>> b.dtype
torch.bfloat16
```

#### 2. Convert `torch.bfloat16` Tensor to `torch.float` Tensor

Supporting this is more invasive in that the change probably cannot merely reside as part of the C++ Extension but has to be made in the PyTorch codebase. This is because the `float` tensor's copy method(s) need to be made aware of `bfloat16` type and the appropriate conversion semantics. Since the auto-generated code for `CPUFloatType` delegates the copy logic to the dynamic dispatch copy logic resident in `native/Copy.[h,cpp]`, one option is to add supporting code to `_s_copy__cpu()` function.

```c++
...
  if (self.type().scalarType() == at::ScalarType::Float
      && src.type().scalarType() == at::ScalarType::BFloat16) {
    using self_T = float;
    using src_T = at::BFloat16;
    at::CPU_tensor_apply2<self_T, src_T>(
      self, src, [](self_T& self_val, const src_T& src_val) {
        self_val = src_val;
      });
    return self;
  }
...
```

The above change will make following statements operational:

```python
...
>>> b.dtype
torch.bfloat16
>>> f = b.float()
>>> f.dtype
torch.float32
```

#### 3. Print `bfloat16` Tensor

From what I can tell, printing a tensor invokes the [`item()` method](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Scalar.cpp#L7) for each tensor element, which subsequently calls the `_local_dense_scalar()` for the scalar type. Therefore, overriding that method for `CPUBFloat16Type`
will suffice.

1. Add appropriate hooks in the `__str__` method for `torch.tensor` in python. `torch/_tensor_str.py` houses the tensor printing code, which is invoked from the `Tensor` class defined in `torch/tensor.py`.

**Note:** A cursory look at the code around printing suggests that including `bfloat16` type in the `is_floating_point()` and/or `isFloatingPoint()` methods may take care of printing out of the box. But this needs to be explored and confirmed.

#### 4. Matrix Multiplication

In this example, I define a simple serial matrix-multiplication implementation which is restricted to only contiguous `bfloat16` matrices. Therefore, for instance, `x.mm(x.t())` is not supported because `t()` transpose method does not change the layout of the tensor but it merely alters a tensor property to record that the tensor has been transposed.

Another important thing to note is that the accumulation in this matrix-multiplication kernel is performed in `float` even though the tensor elements are in `bfloat16`. The final accumulated result is converted back to `bfloat16` to generate a `bfloat16` output tensor. Refer to [`MatMulOp::Compute()` in TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/matmul_op.cc) for details.

This method also requires that the `resize_()` method be defined for `bfloat16` tensor.

Refer to [`addmm()` function](https://github.com/pytorch/pytorch/blob/master/aten/src/TH/generic/THTensorMath.cpp) for context.

```c++
Tensor mm(const Tensor & self, const Tensor & mat2) const override {
    if (self.is_sparse())
        THError("Operation on sparse tensor type not supported");

    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
                       CPUTensorId(), caffe2::TypeMeta::Make<at::BFloat16>(),
                       allocator(), false).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    result.resize_({ self.size(0), mat2.size(1) });
    auto self_ = checked_tensor_unwrap(self,"self",1, false, Backend::CPU, ScalarType::BFloat16);
    auto mat2_ = checked_tensor_unwrap(mat2,"mat2",2, false, Backend::CPU, ScalarType::BFloat16);

    if (self_->dim() != 2 || mat2_->dim() != 2)
        THError("matrices expected, got %dD, %dD tensors", self_->dim(), mat2_->dim());

    if (self_->size(1) != mat2_->size(0))
        THError("size mismatch");

    // Throw error if non-contiguous matrices
    if (self_->stride(1) != 1 || mat2_->stride(1) != 1)
        THError("mm() currently supported for row-major contiguous matrices only");

    // Multiply matrices of shapes (M, N) and (N, P)
    auto M = self_->size(0);
    auto N = self_->size(1);
    auto P = mat2_->size(1);

    // Get pointers to the tensor data elements
    auto m1_p = self_->data<at::BFloat16>();
    auto m2_p = mat2_->data<at::BFloat16>();
    auto r_p = result_->data<at::BFloat16>();

    // Standard O(n**3) matrix multiplication
    for (auto i = 0; i < M; ++i) {
        for (auto j = 0; j < P; ++j) {
            // Accumulate sum in float
            float sum = 0;
            for (auto k = 0; k < N; ++k) {
                sum += static_cast<float>(m1_p[i*N + k]) * static_cast<float>(m2_p[k*P + j]);
            }
            // Quantize sum to BFloat16
            r_p[i*P + j] = sum;
        }
    }

    result_->maybe_zero_dim(self_->dim() == 0 && mat2_->dim() == 0);
    return result;
}
```
