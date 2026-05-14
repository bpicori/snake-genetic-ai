#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
  size_t ndim;
  size_t element_count; /* product of dims, cached at creation */
  size_t* dims;
  size_t* strides;
  float* data;
} Tensor;

/* ------------------------------------------------------------------ */
/* Creation / destruction                                              */
/* ------------------------------------------------------------------ */

/* Allocate a zero-filled tensor with the given shape.
     size_t dims[] = {2, 3};
     Tensor* t = tensor_create(2, dims);
     t = [[0, 0, 0], [0, 0, 0]]                                       */
Tensor* tensor_create(size_t ndim, const size_t* dims);

/* Allocate a tensor with values uniform in [0, 1).
     Tensor* t = tensor_rand(1, (size_t[]){4});
     t ≈ [0.39, 0.84, 0.21, 0.93]                                     */
Tensor* tensor_rand(size_t ndim, const size_t* dims);

/* Allocate a tensor and copy `data` into it (row-major).
     float v[] = {1, 2, 3, 4};
     Tensor* t = tensor_from_array(2, (size_t[]){2, 2}, v);
     t = [[1, 2], [3, 4]]                                             */
Tensor* tensor_from_array(size_t ndim, const size_t* dims, const float* data);

/* Deep-copy a tensor. Mutating the copy does not affect the original.
     Tensor* b = tensor_clone(a);                                     */
Tensor* tensor_clone(const Tensor* t);

/* Free a tensor and all its owned allocations.
     tensor_destroy(t);                                               */
void tensor_destroy(Tensor* t);

/* ------------------------------------------------------------------ */
/* Indexing / shape                                                    */
/* ------------------------------------------------------------------ */

/* Read one element by multi-index.
     t = [[1, 2], [3, 4]]
     tensor_get(t, (size_t[]){1, 0}) = 3.0                            */
float tensor_get(const Tensor* t, const size_t* idx);

/* Write one element by multi-index.
     t = [[1, 2], [3, 4]]
     tensor_set(t, (size_t[]){1, 0}, 9.0)
     -> t = [[1, 2], [9, 4]]                                          */
void tensor_set(Tensor* t, const size_t* idx, float v);

/* Reshape a tensor (not yet implemented). */
Tensor* tensor_reshape(const Tensor* t, size_t ndim, const size_t* dims);

/* ------------------------------------------------------------------ */
/* Element-wise math (same-shape, no broadcasting)                     */
/* ------------------------------------------------------------------ */

/* c[i] = a[i] + b[i].
     [[1, 2], [3, 4]] + [[10, 20], [30, 40]] = [[11, 22], [33, 44]]   */
Tensor* tensor_add(const Tensor* a, const Tensor* b);

/* c[i] = a[i] - b[i].
     [[10, 20], [30, 40]] - [[1, 2], [3, 4]] = [[9, 18], [27, 36]]    */
Tensor* tensor_sub(const Tensor* a, const Tensor* b);

/* c[i] = a[i] * b[i]  (Hadamard product, not matmul).
     [[1, 2], [3, 4]] * [[10, 20], [30, 40]] = [[10, 40], [90, 160]]  */
Tensor* tensor_mul(const Tensor* a, const Tensor* b);

/* c[i] = a[i] * s.
     [[1, 2], [3, 4]] * 2.0 = [[2, 4], [6, 8]]                        */
Tensor* tensor_scalar_mul(const Tensor* a, float s);

/* c[i] = a[i] + s.
     [[1, 2], [3, 4]] + 5.0 = [[6, 7], [8, 9]]                        */
Tensor* tensor_scalar_add(const Tensor* a, float s);

/* ------------------------------------------------------------------ */
/* Reductions                                                          */
/* ------------------------------------------------------------------ */

/* Sum of all elements (any rank).
     tensor_sum([1, 2, 3, 4]) = 10.0                                  */
float tensor_sum(const Tensor* t);

/* Arithmetic mean of all elements.
     tensor_mean([1, 2, 3, 4]) = 2.5                                  */
float tensor_mean(const Tensor* t);

/* Largest element value.
     tensor_max([1, 4, 2, 3]) = 4.0                                   */
float tensor_max(const Tensor* t);

/* Flat-buffer index of the largest element.
     tensor_argmax([1, 4, 2, 3]) = 1                                  */
size_t tensor_argmax(const Tensor* t);

/* ------------------------------------------------------------------ */
/* Linear algebra                                                      */
/* ------------------------------------------------------------------ */

/* Batched matmul. Last two dims are the matrix; leading dims are batch.
   a: [..., M, K]   b: [..., K, N]   ->   out: [..., M, N]
     [[1, 2, 3], [4, 5, 6]]  @  [[1, 2], [3, 4], [5, 6]]
        shape [2, 3]              shape [3, 2]
     = [[22, 28], [49, 64]]    shape [2, 2]                           */
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);

/* Reorder axes. new axis i adopts the size of old axis axis_order[i].
   2D transpose:
     [[1, 2, 3], [4, 5, 6]]  permute({1, 0})  =  [[1, 4], [2, 5], [3, 6]]
   4D head-swap [B, S, H, D] -> [B, H, S, D]:
     permute({0, 2, 1, 3})                                            */
Tensor* tensor_permute(const Tensor* t, const size_t* axis_order);

/* ------------------------------------------------------------------ */
/* Activations / loss / bias broadcast                                 */
/* ------------------------------------------------------------------ */

/* Element-wise max(0, x).
     tensor_relu([-1, 2, -3, 4]) = [0, 2, 0, 4]                       */
Tensor* tensor_relu(const Tensor* t);

/* Element-wise tanh(x). */
Tensor* tensor_tanh(const Tensor* t);

/* Mean squared error: mean((pred - target)^2). Returns a scalar.
     pred = [1, 2], target = [2, 4]
     -> mean([1, 4]) = 2.5                                            */
float tensor_mse_loss(const Tensor* pred, const Tensor* target);

/* Add bias [N] to the last dim of `in` [..., N]; same bias for every
   leading-dim combination. Works for [B,N], [B,S,N], [B,H,S,N], etc.
     mat = [[1, 2], [3, 4]],  bias = [10, 20]
     -> [[11, 22], [13, 24]]                                          */
Tensor* tensor_add_bias(const Tensor* in, const Tensor* bias);

/* ------------------------------------------------------------------ */
/* Backward (manual gradients — one per forward op)                    */
/* ------------------------------------------------------------------ */

/* grad_A for C = A @ B, given grad_C. Equivalent to grad_C @ Bᵀ.
   Shapes: grad_C [..., M, N], B [..., K, N] -> grad_A [..., M, K]    */
Tensor* tensor_matmul_backward_a(const Tensor* grad_out, const Tensor* b);

/* grad_B for C = A @ B, given grad_C. Equivalent to Aᵀ @ grad_C.
   Shapes: A [..., M, K], grad_C [..., M, N] -> grad_B [..., K, N]    */
Tensor* tensor_matmul_backward_b(const Tensor* a, const Tensor* grad_out);

/* Pass grad through where the forward input was positive, zero else.
     grad_out = [1, 2, 3, 4],  input = [-1, 2, -3, 4]
     -> [0, 2, 0, 4]                                                  */
Tensor* tensor_relu_backward(const Tensor* grad_out, const Tensor* input);

/* d(tanh)/d(x) = 1 - tanh(x)^2.
   Pass the *forward output* of tanh (not the pre-activation).          */
Tensor* tensor_tanh_backward(const Tensor* grad_out, const Tensor* forward_output);

/* grad_bias [N] = sum of grad_out across all leading dims.
     grad_out = [[1, 2], [3, 4]]
     -> grad_bias = [4, 6]                                            */
Tensor* tensor_add_bias_backward_bias(const Tensor* grad_out);

/* d(mse)/d(pred) = (2/n) * (pred - target).
     pred = [1, 2],  target = [2, 4]   (n = 2)
     -> (2/2) * [-1, -2] = [-1, -2]                                   */
Tensor* tensor_mse_loss_backward(const Tensor* pred, const Tensor* target);

/* ------------------------------------------------------------------ */
/* Optimizer (in-place)                                                */
/* ------------------------------------------------------------------ */

/* In-place SGD update: param -= lr * grad.
     param = [1, 2, 3],  grad = [0.1, 0.2, 0.3],  lr = 0.1
     -> param = [0.99, 1.98, 2.97]                                    */
void tensor_sgd_step(Tensor* param, const Tensor* grad, float lr);

/* ------------------------------------------------------------------ */
/* Misc                                                                */
/* ------------------------------------------------------------------ */

/* Seed the PRNG used by tensor_rand. Same seed => same values.
     tensor_seed(42);                                                 */
void tensor_seed(unsigned int seed);

/* Print "Tensor(shape=[...], strides=[...])" to stdout.
     tensor_print_shape(t);                                           */
void tensor_print_shape(const Tensor* t);

#endif
