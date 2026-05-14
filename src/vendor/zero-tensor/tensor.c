#include "tensor.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void assert_same_shape(const Tensor* a, const Tensor* b) {
  assert(a->ndim == b->ndim);
  for (size_t i = 0; i < a->ndim; i++) assert(a->dims[i] == b->dims[i]);
}

Tensor* tensor_create(size_t ndim, const size_t* dims) {
  Tensor* t = malloc(sizeof(Tensor));
  t->ndim = ndim;
  t->dims = malloc(ndim * sizeof(size_t));
  t->strides = malloc(ndim * sizeof(size_t));
  memcpy(t->dims, dims, ndim * sizeof(size_t));

  /* Row-major (C-order) strides, counted in elements.
     The last dim has stride 1; each earlier dim multiplies by the size
     of all dims to its right. The running product is also the element_count. */
  size_t element_count = 1;
  for (size_t i = ndim; i > 0; i--) {
    t->strides[i - 1] = element_count;
    element_count *= dims[i - 1];
  }
  t->element_count = element_count;

  t->data = calloc(element_count, sizeof(float));
  return t;
}

Tensor* tensor_from_array(size_t ndim, const size_t* dims, const float* data) {
  Tensor* t = tensor_create(ndim, dims);
  memcpy(t->data, data, t->element_count * sizeof(float));
  return t;
}

Tensor* tensor_clone(const Tensor* t) { return tensor_from_array(t->ndim, t->dims, t->data); }

Tensor* tensor_rand(size_t ndim, const size_t* dims) {
  Tensor* t = tensor_create(ndim, dims);
  size_t n = t->element_count;
  for (size_t i = 0; i < n; i++) {
    t->data[i] = (float)rand() / ((float)RAND_MAX + 1.0f);
  }
  return t;
}

float tensor_get(const Tensor* t, const size_t* idx) {
  size_t offset = 0;
  for (size_t i = 0; i < t->ndim; i++) {
    offset += idx[i] * t->strides[i];
  }
  return t->data[offset];
}

void tensor_set(Tensor* t, const size_t* idx, float v) {
  size_t offset = 0;
  for (size_t i = 0; i < t->ndim; i++) {
    offset += idx[i] * t->strides[i];
  }
  t->data[offset] = v;
}

Tensor* tensor_permute(const Tensor* t, const size_t* axis_order) {
  size_t nd = t->ndim;
  size_t* new_dims = malloc(nd * sizeof(size_t));
  for (size_t i = 0; i < nd; i++) new_dims[i] = t->dims[axis_order[i]];
  Tensor* out = tensor_create(nd, new_dims);
  free(new_dims);

  size_t* src_idx = calloc(nd, sizeof(size_t));
  size_t* dst_idx = malloc(nd * sizeof(size_t));
  size_t n = t->element_count;

  for (size_t flat = 0; flat < n; flat++) {
    /* dst_idx[i] = src_idx[axis_order[i]] : new axis i pulls from old axis. */
    for (size_t i = 0; i < nd; i++) dst_idx[i] = src_idx[axis_order[i]];
    tensor_set(out, dst_idx, tensor_get(t, src_idx));

    /* Increment src_idx like a multi-digit odometer (last dim varies fastest). */
    for (size_t i = nd; i > 0; i--) {
      src_idx[i - 1]++;
      if (src_idx[i - 1] < t->dims[i - 1]) break;
      src_idx[i - 1] = 0;
    }
  }

  free(src_idx);
  free(dst_idx);
  return out;
}

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
  /* Batched matmul: a: [..., M, K]  b: [..., K, N]  ->  c: [..., M, N].
     Leading dims (everything before the last two) must match exactly. */
  size_t nd = a->ndim;
  size_t M = a->dims[nd - 2];
  size_t K = a->dims[nd - 1];
  size_t N = b->dims[nd - 1];

  size_t batch = 1;
  for (size_t i = 0; i + 2 < nd; i++) batch *= a->dims[i];

  size_t* out_dims = malloc(nd * sizeof(size_t));
  for (size_t i = 0; i + 2 < nd; i++) out_dims[i] = a->dims[i];
  out_dims[nd - 2] = M;
  out_dims[nd - 1] = N;
  Tensor* c = tensor_create(nd, out_dims);
  free(out_dims);

  /* For each batch slot, run the same 2D matmul as before on its slice
     of the flat buffers. Since tensors are row-major contiguous, slot
     `bi` of a [..., M, K] tensor starts at offset bi*M*K. */
  for (size_t bi = 0; bi < batch; bi++) {
    const float* A = a->data + bi * M * K;
    const float* B = b->data + bi * K * N;
    float* C = c->data + bi * M * N;
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  }
  return c;
}

/* ------------------------------------------------------------------ */
/* Element-wise math (same-shape, no broadcasting)                     */
/* ------------------------------------------------------------------ */

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
  assert_same_shape(a, b);
  Tensor* c = tensor_create(a->ndim, a->dims);
  size_t n = a->element_count;
  for (size_t i = 0; i < n; i++) c->data[i] = a->data[i] + b->data[i];
  return c;
}

Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
  assert_same_shape(a, b);
  Tensor* c = tensor_create(a->ndim, a->dims);
  size_t n = a->element_count;
  for (size_t i = 0; i < n; i++) c->data[i] = a->data[i] - b->data[i];
  return c;
}

Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
  assert_same_shape(a, b);
  Tensor* c = tensor_create(a->ndim, a->dims);
  size_t n = a->element_count;
  for (size_t i = 0; i < n; i++) c->data[i] = a->data[i] * b->data[i];
  return c;
}

Tensor* tensor_scalar_mul(const Tensor* a, float s) {
  Tensor* c = tensor_create(a->ndim, a->dims);
  size_t n = a->element_count;
  for (size_t i = 0; i < n; i++) c->data[i] = a->data[i] * s;
  return c;
}

Tensor* tensor_scalar_add(const Tensor* a, float s) {
  Tensor* c = tensor_create(a->ndim, a->dims);
  size_t n = a->element_count;
  for (size_t i = 0; i < n; i++) c->data[i] = a->data[i] + s;
  return c;
}

/* ------------------------------------------------------------------ */
/* Reductions                                                          */
/* ------------------------------------------------------------------ */

float tensor_sum(const Tensor* t) {
  size_t n = t->element_count;
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) sum += t->data[i];
  return sum;
}

float tensor_mean(const Tensor* t) { return tensor_sum(t) / (float)t->element_count; }

float tensor_max(const Tensor* t) {
  size_t n = t->element_count;
  float m = t->data[0];
  for (size_t i = 1; i < n; i++) {
    if (t->data[i] > m) m = t->data[i];
  }
  return m;
}

size_t tensor_argmax(const Tensor* t) {
  size_t n = t->element_count;
  size_t best = 0;
  for (size_t i = 1; i < n; i++) {
    if (t->data[i] > t->data[best]) best = i;
  }
  return best;
}

/* ------------------------------------------------------------------ */
/* Activations / loss / bias broadcast                                 */
/* ------------------------------------------------------------------ */

Tensor* tensor_relu(const Tensor* t) {
  Tensor* out = tensor_create(t->ndim, t->dims);
  size_t n = t->element_count;
  for (size_t i = 0; i < n; i++) {
    out->data[i] = t->data[i] > 0.0f ? t->data[i] : 0.0f;
  }
  return out;
}

Tensor* tensor_tanh(const Tensor* t) {
  Tensor* out = tensor_create(t->ndim, t->dims);
  size_t n = t->element_count;
  for (size_t i = 0; i < n; i++) {
    out->data[i] = tanhf(t->data[i]);
  }
  return out;
}

Tensor* tensor_add_bias(const Tensor* in, const Tensor* bias) {
  /* Input has any rank: [..., N]. Bias is 1D with size N (the last dim).
     Same bias gets added to every "row of N" in the flat buffer.
     Works for [B,N], [B,S,N], [B,H,S,N], etc.                          */
  assert(bias->ndim == 1);
  assert(bias->dims[0] == in->dims[in->ndim - 1]);

  size_t N = in->dims[in->ndim - 1];
  Tensor* out = tensor_create(in->ndim, in->dims);
  /* The last dim is the innermost in memory (stride 1), so as `i` walks
     the flat buffer, `i % N` cycles 0..N-1 across each row of features. */
  for (size_t i = 0; i < in->element_count; i++) {
    out->data[i] = in->data[i] + bias->data[i % N];
  }
  return out;
}

float tensor_mse_loss(const Tensor* pred, const Tensor* target) {
  assert_same_shape(pred, target);
  size_t n = pred->element_count;
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    float d = pred->data[i] - target->data[i];
    sum += d * d;
  }
  return sum / (float)n;
}

/* ------------------------------------------------------------------ */
/* Backward (manual gradients)                                         */
/* ------------------------------------------------------------------ */

Tensor* tensor_mse_loss_backward(const Tensor* pred, const Tensor* target) {
  /* d(loss)/d(pred_i) = (2/n) * (pred_i - target_i) */
  assert_same_shape(pred, target);
  Tensor* grad = tensor_create(pred->ndim, pred->dims);
  size_t n = pred->element_count;
  for (size_t i = 0; i < n; i++) {
    grad->data[i] = 2.0f * (pred->data[i] - target->data[i]) / (float)n;
  }
  return grad;
}

Tensor* tensor_relu_backward(const Tensor* grad_out, const Tensor* input) {
  /* Pass grad through where the forward input was positive, zero elsewhere. */
  assert_same_shape(grad_out, input);
  Tensor* grad_in = tensor_create(input->ndim, input->dims);
  for (size_t i = 0; i < input->element_count; i++) {
    grad_in->data[i] = input->data[i] > 0.0f ? grad_out->data[i] : 0.0f;
  }
  return grad_in;
}

Tensor* tensor_tanh_backward(const Tensor* grad_out, const Tensor* forward_output) {
  assert_same_shape(grad_out, forward_output);
  Tensor* grad_in = tensor_create(forward_output->ndim, forward_output->dims);
  for (size_t i = 0; i < forward_output->element_count; i++) {
    float y = forward_output->data[i];
    grad_in->data[i] = grad_out->data[i] * (1.0f - y * y);
  }
  return grad_in;
}

Tensor* tensor_add_bias_backward_bias(const Tensor* grad_out) {
  /* Bias was broadcast across every leading dim. Its grad is the sum
     of grad_out across all axes except the last (feature) axis.
     tensor_create zero-fills, so we just accumulate via +=.            */
  size_t N = grad_out->dims[grad_out->ndim - 1];
  size_t bias_dims[] = {N};
  Tensor* grad_bias = tensor_create(1, bias_dims);
  for (size_t i = 0; i < grad_out->element_count; i++) {
    grad_bias->data[i % N] += grad_out->data[i];
  }
  return grad_bias;
}

/* Swap the last two axes (transpose-for-matmul). Works for any rank. */
static Tensor* swap_last_two_axes(const Tensor* t) {
  size_t nd = t->ndim;
  size_t* axes = malloc(nd * sizeof(size_t));
  for (size_t i = 0; i < nd; i++) axes[i] = i;
  axes[nd - 1] = nd - 2;
  axes[nd - 2] = nd - 1;
  Tensor* out = tensor_permute(t, axes);
  free(axes);
  return out;
}

Tensor* tensor_matmul_backward_a(const Tensor* grad_out, const Tensor* b) {
  /* grad_A = grad_out @ Bᵀ */
  Tensor* b_T = swap_last_two_axes(b);
  Tensor* grad_a = tensor_matmul(grad_out, b_T);
  tensor_destroy(b_T);
  return grad_a;
}

Tensor* tensor_matmul_backward_b(const Tensor* a, const Tensor* grad_out) {
  /* grad_B = Aᵀ @ grad_out */
  Tensor* a_T = swap_last_two_axes(a);
  Tensor* grad_b = tensor_matmul(a_T, grad_out);
  tensor_destroy(a_T);
  return grad_b;
}

/* ------------------------------------------------------------------ */
/* Optimizer (in-place)                                                */
/* ------------------------------------------------------------------ */

void tensor_sgd_step(Tensor* param, const Tensor* grad, float lr) {
  assert_same_shape(param, grad);
  for (size_t i = 0; i < param->element_count; i++) {
    param->data[i] -= lr * grad->data[i];
  }
}

void tensor_seed(unsigned int seed) { srand(seed); }

void tensor_destroy(Tensor* t) {
  if (!t) return;
  free(t->dims);
  free(t->strides);
  free(t->data);
  free(t);
}

void tensor_print_shape(const Tensor* t) {
  printf("Tensor(shape=[");
  for (size_t i = 0; i < t->ndim; i++) {
    printf("%zu%s", t->dims[i], i + 1 < t->ndim ? ", " : "");
  }
  printf("], strides=[");
  for (size_t i = 0; i < t->ndim; i++) {
    printf("%zu%s", t->strides[i], i + 1 < t->ndim ? ", " : "");
  }
  printf("])\n");
}
