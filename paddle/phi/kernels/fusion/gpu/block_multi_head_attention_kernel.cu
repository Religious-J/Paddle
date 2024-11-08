// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/fusion/cutlass/variable_length_memory_efficient_attention.h"
#include "paddle/phi/kernels/fusion/gpu/block_attn.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/utils/none.h"

inline int getSMVersion() {
  const int device = phi::backends::gpu::GetCurrentDeviceId();
  const phi::gpuDeviceProp prop =
      phi::backends::gpu::GetDeviceProperties(device);
  return prop.major * 10 + prop.minor;
}

#if defined(__CUDACC__) && CUDA_VERSION >= 11000
#define CUDA_BFLOAT16_AVALIABLE
#include <cuda_bf16.h>
#endif

namespace phi {
namespace fusion {

int GetMaxLen(const phi::GPUContext& dev_ctx,
              const phi::DenseTensor& seq_lens_tensor,
              phi::DenseTensor* max_len_tensor,
              const int batch_size) {
  constexpr int blockSize = 128;
  int max_len_cpu = 0;
  GetMaxLenKernel<blockSize><<<1, blockSize, 0, dev_ctx.stream()>>>(
      seq_lens_tensor.data<int>(), max_len_tensor->data<int>(), batch_size);
  memory_utils::Copy(phi::CPUPlace(),
                     &max_len_cpu,
                     dev_ctx.GetPlace(),
                     max_len_tensor->data<int>(),
                     sizeof(int),
                     dev_ctx.stream());
  return max_len_cpu;
}

template <typename data_t>
inline HOSTDEVICE data_t RoundWithTiesToEven(data_t x) {
  data_t xLower = floor(x);
  data_t xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  data_t dLower = x - xLower;
  data_t dUpper = xUpper - x;
  return static_cast<data_t>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T>
__forceinline__ __device__ T add_mul(T a, T b, T c) {
  return (a + b) * c;
}

template <>
__forceinline__ __device__ half add_mul<half>(half a, half b, half c) {
  return __hmul(__hadd(a, b), c);
}

#ifdef CUDA_BFLOAT16_AVALIABLE
template <>
__forceinline__ __device__ __nv_bfloat16
add_mul<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hmul(__hadd(a, b), c);
#endif
}
#endif

template <typename data_t>
__forceinline__ __device__ int8_t quant_helper(const data_t input,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(RoundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename data_t>
__forceinline__ __device__ int8_t quant_helper(const data_t input,
                                               const data_t shift,
                                               const data_t smooth,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  auto smooth_out = add_mul(input, shift, smooth);
  float quant_value = max_bound * scale * static_cast<float>(smooth_out);

  if (round_type == 0) {
    quant_value = static_cast<float>(RoundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename data_t>
__forceinline__ __device__ phi::dtype::float8_e4m3fn fp8_quant_helper(
    const data_t input,
    const float scale,
    const int round_type,
    const float max_bound,
    const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<phi::dtype::float8_e4m3fn>(quant_value);
}

template <typename data_t>
__global__ void QuantKernel(const data_t* input,
                            char4* output,
                            const float scale,
                            const int m,
                            const int n,
                            const int round_type,
                            const float max_bound,
                            const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;
  bool check = ((m_id < m) && (n_id < n));

  if (check) {
    char4 tmp;
    tmp.x = quant_helper(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper(
        input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper(
        input[m_id * n + n_id + 2], scale, round_type, max_bound, min_bound);
    tmp.w = quant_helper(
        input[m_id * n + n_id + 3], scale, round_type, max_bound, min_bound);

    output[(m_id * n + n_id) >> 2] = tmp;
  }
}

template <typename data_t>
__global__ void FP8QuantKernel(const data_t* input,
                               phi::dtype::float8_e4m3fn* output,
                               const float scale,
                               const int m,
                               const int n,
                               const int round_type,
                               const float max_bound,
                               const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;
  bool check = ((m_id < m) && (n_id < n));

  if (check) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output[m_id * n + n_id] = fp8_quant_helper(
          input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
      output[m_id * n + n_id + 1] = fp8_quant_helper(
          input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
      output[m_id * n + n_id + 2] = fp8_quant_helper(
          input[m_id * n + n_id + 2], scale, round_type, max_bound, min_bound);
      output[m_id * n + n_id + 3] = fp8_quant_helper(
          input[m_id * n + n_id + 3], scale, round_type, max_bound, min_bound);
    }
  }
}

template <typename data_t>
__global__ void QuantKernel(const data_t* input,
                            const data_t* shift,
                            const data_t* smooth,
                            char4* output,
                            const float scale,
                            const int m,
                            const int n,
                            const int round_type,
                            const float max_bound,
                            const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;
  bool check = ((m_id < m) && (n_id < n));

  if (check) {
    char4 tmp;
    tmp.x = quant_helper(input[m_id * n + n_id],
                         shift[n_id],
                         smooth[n_id],
                         scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.y = quant_helper(input[m_id * n + n_id + 1],
                         shift[n_id + 1],
                         smooth[n_id + 1],
                         scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.z = quant_helper(input[m_id * n + n_id + 2],
                         shift[n_id + 2],
                         smooth[n_id + 2],
                         scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.w = quant_helper(input[m_id * n + n_id + 3],
                         shift[n_id + 3],
                         smooth[n_id + 3],
                         scale,
                         round_type,
                         max_bound,
                         min_bound);

    output[(m_id * n + n_id) >> 2] = tmp;
  }
}

template <typename T, int VecSize>
__global__ void DequantKernel(T* output,
                              const int32_t* input,
                              const int64_t m,  // batch size
                              const int64_t n,  // hidden
                              const float* dequant_out_scale_data) {
  int64_t numel = m * n;
  int64_t stride = blockDim.x * gridDim.x * VecSize;
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int64_t col_id = idx % n;

  phi::AlignedVector<int32_t, VecSize> in_vec;
  phi::AlignedVector<float, VecSize> out_scale_vec;
  phi::AlignedVector<T, VecSize> out_vec;

  for (; idx < numel; idx += stride) {
    phi::Load<int32_t, VecSize>(input + idx, &in_vec);
    phi::Load<float, VecSize>(dequant_out_scale_data + col_id, &out_scale_vec);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] =
          static_cast<T>(static_cast<float>(in_vec[i]) * out_scale_vec[i]);
    }

    phi::Store<T, VecSize>(out_vec, output + idx);
  }
}

template <typename T, typename Context>
void DispatchWithDtype(
    const Context& dev_ctx,
    const DenseTensor& qkv,
    const DenseTensor& key_cache,               // cache 用于解码时的注意力机制
    const DenseTensor& value_cache,
    const DenseTensor& seq_lens_encoder,        // encoder 序列长度向量
    const DenseTensor& seq_lens_decoder,        // decoder 序列长度向量
    const DenseTensor& seq_lens_this_time,
    const DenseTensor& padding_offsets,
    const DenseTensor& cum_offsets,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& block_tables,            // 块索引表
    const paddle::optional<DenseTensor>& pre_key_cache,   // 之前的缓存数据，用于增量计算和解码器的自回归
    const paddle::optional<DenseTensor>& pre_value_cache,
    const paddle::optional<DenseTensor>& rope_emb,
    const paddle::optional<DenseTensor>& mask,
    const paddle::optional<DenseTensor>& tgt_mask,
    const paddle::optional<DenseTensor>& cache_k_quant_scales,
    const paddle::optional<DenseTensor>& cache_v_quant_scales,
    const paddle::optional<DenseTensor>& cache_k_dequant_scales,
    const paddle::optional<DenseTensor>& cache_v_dequant_scales,
    const paddle::optional<DenseTensor>& qkv_out_scale,
    const paddle::optional<DenseTensor>& qkv_bias,
    const paddle::optional<DenseTensor>& out_shift,
    const paddle::optional<DenseTensor>& out_smooth,
    const paddle::optional<DenseTensor>& max_enc_len_this_time,
    const paddle::optional<DenseTensor>& max_dec_len_this_time,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    const bool dynamic_cachekv_quant,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_scale,
    const std::string& compute_dtype,
    const float rope_theta,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,      // qkv_out is probely output relevent 
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out) {
  phi::DenseTensor qkv_buf;
  phi::DenseTensor fmha_buf;

 // fmha => flash multi-head attention
 // 分配输出张量的内存
  VLOG(1) << "fmha_out " << fmha_out->dims();
  if (fmha_out->dtype() == phi::DataType::INT8) {
    fmha_buf.Resize(fmha_out->dims());
    dev_ctx.template Alloc<T>(&fmha_buf);
    dev_ctx.template Alloc<int8_t>(fmha_out);
  } else if (fmha_out->dtype() == phi::DataType::FLOAT8_E4M3FN) {
    fmha_buf.Resize(fmha_out->dims());
    dev_ctx.template Alloc<T>(&fmha_buf);
    dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(fmha_out);
  } else {
    dev_ctx.template Alloc<T>(fmha_out);
    fmha_buf = *fmha_out;
  }

  // 初始化维度参数
  const auto& input_dims = qkv.dims();
  const auto& key_cache_dims = key_cache.dims();
  const int token_num = input_dims[0];
  const int kv_num_head = key_cache_dims[1];
  const int dim_head = key_cache_dims[3];
  const int total_num_head = qkv.dims()[qkv.dims().size() - 1] / dim_head;
  const int q_num_head = total_num_head - 2 * kv_num_head;    // mark
  const int bsz = cum_offsets.dims()[0];
  const int max_block_per_seq = block_tables.dims()[1];
  VLOG(3) << "bsz: " << bsz << " token_num: " << token_num
          << " q_num_head: " << q_num_head << " kv_num_head: " << kv_num_head
          << " dim_head: " << dim_head
          << " max_block_per_seq: " << max_block_per_seq;
  VLOG(3) << "fmha_out_dims: " << fmha_out->dims();

 // causual 变量控制是否使用因果注意力机制（Causal Attention），这是自回归模型中的一种约束，用来确保序列中的第 i 个位置只能关注之前的 i-1 个位置
  bool causual = true;
  if (mask) {          // why ??
    causual = false;
  }

  bool use_pre_cache = false;
  int pre_cache_length = 0;
  if (pre_key_cache) {
    use_pre_cache = true;
    pre_cache_length = pre_key_cache.get().dims()[2];
  }
  VLOG(3) << "token_num: " << token_num
          << " pre_cache_length: " << pre_cache_length;

  // [数据处理]
  // 计算当前解码和编码时的最大长度。如果 max_dec_len_this_time 或 max_enc_len_this_time 为空，函数会通过辅助张量动态计算出序列的最大长度
  int max_dec_len_this_time_data(0);
  // 计算解码长度
  if (!max_dec_len_this_time) {          // max_dec_len_this_time 为 flase
    phi::DenseTensor max_dec_len_tensor;
    max_dec_len_tensor.Resize({{1}});
    auto* max_dec_len_data = dev_ctx.template Alloc<int>(
        &max_dec_len_tensor, max_dec_len_tensor.numel() * sizeof(int));
    max_dec_len_this_time_data =
        GetMaxLen(dev_ctx, seq_lens_decoder, &max_dec_len_tensor, bsz);
  } else {
    PADDLE_ENFORCE_EQ(
        max_dec_len_this_time.get().place().GetType(),
        phi::AllocationType::CPU,
        errors::InvalidArgument(
            "The place of input max_dec_len_this_time must be CPU, but got %s.",
            max_dec_len_this_time.get().place()));
    max_dec_len_this_time_data = *max_dec_len_this_time.get().data<int>();
  }

  int max_enc_len_this_time_data(0);
  // 计算编码长度
  if (!max_enc_len_this_time) {
    phi::DenseTensor max_enc_len_tensor;
    max_enc_len_tensor.Resize({{1}});
    auto* max_enc_len_data = dev_ctx.template Alloc<int>(
        &max_enc_len_tensor, max_enc_len_tensor.numel() * sizeof(int));
    max_enc_len_this_time_data =
        GetMaxLen(dev_ctx, seq_lens_encoder, &max_enc_len_tensor, bsz);
  } else {
    PADDLE_ENFORCE_EQ(
        max_enc_len_this_time.get().place().GetType(),
        phi::AllocationType::CPU,
        errors::InvalidArgument(
            "The place of input max_enc_len_this_time must be CPU, but got %s.",
            max_enc_len_this_time.get().place()));
    max_enc_len_this_time_data = *max_enc_len_this_time.get().data<int>();
  }

  // 为解码器配置输出张量的尺寸和内存
  phi::DenseTensor qkv_out_decoder;
  if (max_dec_len_this_time_data > 0) {
    // yes normal
    if (q_num_head == kv_num_head) {
      qkv_out_decoder.Resize({{bsz, 3, q_num_head, dim_head}});
    } else {
      qkv_out_decoder.Resize({{bsz, q_num_head + 2 * kv_num_head, dim_head}});
    }
    auto* qkv_out_decoder_data = dev_ctx.template Alloc<T>(
        &qkv_out_decoder, qkv_out_decoder.numel() * sizeof(T));
  }
  VLOG(3) << "max_len end";
  phi::DenseTensor unpadding_q, unpadding_k, unpadding_v;
  phi::DenseTensor softmax_out, softmax_lse, seed_offset;
  phi::DenseTensor q_trans, k_trans, v_trans, qktv_out;
  int sm = getSMVersion();
  if (max_enc_len_this_time_data > 0) {
    if (!use_pre_cache && sm >= 80) {
      unpadding_q.Resize({{token_num, q_num_head, dim_head}});
      unpadding_k.Resize({{token_num, kv_num_head, dim_head}});
      unpadding_v.Resize({{token_num, kv_num_head, dim_head}});

      dev_ctx.template Alloc<T>(&unpadding_q, unpadding_q.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&unpadding_k, unpadding_k.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&unpadding_v, unpadding_v.numel() * sizeof(T));
    } else {
      q_trans.Resize({{bsz, q_num_head, max_enc_len_this_time_data, dim_head}});
      k_trans.Resize({{bsz,
                       kv_num_head,
                       max_enc_len_this_time_data + pre_cache_length,
                       dim_head}});
      v_trans.Resize({{bsz,
                       kv_num_head,
                       max_enc_len_this_time_data + pre_cache_length,
                       dim_head}});
      qktv_out.Resize(
          {{bsz, q_num_head, max_enc_len_this_time_data, dim_head}});

      dev_ctx.template Alloc<T>(&q_trans, q_trans.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&k_trans, k_trans.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&v_trans, v_trans.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
    }
  }
  VLOG(3) << "encoder";
  VLOG(3) << "max_enc_len_this_time: " << max_enc_len_this_time_data;

  // [量化处理]
  // 如果 qkv_out_scale 存在，这里会对 qkv 数据进行反量化操作
  if (qkv_out_scale) {
    VLOG(1) << "qkv_out_scale: " << qkv_out_scale.get_ptr()->dims();
    qkv_buf.Resize(qkv.dims());
    dev_ctx.template Alloc<T>(&qkv_buf, qkv_buf.numel() * sizeof(T));

    int64_t numel = qkv.numel();
    constexpr int64_t thread_per_block = 512;
    constexpr int DequantKernelVecSize = 4;
    int64_t block_per_grid =
        (numel / DequantKernelVecSize + thread_per_block - 1) /
        thread_per_block;
    DequantKernel<T, DequantKernelVecSize>
        <<<block_per_grid, thread_per_block, 0, dev_ctx.stream()>>>(
            qkv_buf.data<T>(),
            qkv.data<int32_t>(),
            input_dims[0],
            input_dims[1],
            qkv_out_scale.get_ptr()->data<float>());
  } else {
    VLOG(1) << "qkv_out_scale is none";
    qkv_buf = qkv;
  }

  // 如果存在偏置（qkv_bias），代码会将其添加到 qkv_buf 中
  if (qkv_bias) {
    VLOG(1) << "has bias";
    std::vector<const phi::DenseTensor*> ins = {&qkv_buf, qkv_bias.get_ptr()};
    std::vector<phi::DenseTensor*> outs = {&qkv_buf};
    // 使用 BroadcastKernel 函数，利用 AddFunctor 对 qkv_buf 和 qkv_bias 的元素进行逐个相加，并将结果写回 qkv_buf
    phi::funcs::BroadcastKernel<T>(
        dev_ctx, ins, &outs, phi::funcs::AddFunctor<T>());
  }

  // -------------------> up all is 对输入数据的处理 


  // 位置编码（Rotary Position Embedding）
  // -------------------> begin - encoder ------------------
  if (max_enc_len_this_time_data > 0) {
    const int* sequence_lengths_data = seq_lens_encoder.data<int>();
    // VLOGMatrix(
    //     qkv_buf.data<T>(), qkv_buf.numel(), "qkv_buf before",
    //     qkv_buf.numel());
    if (rope_emb) {
      if (q_num_head == kv_num_head) {
        // rotary qk
        rotary_qk_variable(dev_ctx,
                           qkv_buf.data<T>(),
                           qkv_buf.data<T>(),
                           rope_emb.get().data<float>(),
                           padding_offsets.data<int>(),
                           sequence_lengths_data,
                           token_num,
                           q_num_head,
                           max_seq_len,
                           rope_emb.get().dims()[2],
                           dim_head,
                           use_neox_style);
      } else {
        gqa_rotary_qk_variable(dev_ctx,
                               qkv_buf.data<T>(),
                               qkv_buf.data<T>(),
                               rope_emb.get().data<float>(),
                               padding_offsets.data<int>(),
                               sequence_lengths_data,
                               token_num,
                               q_num_head,
                               kv_num_head,
                               max_seq_len,
                               rope_emb.get().dims()[2],
                               dim_head,
                               use_neox_style);
      }
    }
    // VLOGMatrix(
    //     qkv_buf.data<T>(), qkv_buf.numel(), "qkv_buf after",
    //     qkv_buf.numel());
    VLOG(3) << "rope end";
    VLOG(3) << "causual: " << causual;
    
    // 解开填充并进行张量变换
    // 调用 qkv_transpose_split 函数，将 qkv_buf 中的数据去掉填充项，分割成 unpadding_q、unpadding_k 和 unpadding_v
    // 输入的 qkv_buf buf?? 中的 从 padding => unpadding 
    if (!use_pre_cache && sm >= 80) {
      // 解 padding
      qkv_transpose_split<T>(dev_ctx,
                             unpadding_q.data<T>(),
                             unpadding_k.data<T>(),
                             unpadding_v.data<T>(),
                             qkv_buf.data<T>(),
                             padding_offsets.data<int>(),
                             sequence_lengths_data,
                             token_num,
                             bsz,
                             q_num_head,
                             kv_num_head,
                             max_seq_len,
                             dim_head);
      VLOG(3) << "qkv split end";
      // VLOGMatrix(unpadding_q.data<T>(),
      //            unpadding_q.numel(),
      //            "unpadding_q",
      //            unpadding_q.numel());
      // VLOGMatrix(unpadding_k.data<T>(),
      //            unpadding_k.numel(),
      //            "unpadding_k",
      //            unpadding_k.numel());
      // VLOGMatrix(unpadding_v.data<T>(),
      //            unpadding_v.numel(),
      //            "unpadding_v",
      //            unpadding_v.numel());
      // Reshape fmha_buf to 3-D because FlashAttnUnpaddedKernel requires
      // q,k,v,out all in 3-D [token_num, q_num_head, dim_head].
      auto fmha_shape = fmha_buf.dims();
      fmha_buf.Resize({token_num, q_num_head, dim_head});
      // encoder
      // 进行无填充的 Flash 注意力计算
      phi::FlashAttnUnpaddedKernel<T>(dev_ctx,
                                      unpadding_q,
                                      unpadding_k,
                                      unpadding_v,
                                      cu_seqlens_q,
                                      cu_seqlens_k,
                                      paddle::none /*fixed_seed_offset*/,
                                      causual ? paddle::none : mask,
                                      max_enc_len_this_time_data,
                                      max_enc_len_this_time_data,
                                      1.0f / sqrt(static_cast<float>(dim_head)),
                                      0.0,
                                      causual,
                                      false,
                                      true /* is_test*/,
                                      "" /*rng_name*/,
                                      // => output
                                      &fmha_buf,
                                      &softmax_out,
                                      &softmax_lse,
                                      &seed_offset);
      // Reshape fmha_buf back (to 2-D), to not affect following codes.
      fmha_buf.Resize(fmha_shape);
    } else {
      // 这边应该和上述的计算类似，只是为性能较差的实现
      if (sm < 80 && !use_pre_cache) {
        // 代码会调用 MultiHeadAttentionVariableForwardKernel，这是 Paddle 提供的另一种优化的多头自注意力计算内核
        qkv_transpose_split<T>(
            dev_ctx,
            q_trans.data<T>(),
            k_trans.data<T>(),
            v_trans.data<T>(),
            qkv_buf.data<T>(),
            pre_key_cache ? pre_key_cache.get().data<T>() : nullptr,
            pre_value_cache ? pre_value_cache.get().data<T>() : nullptr,
            padding_offsets.data<int>(),
            sequence_lengths_data,
            token_num,
            bsz,
            q_num_head,
            kv_num_head,
            max_enc_len_this_time_data,
            max_seq_len,
            pre_cache_length,
            dim_head);
      } else {
        qkv_transpose_split<T>(
            dev_ctx,
            q_trans.data<T>(),
            k_trans.data<T>(),
            v_trans.data<T>(),
            qkv.data<T>(),
            pre_key_cache ? pre_key_cache.get().data<T>() : nullptr,
            pre_value_cache ? pre_value_cache.get().data<T>() : nullptr,
            padding_offsets.data<int>(),
            sequence_lengths_data,
            token_num,
            bsz,
            q_num_head,
            kv_num_head,
            max_enc_len_this_time_data,
            max_seq_len,
            pre_cache_length,
            dim_head);
      }
#ifdef PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION
      phi::fusion::MultiHeadAttentionVariableForwardKernel<T, phi::GPUContext>(
          dev_ctx,
          q_trans,
          k_trans,
          v_trans,
          seq_lens_encoder,
          seq_lens_encoder,
          (sm < 80 && !use_pre_cache) ? paddle::none : mask,
          1.0f / sqrt(static_cast<float>(dim_head)),
          (sm < 80 && !use_pre_cache) ? causual : false,
          pre_cache_length,
          &qktv_out);
#elif defined(PADDLE_WITH_HIP)
      const bool is_precache_infer = true;
      phi::FlashAttnKernel<T>(
          dev_ctx,
          q_trans,
          k_trans,
          v_trans,
          paddle::none /*fixed_seed_offset*/,
          paddle::none /*mask*/,
          0.0,
          is_precache_infer ? true : causual /*precache_infer_casual*/,
          false,
          is_precache_infer /*is_test*/,
          "" /*rng_name*/,
          &qktv_out,
          &softmax_out,
          &softmax_lse,
          &seed_offset);
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "Not supports MultiHeadAttentionVariableForwardKernel."));
#endif
      InvokeTransposeRemovePadding<T>(dev_ctx,
                                      qktv_out.data<T>(),
                                      sequence_lengths_data,
                                      fmha_buf.data<T>(),
                                      bsz,
                                      q_num_head,
                                      max_enc_len_this_time_data,
                                      max_seq_len,
                                      dim_head,
                                      token_num,
                                      padding_offsets.data<int>());
    }

    // 代码会调用 MultiHeadAttentionVariableForwardKernel，这是 Paddle 提供的另一种优化的多头自注意力计算内核
    VLOG(3) << "flash end";
    if (cache_k_quant_scales && dynamic_cachekv_quant) {
      DynamicQuantCacheKernel<T>(dev_ctx,
                                 qkv_buf,
                                 block_tables,
                                 padding_offsets,
                                 seq_lens_encoder,
                                 *(cache_k_quant_scales.get_ptr()),
                                 *(cache_v_quant_scales.get_ptr()),
                                 *(cache_k_dequant_scales.get_ptr()),
                                 *(cache_v_dequant_scales.get_ptr()),
                                 pre_key_cache,
                                 pre_value_cache,
                                 bsz,
                                 q_num_head,
                                 kv_num_head,
                                 dim_head,
                                 max_seq_len,
                                 pre_cache_length,
                                 key_cache_out,
                                 value_cache_out);
    } else {
      // CacheKernel：否则，使用普通缓存内核，支持量化和去量化的配置，并为 key 和 value 缓存设置对应的量化、去量化缩放因子。???
      CacheKernel<T>(dev_ctx,
                     qkv_buf,
                     block_tables,
                     padding_offsets,
                     seq_lens_encoder,
                     pre_key_cache,
                     pre_value_cache,
                     cache_k_quant_scales,
                     cache_v_quant_scales,
                     bsz,
                     token_num,
                     q_num_head,
                     kv_num_head,
                     dim_head,
                     max_seq_len,
                     pre_cache_length,
                     key_cache_out,
                     value_cache_out,
                     quant_round_type,
                     quant_max_bound,
                     quant_min_bound);
    }
    VLOG(3) << "cache end";
  }
  // -------------------> end - encoder --------------------
  
  
  
  VLOG(3) << "encoder done";
  VLOG(3) << "max_dec_len_this_time: " << max_dec_len_this_time_data;
  // 解码器输出和处理
  // 如果解码器部分有数据处理需求（即 max_dec_len_this_time_data > 0），则对 qkv_buf 进行解码器输出准备
  // -------------------> begin - decoder ------------------ 
  if (max_dec_len_this_time_data > 0) {
    // 调用 GetDecoderTensor 提取解码器张量 qkv_out_decoder，并根据缓存量化模式设置 cachekv_quant_mode
    GetDecoderTensor<T>(dev_ctx,
                        qkv_buf,  // mark
                        nullptr,
                        cum_offsets.data<int>(),
                        &qkv_out_decoder,  // <= out  
                        nullptr,
                        token_num,
                        bsz,
                        q_num_head,
                        kv_num_head,
                        max_seq_len,
                        dim_head);
    VLOG(3) << "qkv_out_decoder: " << qkv_out_decoder.dims();
    int cachekv_quant_mode = 0;
    if (cache_k_quant_scales) {
      if (dynamic_cachekv_quant) {
        cachekv_quant_mode = 2;
      } else {
        cachekv_quant_mode = 1;
      }
    }
    // VLOGMatrix(qkv_out_decoder.data<T>(),
    //            qkv_out_decoder.numel(),
    //            "qkv_out_decoder",
    //            qkv_out_decoder.numel());
    // 最终调用 blha 函数执行解码器计算，支持可选的 rope_emb（旋转嵌入）和缓存量化配置，计算结果会保存在 fmha_buf 等变量中
    VLOG(1) << "cachekv_quant_mode " << cachekv_quant_mode;

    // !!!! blha  ==> decoder !!!! 
    blha<T>(dev_ctx,
            qkv_out_decoder,
            nullptr,  // qkv_bias
            &block_tables,
            tgt_mask ? &tgt_mask.get() : nullptr,
            &cum_offsets,
            &seq_lens_decoder,
            rope_emb ? &rope_emb.get() : nullptr,  // rope_emb
            key_cache_out,
            value_cache_out,
            &fmha_buf,
            bsz,
            max_block_per_seq,
            block_size,
            max_seq_len,
            pre_cache_length,
            q_num_head,
            kv_num_head,
            dim_head,
            max_dec_len_this_time_data,
            rope_emb ? 1 : 0,
            1. / sqrt(dim_head),
            rope_theta,
            /*compute_bias*/ false,
            use_neox_style,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            cache_k_quant_scales ? cache_k_quant_scales.get_ptr() : nullptr,
            cache_v_quant_scales ? cache_v_quant_scales.get_ptr() : nullptr,
            cache_k_dequant_scales ? cache_k_dequant_scales.get_ptr() : nullptr,
            cache_v_dequant_scales ? cache_v_dequant_scales.get_ptr() : nullptr,
            nullptr,  // dequant_qkv_scales
            nullptr,  // shift
            nullptr,  // smooth
            -1,       // quant_fmha_out_scale
            cachekv_quant_mode);
    VLOG(3) << "blha end";
  }
  // -------------------> end - decoder --------------------

  // VLOGMatrix(
  //     fmha_buf.data<T>(), fmha_buf.numel(), "fmha_buf", fmha_buf.numel());

  // 应该又是量化相关的
  if (out_scale > 0) {
    int m = fmha_out->dims()[0];
    int n = fmha_out->dims()[1];
#ifdef PADDLE_WITH_HIP
    dim3 grid(((n >> 2) + 63) / 64, (m + 7) / 8);
    dim3 block(64, 8);
#else
    dim3 grid((n >> 2 + 31) / 32, (m + 31) / 32);
    dim3 block(32, 32);
#endif
    if (out_shift && out_smooth) {
      QuantKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
          fmha_buf.data<T>(),
          out_shift.get_ptr()->data<T>(),
          out_smooth.get_ptr()->data<T>(),
          reinterpret_cast<char4*>(fmha_out->data<int8_t>()),
          out_scale,
          m,
          n,
          quant_round_type,
          quant_max_bound,
          quant_min_bound);
    } else {
      if (fmha_out->dtype() == phi::DataType::FLOAT8_E4M3FN) {
        FP8QuantKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
            fmha_buf.data<T>(),
            fmha_out->data<phi::dtype::float8_e4m3fn>(),
            out_scale,
            m,
            n,
            quant_round_type,
            quant_max_bound,
            quant_min_bound);
      } else {
        QuantKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
            fmha_buf.data<T>(),
            reinterpret_cast<char4*>(fmha_out->data<int8_t>()),
            out_scale,
            m,
            n,
            quant_round_type,
            quant_max_bound,
            quant_min_bound);
      }
    }
    VLOG(3) << "decoder done";
  }
}

// HERE input
template <typename T, typename Context>
void BlockMultiheadAttentionKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv,
    const DenseTensor& key_cache,
    const DenseTensor& value_cache,
    const DenseTensor& seq_lens_encoder,
    const DenseTensor& seq_lens_decoder,
    const DenseTensor& seq_lens_this_time,
    const DenseTensor& padding_offsets,
    const DenseTensor& cum_offsets,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& block_tables,
    const paddle::optional<DenseTensor>& pre_key_cache,
    const paddle::optional<DenseTensor>& pre_value_cache,
    const paddle::optional<DenseTensor>& rope_emb,
    const paddle::optional<DenseTensor>& mask,
    const paddle::optional<DenseTensor>& tgt_mask,
    const paddle::optional<DenseTensor>& cache_k_quant_scales,
    const paddle::optional<DenseTensor>& cache_v_quant_scales,
    const paddle::optional<DenseTensor>& cache_k_dequant_scales,
    const paddle::optional<DenseTensor>& cache_v_dequant_scales,
    const paddle::optional<DenseTensor>& qkv_out_scale,
    const paddle::optional<DenseTensor>& qkv_bias,
    const paddle::optional<DenseTensor>& out_shift,
    const paddle::optional<DenseTensor>& out_smooth,
    const paddle::optional<DenseTensor>& max_enc_len_this_time,
    const paddle::optional<DenseTensor>& max_dec_len_this_time,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    const bool dynamic_cachekv_quant,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_scale,
    const std::string& compute_dtype,
    const float rope_theta,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out) {

  // INT32 ?? quant? preliminary skip
  if (qkv.dtype() == phi::DataType::INT32) {
    VLOG(1) << "qkv.dtype() int32";
    if (compute_dtype == "fp16") {
      VLOG(1) << "compute_dtype fp16";
      DispatchWithDtype<phi::dtype::float16, Context>(dev_ctx,
                                                      qkv,
                                                      key_cache,
                                                      value_cache,
                                                      seq_lens_encoder,
                                                      seq_lens_decoder,
                                                      seq_lens_this_time,
                                                      padding_offsets,
                                                      cum_offsets,
                                                      cu_seqlens_q,
                                                      cu_seqlens_k,
                                                      block_tables,
                                                      pre_key_cache,
                                                      pre_value_cache,
                                                      rope_emb,
                                                      mask,
                                                      tgt_mask,
                                                      cache_k_quant_scales,
                                                      cache_v_quant_scales,
                                                      cache_k_dequant_scales,
                                                      cache_v_dequant_scales,
                                                      qkv_out_scale,
                                                      qkv_bias,
                                                      out_shift,
                                                      out_smooth,
                                                      max_enc_len_this_time,
                                                      max_dec_len_this_time,
                                                      max_seq_len,
                                                      block_size,
                                                      use_neox_style,
                                                      dynamic_cachekv_quant,
                                                      quant_round_type,
                                                      quant_max_bound,
                                                      quant_min_bound,
                                                      out_scale,
                                                      compute_dtype,
                                                      rope_theta,
                                                      fmha_out,
                                                      qkv_out,
                                                      key_cache_out,
                                                      value_cache_out);
    } else if (compute_dtype == "bf16") {
#ifdef CUDA_BFLOAT16_AVALIABLE
      DispatchWithDtype<phi::dtype::bfloat16, Context>(dev_ctx,
                                                       qkv,
                                                       key_cache,
                                                       value_cache,
                                                       seq_lens_encoder,
                                                       seq_lens_decoder,
                                                       seq_lens_this_time,
                                                       padding_offsets,
                                                       cum_offsets,
                                                       cu_seqlens_q,
                                                       cu_seqlens_k,
                                                       block_tables,
                                                       pre_key_cache,
                                                       pre_value_cache,
                                                       rope_emb,
                                                       mask,
                                                       tgt_mask,
                                                       cache_k_quant_scales,
                                                       cache_v_quant_scales,
                                                       cache_k_dequant_scales,
                                                       cache_v_dequant_scales,
                                                       qkv_out_scale,
                                                       qkv_bias,
                                                       out_shift,
                                                       out_smooth,
                                                       max_enc_len_this_time,
                                                       max_dec_len_this_time,
                                                       max_seq_len,
                                                       block_size,
                                                       use_neox_style,
                                                       dynamic_cachekv_quant,
                                                       quant_round_type,
                                                       quant_max_bound,
                                                       quant_min_bound,
                                                       out_scale,
                                                       compute_dtype,
                                                       rope_theta,
                                                       fmha_out,
                                                       qkv_out,
                                                       key_cache_out,
                                                       value_cache_out);
#endif
    }
  } 
  // HERE ??
  else {
    VLOG(1) << "qkv.dtype() NOT int32";
    // float16 -> half
    if (std::is_same<T, phi::dtype::float16>::value) {
      DispatchWithDtype<phi::dtype::float16, Context>(dev_ctx,
                                                      qkv,
                                                      key_cache,
                                                      value_cache,
                                                      seq_lens_encoder,
                                                      seq_lens_decoder,
                                                      seq_lens_this_time,
                                                      padding_offsets,
                                                      cum_offsets,
                                                      cu_seqlens_q,
                                                      cu_seqlens_k,
                                                      block_tables,
                                                      pre_key_cache,
                                                      pre_value_cache,
                                                      rope_emb,
                                                      mask,
                                                      tgt_mask,
                                                      cache_k_quant_scales,
                                                      cache_v_quant_scales,
                                                      cache_k_dequant_scales,
                                                      cache_v_dequant_scales,
                                                      qkv_out_scale,
                                                      qkv_bias,
                                                      out_shift,
                                                      out_smooth,
                                                      max_enc_len_this_time,
                                                      max_dec_len_this_time,
                                                      max_seq_len,
                                                      block_size,
                                                      use_neox_style,
                                                      dynamic_cachekv_quant,
                                                      quant_round_type,
                                                      quant_max_bound,
                                                      quant_min_bound,
                                                      out_scale,
                                                      compute_dtype,
                                                      rope_theta,
                                                      fmha_out,
                                                      qkv_out,
                                                      key_cache_out,
                                                      value_cache_out);
    } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
#ifdef CUDA_BFLOAT16_AVALIABLE
      DispatchWithDtype<phi::dtype::bfloat16, Context>(dev_ctx,
                                                       qkv,
                                                       key_cache,
                                                       value_cache,
                                                       seq_lens_encoder,
                                                       seq_lens_decoder,
                                                       seq_lens_this_time,
                                                       padding_offsets,
                                                       cum_offsets,
                                                       cu_seqlens_q,
                                                       cu_seqlens_k,
                                                       block_tables,
                                                       pre_key_cache,
                                                       pre_value_cache,
                                                       rope_emb,
                                                       mask,
                                                       tgt_mask,
                                                       cache_k_quant_scales,
                                                       cache_v_quant_scales,
                                                       cache_k_dequant_scales,
                                                       cache_v_dequant_scales,
                                                       qkv_out_scale,
                                                       qkv_bias,
                                                       out_shift,
                                                       out_smooth,
                                                       max_enc_len_this_time,
                                                       max_dec_len_this_time,
                                                       max_seq_len,
                                                       block_size,
                                                       use_neox_style,
                                                       dynamic_cachekv_quant,
                                                       quant_round_type,
                                                       quant_max_bound,
                                                       quant_min_bound,
                                                       out_scale,
                                                       compute_dtype,
                                                       rope_theta,
                                                       fmha_out,
                                                       qkv_out,
                                                       key_cache_out,
                                                       value_cache_out);
#endif
    }
  }
}

}  // namespace fusion
}  // namespace phi

#ifdef CUDA_BFLOAT16_AVALIABLE
PD_REGISTER_KERNEL(block_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BlockMultiheadAttentionKernel,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t) {
  kernel->InputAt(24).SetBackend(phi::Backend::CPU);
  kernel->InputAt(25).SetBackend(phi::Backend::CPU);
}
#else
PD_REGISTER_KERNEL(block_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BlockMultiheadAttentionKernel,
                   phi::dtype::float16,
                   int32_t) {
  kernel->InputAt(24).SetBackend(phi::Backend::CPU);
  kernel->InputAt(25).SetBackend(phi::Backend::CPU);
}
#endif
