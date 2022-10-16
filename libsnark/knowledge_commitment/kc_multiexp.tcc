/** @file
 *****************************************************************************
 * @author     This file is part of libsnark, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef KC_MULTIEXP_TCC_
#define KC_MULTIEXP_TCC_

#ifdef USE_GPU
#include "cgbn_math.h"
#include "cgbn_fp.h"
#include "cgbn_fp2.h"
//#include "cgbn_alt_bn128_g1.h"
//#include "cgbn_alt_bn128_g2.h"
#include "low_func_gpu.h"
#include "bigint_256.cuh"
#include <cuda_runtime.h>
#endif

namespace libsnark {

template<typename T1, typename T2, mp_size_t n>
knowledge_commitment<T1,T2> opt_window_wnaf_exp(const knowledge_commitment<T1,T2> &base,
                                                const libff::bigint<n> &scalar, const size_t scalar_bits)
{
    return knowledge_commitment<T1,T2>(opt_window_wnaf_exp(base.g, scalar, scalar_bits),
                                       opt_window_wnaf_exp(base.h, scalar, scalar_bits));
}


#ifdef USE_GPU

template<typename T, typename FieldT, libff::multi_exp_method Method>
T gpu_kc_multi_exp_with_mixed_addition_g2_mcl(const sparse_vector<T> &vec,
                                    typename std::vector<FieldT>::const_iterator scalar_start,
                                    typename std::vector<FieldT>::const_iterator scalar_end,
                                    std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                    const Config& config,
                                    libff::GpuMclData<T, FieldT, gpu::mcl_bn128_g2>& gpu_mcl_data)
{
    //libff::enter_block("Process scalar vector");
    auto index_it = vec.indices.begin();
    auto value_it = vec.values.begin();

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();

    //size_t num_skip = 0;
    //size_t num_add = 0;
    //size_t num_other = 0;

    const size_t scalar_size = std::distance(scalar_start, scalar_end);
    const size_t scalar_length = vec.indices.size();

    //libff::enter_block("allocate density memory");
    std::vector<bool> density(scalar_length);
    //libff::leave_block("allocate density memory");

    //std::vector<libff::bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    std::vector<libff::bigint<FieldT::num_limbs>> bn_exponents(scalar_length);

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);

    //libff::enter_block("find max index");
    std::vector<unsigned int> partial_max_indices(ranges.size(), 0xffffffff);
    //printf("ranges.size = %d\n", ranges.size());
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        T result = T::zero();
        unsigned int count = 0;
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            if(index_it[i] >= scalar_size)
            {
                partial_max_indices[j] = i;
                break;
            }
        }
    }

    unsigned int actual_max_idx = scalar_length;
    for (size_t j = 0; j < ranges.size(); j++)
    {
        if (partial_max_indices[j] != 0xffffffff)
        {
            actual_max_idx = partial_max_indices[j];
            break;
        }
    }
    //libff::leave_block("find max index");

    ranges = get_cpu_ranges(0, actual_max_idx);

    std::vector<T> partial(ranges.size(), T::zero());
    std::vector<unsigned int> counters(ranges.size(), 0);

    //call gpu
    if(true){
      int max_depth = 0, min_depth = 30130; 
      for(int i = 0; i < ranges.size(); i++){
          max_depth = std::max(max_depth, (int)(ranges[i].second-ranges[i].first));
      }
      gpu_mcl_data.max_depth = max_depth;

      const int local_instances = BlockDepth * 64;
      const int blocks = (max_depth + local_instances - 1) / local_instances;
      unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
      //unsigned int chunks = config.num_threads;
      unsigned int chunks = 1;//(254 + c - 1) / c;
      const int instances = gpu::BUCKET_INSTANCES;
      const int length = vec.values.size();
      const int ranges_size = ranges.size();
      const int values_size = vec.values.size();
      const int indices_size = vec.indices.size();

      cudaStream_t& stream = gpu_mcl_data.stream;
      libff::copy_t<T, FieldT>(T::zero(), gpu_mcl_data.d_t_zero, 0, stream);
      libff::copy_t<T, FieldT>(T::one(), gpu_mcl_data.d_t_one, 0, stream);
      libff::copy_field<T, FieldT>(zero, gpu_mcl_data.d_field_zero, 0, stream);
      libff::copy_field<T, FieldT>(one, gpu_mcl_data.d_field_one, 0, stream);

      uint64_t const_field_inv = scalar_start[0].inv;
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_one.mont_repr_data, value_it[0].pt.z.a.one().getUnit(), 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_p.mont_repr_data, value_it[0].pt.z.a.getOp().p, 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_a2.c0.mont_repr_data, value_it[0].pt.a_.a.getUnit(), 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_a2.c1.mont_repr_data, value_it[0].pt.a_.b.getUnit(), 32, stream);

      gpu_mcl_data.h_values.resize_host(values_size);
      gpu_mcl_data.d_values.resize(values_size);
      gpu_mcl_data.h_scalars.resize_host(scalar_size);
      gpu_mcl_data.d_scalars.resize(scalar_size);
      gpu_mcl_data.d_partial.resize(values_size);
      //gpu_mcl_data.d_partial.resize(ranges_size * gpu::REDUCE_BLOCKS_PER_RANGE * gpu::INSTANCES_PER_BLOCK);
      gpu_mcl_data.d_bn_exponents.resize(bn_exponents.size());
      //gpu_mcl_data.h_bn_exponents.resize_host(bn_exponents.size());
      for(int i = 0; i < chunks; i++){
        gpu_mcl_data.d_values2[i].resize(length);
        gpu_mcl_data.d_buckets[i].resize((1<<c) * instances);
        gpu_mcl_data.d_buckets2[i].resize((1<<c));
        gpu_mcl_data.d_block_sums[i].resize((1<<c) / 32);
        gpu_mcl_data.d_block_sums2[i].resize((1<<c) / 32/32);
      }
      {
        gpu_mcl_data.d_counters.resize(ranges_size * max_depth * sizeof(uint32_t));
        gpu_mcl_data.d_counters2.resize(ranges_size * sizeof(uint32_t));
        gpu_mcl_data.d_firsts.resize(ranges_size * sizeof(uint32_t));
        gpu_mcl_data.d_seconds.resize(ranges_size * sizeof(uint32_t));
        gpu_mcl_data.d_bucket_counters.resize((1<<c) * sizeof(int) * chunks);
        gpu_mcl_data.d_starts.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        gpu_mcl_data.d_indexs.resize((1<<c)*sizeof(int)*chunks*2);
        gpu_mcl_data.d_ids.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        gpu_mcl_data.d_instance_bucket_ids.resize((length+1) * sizeof(int) * chunks * 2);
        gpu_mcl_data.d_index_it.resize(indices_size * sizeof(size_t));
        gpu_mcl_data.d_density.resize(density.size());
        gpu_mcl_data.d_flags.resize(scalar_size);
      }
      std::vector<uint32_t> firsts(ranges_size), seconds(ranges_size);
      for(int i = 0; i < ranges_size; i++){
        firsts[i] = ranges[i].first;
        seconds[i] = ranges[i].second;
      }
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_firsts.ptr, firsts.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_seconds.ptr, seconds.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_index_it.ptr, vec.indices.data(), sizeof(size_t) * indices_size, stream);

#pragma omp parallel for
      for(int i = 0; i < values_size; i++){
        libff::copy_t_h<T, FieldT>(value_it[i], gpu_mcl_data.h_values, i);
      }
#pragma omp parallel for
      for(int i = 0; i < scalar_size; i++){
        libff::copy_field_h<T, FieldT>(scalar_start[i], gpu_mcl_data.h_scalars, i);
      }

      gpu_mcl_data.d_values.copy_from_cpu(gpu_mcl_data.h_values, stream);
      gpu_mcl_data.d_scalars.copy_from_cpu(gpu_mcl_data.h_scalars, stream);
      {
          gpu::mcl_bn128_g2_reduce_sum_new(
                  gpu_mcl_data.d_values, 
                  gpu_mcl_data.d_scalars, 
                  (size_t*)gpu_mcl_data.d_index_it.ptr, 
                  gpu_mcl_data.d_partial, 
                  (uint32_t*)gpu_mcl_data.d_counters.ptr, 
                  (char*)gpu_mcl_data.d_flags.ptr,
                  ranges_size, 
                  (uint32_t*)gpu_mcl_data.d_firsts.ptr, (uint32_t*)gpu_mcl_data.d_seconds.ptr,
                  gpu_mcl_data.d_t_zero, gpu_mcl_data.d_field_zero, gpu_mcl_data.d_field_one, (char*)gpu_mcl_data.d_density.ptr, gpu_mcl_data.d_bn_exponents.ptr, 
                  gpu_mcl_data.d_field_modulus.ptr, const_field_inv, 
                  gpu_mcl_data.d_one, gpu_mcl_data.d_p, gpu_mcl_data.d_a2, value_it[0].pt.specialA_, value_it[0].pt.mode_,value_it[0].pt.x.a.getOp().rp,
                  gpu_mcl_data.max_depth, values_size, stream);
          //gpu::sync(stream);
      }

      T gpu_acc;
      libff::copy_back<T, FieldT>(gpu_acc, gpu_mcl_data.d_partial, 0, stream);
      gpu::sync(stream);

      T exp_out = libff::multi_exp_with_density_gpu_mcl_g2<T, FieldT, false, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config, gpu_mcl_data.d_values, (char*)gpu_mcl_data.d_density.ptr, gpu_mcl_data.d_bn_exponents, gpu_mcl_data.dmax_value, gpu_mcl_data.d_modulus, gpu_mcl_data.d_t_zero, gpu_mcl_data.d_values2, gpu_mcl_data.d_buckets, gpu_mcl_data.d_buckets2, gpu_mcl_data.d_block_sums, gpu_mcl_data.d_block_sums2, (int*)gpu_mcl_data.d_bucket_counters.ptr, (int*)gpu_mcl_data.d_starts.ptr, (int*)gpu_mcl_data.d_indexs.ptr, (int*)gpu_mcl_data.d_ids.ptr, (int*)gpu_mcl_data.d_instance_bucket_ids.ptr, gpu_mcl_data.d_one, gpu_mcl_data.d_p, gpu_mcl_data.d_a2, stream);
      //T exp_out = libff::multi_exp_with_density<T, FieldT, true, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
      
      //gpu::sync(stream);
      auto tmp = gpu_acc + exp_out;

      return tmp;
    }
}
#endif

template<typename T, typename FieldT, libff::multi_exp_method Method>
T kc_multi_exp_with_mixed_addition(const sparse_vector<T> &vec,
                                    typename std::vector<FieldT>::const_iterator scalar_start,
                                    typename std::vector<FieldT>::const_iterator scalar_end,
                                    std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                    const Config& config)
{
    //libff::enter_block("Process scalar vector");
    auto index_it = vec.indices.begin();
    auto value_it = vec.values.begin();

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();

    //size_t num_skip = 0;
    //size_t num_add = 0;
    //size_t num_other = 0;

    const size_t scalar_size = std::distance(scalar_start, scalar_end);
    const size_t scalar_length = vec.indices.size();

    //libff::enter_block("allocate density memory");
    std::vector<bool> density(scalar_length);
    //libff::leave_block("allocate density memory");

    std::vector<libff::bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    if (bn_exponents.size() < scalar_length)
    {
        bn_exponents.resize(scalar_length);
    }

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);

    //libff::enter_block("find max index");
    std::vector<unsigned int> partial_max_indices(ranges.size(), 0xffffffff);
    //printf("ranges.size = %d\n", ranges.size());
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        T result = T::zero();
        unsigned int count = 0;
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            if(index_it[i] >= scalar_size)
            {
                partial_max_indices[j] = i;
                break;
            }
        }
    }

    unsigned int actual_max_idx = scalar_length;
    for (size_t j = 0; j < ranges.size(); j++)
    {
        if (partial_max_indices[j] != 0xffffffff)
        {
            actual_max_idx = partial_max_indices[j];
            break;
        }
    }
    //libff::leave_block("find max index");

    ranges = get_cpu_ranges(0, actual_max_idx);

    std::vector<T> partial(ranges.size(), T::zero());
    std::vector<unsigned int> counters(ranges.size(), 0);
    //libff::enter_block("reduce");
    //libff::enter_block("evaluation reduce");

#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        T result = T::zero();
        unsigned int count = 0;
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            const FieldT scalar = scalar_start[index_it[i]];
            if (scalar == zero)
            {
                // do nothing
                //++num_skip;
            }
            else if (scalar == one)
            {
#ifdef USE_MIXED_ADDITION
                result = result.mixed_add(value_it);
#else
								//result.resize_gpu();
								//value_it[i].resize_gpu();
                result = result + value_it[i];
                //result = result.gpu_add(value_it[i]);
#endif
                //++num_add;
            }
            else
            {
                density[i] = true;
                bn_exponents[i] = scalar.as_bigint();
                ++count;
                //++num_other;
            }
        }
        partial[j] = result; 
        counters[j] = count;
    }

    T acc = T::zero();
    unsigned int totalCount = 0;
    for (unsigned int i = 0; i < ranges.size(); i++)
    {
        acc = acc + partial[i];
        totalCount += counters[i];
    }
    //libff::leave_block("evaluation reduce");
    //libff::leave_block("reduce");

    //libff::leave_block("Process scalar vector");

    return acc + libff::multi_exp_with_density<T, FieldT, true, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
}

#ifdef USE_GPU
template<typename T, typename FieldT, libff::multi_exp_method Method>
void kc_multi_exp_with_mixed_addition_mcl_preprocess(const sparse_vector<T> &vec,
                                    typename std::vector<FieldT>::const_iterator scalar_start,
                                    typename std::vector<FieldT>::const_iterator scalar_end,
                                    std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                    const Config& config,
                                    libff::GpuMclData<T, FieldT>& gpu_mcl_data)
{
    //libff::enter_block("Process scalar vector");
    auto index_it = vec.indices.begin();
    auto value_it = vec.values.begin();

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();

    //size_t num_skip = 0;
    //size_t num_add = 0;
    //size_t num_other = 0;

    const size_t scalar_size = std::distance(scalar_start, scalar_end);
    const size_t scalar_length = vec.indices.size();

    //libff::enter_block("allocate density memory");
    std::vector<bool> density(scalar_length);
    //libff::leave_block("allocate density memory");

    //std::vector<libff::bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    std::vector<libff::bigint<FieldT::num_limbs>> bn_exponents(scalar_length); 

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);

    //libff::enter_block("find max index");
    std::vector<unsigned int> partial_max_indices(ranges.size(), 0xffffffff);
    //printf("ranges.size = %d\n", ranges.size());
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        T result = T::zero();
        unsigned int count = 0;
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            if(index_it[i] >= scalar_size)
            {
                partial_max_indices[j] = i;
                break;
            }
        }
    }

    unsigned int actual_max_idx = scalar_length;
    for (size_t j = 0; j < ranges.size(); j++)
    {
        if (partial_max_indices[j] != 0xffffffff)
        {
            actual_max_idx = partial_max_indices[j];
            break;
        }
    }
    //libff::leave_block("find max index");

    ranges = get_cpu_ranges(0, actual_max_idx);

    std::vector<T> partial(ranges.size(), T::zero());
    std::vector<unsigned int> counters(ranges.size(), 0);


    //call gpu
    if(true){
      int max_depth = 0, min_depth = 30130;
      for(int i = 0; i < ranges.size(); i++){
        max_depth = std::max(max_depth, (int)(ranges[i].second-ranges[i].first));
      }
      gpu_mcl_data.max_depth = max_depth;

      const int local_instances = BlockDepth * 64;
      const int blocks = (max_depth + local_instances - 1) / local_instances;
      unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
      //unsigned int chunks = config.num_threads;
      unsigned int chunks = 1;//(254 + c - 1) / c;
      const int instances = gpu::BUCKET_INSTANCES;
      const int length = vec.values.size();
      const int ranges_size = ranges.size();
      const int values_size = vec.values.size();
      const int indices_size = vec.indices.size();

      cudaStream_t& stream = gpu_mcl_data.stream;

      libff::copy_t<T, FieldT>(T::zero(), gpu_mcl_data.d_t_zero, 0, stream);
      libff::copy_t<T, FieldT>(T::one(), gpu_mcl_data.d_t_one, 0, stream);
      libff::copy_field<T, FieldT>(zero, gpu_mcl_data.d_field_zero, 0, stream);
      libff::copy_field<T, FieldT>(one, gpu_mcl_data.d_field_one, 0, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_one.mont_repr_data, value_it[0].pt.z.one().getUnit(), 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_p.mont_repr_data, value_it[0].pt.z.getOp().p, 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_a.mont_repr_data, value_it[0].pt.a_.getUnit(), 32, stream);

      gpu_mcl_data.h_values.resize_host(values_size);
      gpu_mcl_data.d_values.resize(values_size);
      gpu_mcl_data.h_scalars.resize_host(scalar_size);
      gpu_mcl_data.d_scalars.resize(scalar_size);
      gpu_mcl_data.d_partial.resize(values_size);
      gpu_mcl_data.d_bn_exponents.resize(bn_exponents.size());
      //gpu_mcl_data.h_bn_exponents.resize_host(bn_exponents.size());
      for(int i = 0; i < chunks; i++){
        gpu_mcl_data.d_values2[i].resize(length);
        gpu_mcl_data.d_buckets[i].resize((1<<c) * instances);
        gpu_mcl_data.d_buckets2[i].resize((1<<c));
        gpu_mcl_data.d_block_sums[i].resize((1<<c) / 32);
        gpu_mcl_data.d_block_sums2[i].resize((1<<c) / 32/32);
      }
      {
        gpu_mcl_data.d_counters.resize(ranges_size * max_depth * sizeof(uint32_t));
        gpu_mcl_data.d_counters2.resize(ranges_size * sizeof(uint32_t));
        gpu_mcl_data.d_firsts.resize(ranges_size * sizeof(uint32_t));
        gpu_mcl_data.d_seconds.resize(ranges_size * sizeof(uint32_t));
        gpu_mcl_data.d_bucket_counters.resize((1<<c) * sizeof(int) * chunks);
        gpu_mcl_data.d_starts.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        gpu_mcl_data.d_indexs.resize((1<<c)*sizeof(int)*chunks*2);
        gpu_mcl_data.d_ids.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        gpu_mcl_data.d_instance_bucket_ids.resize((length+1) * sizeof(int) * chunks * 2);
        gpu_mcl_data.d_index_it.resize(indices_size * sizeof(size_t));
        gpu_mcl_data.d_density.resize(density.size());
        gpu_mcl_data.d_flags.resize(scalar_size);
      }
      std::vector<uint32_t> firsts(ranges_size), seconds(ranges_size);
      for(int i = 0; i < ranges_size; i++){
        firsts[i] = ranges[i].first;
        seconds[i] = ranges[i].second;
      }
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_firsts.ptr, firsts.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_seconds.ptr, seconds.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_index_it.ptr, vec.indices.data(), sizeof(size_t) * indices_size, stream);
    }
}

template<typename T, typename FieldT, libff::multi_exp_method Method>
T kc_multi_exp_with_mixed_addition_mcl(const sparse_vector<T> &vec,
                                    typename std::vector<FieldT>::const_iterator scalar_start,
                                    typename std::vector<FieldT>::const_iterator scalar_end,
                                    std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                    const Config& config,
                                    libff::GpuMclData<T, FieldT>& gpu_mcl_data)
{
     const size_t scalar_length = vec.indices.size();
      auto ranges = libsnark::get_cpu_ranges(0, scalar_length);
      const int ranges_size = ranges.size();
      uint64_t const_field_inv = scalar_start[0].inv;
      auto index_it = vec.indices.begin();
      auto value_it = vec.values.begin();
      //int max_depth = 0, min_depth = 30130;
      //for(int i = 0; i < ranges.size(); i++){
      //    max_depth = std::max(max_depth, (int)(ranges[i].second-ranges[i].first));
      //}
      gpu::CudaStream& stream = gpu_mcl_data.stream;

      const int values_size = vec.values.size();
      const size_t scalar_size = std::distance(scalar_start, scalar_end);
#pragma omp parallel for
      for(int i = 0; i < values_size; i++){
        libff::copy_t_h<T, FieldT>(value_it[i], gpu_mcl_data.h_values, i);
      }
#pragma omp parallel for
      for(int i = 0; i < scalar_size; i++){
        libff::copy_field_h<T, FieldT>(scalar_start[i], gpu_mcl_data.h_scalars, i);
      }

      gpu_mcl_data.d_values.copy_from_cpu(gpu_mcl_data.h_values, stream);
      gpu_mcl_data.d_scalars.copy_from_cpu(gpu_mcl_data.h_scalars, stream);
      gpu::mcl_bn128_g1_reduce_sum(
          gpu_mcl_data.d_values, 
          gpu_mcl_data.d_scalars, 
          (size_t*)gpu_mcl_data.d_index_it.ptr, 
          gpu_mcl_data.d_partial, 
         (uint32_t*)gpu_mcl_data.d_counters.ptr, 
          (char*)gpu_mcl_data.d_flags.ptr,
          ranges_size, 
          (uint32_t*)gpu_mcl_data.d_firsts.ptr, (uint32_t*)gpu_mcl_data.d_seconds.ptr,
          gpu_mcl_data.d_t_zero, gpu_mcl_data.d_field_zero, gpu_mcl_data.d_field_one, (char*)gpu_mcl_data.d_density.ptr, gpu_mcl_data.d_bn_exponents.ptr, 
          gpu_mcl_data.d_field_modulus.ptr, const_field_inv, 
          gpu_mcl_data.d_one, gpu_mcl_data.d_p, gpu_mcl_data.d_a, value_it[0].pt.specialA_, value_it[0].pt.mode_,value_it[0].pt.x.getOp().rp,
          gpu_mcl_data.max_depth, stream);


      T gpu_acc;
      libff::copy_back<T, FieldT>(gpu_acc, gpu_mcl_data.d_partial, 0, stream);


      //T exp_out = libff::multi_exp_with_density<T, FieldT, true, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
      T exp_out = libff::multi_exp_with_density_gpu_mcl<T, FieldT, false, Method>(vec.values.begin(), vec.values.end(), config, gpu_mcl_data.d_values, (char*)gpu_mcl_data.d_density.ptr, gpu_mcl_data.d_bn_exponents, gpu_mcl_data.dmax_value, gpu_mcl_data.d_modulus, gpu_mcl_data.d_t_zero, gpu_mcl_data.d_values2, gpu_mcl_data.d_buckets, gpu_mcl_data.d_buckets2, gpu_mcl_data.d_block_sums, gpu_mcl_data.d_block_sums2, (int*)gpu_mcl_data.d_bucket_counters.ptr, (int*)gpu_mcl_data.d_starts.ptr, (int*)gpu_mcl_data.d_indexs.ptr, (int*)gpu_mcl_data.d_ids.ptr, (int*)gpu_mcl_data.d_instance_bucket_ids.ptr, gpu_mcl_data.d_one, gpu_mcl_data.d_p, gpu_mcl_data.d_a, stream);
      
      gpu::sync(stream);
      auto tmp = gpu_acc + exp_out;
      return tmp;
}
#endif

template<typename T1, typename T2, typename FieldT>
knowledge_commitment_vector<T1, T2> kc_batch_exp_internal(const size_t scalar_size,
                                                          const size_t T1_window,
                                                          const size_t T2_window,
                                                          const libff::window_table<T1> &T1_table,
                                                          const libff::window_table<T2> &T2_table,
                                                          const FieldT &T1_coeff,
                                                          const FieldT &T2_coeff,
                                                          const std::vector<FieldT> &v,
                                                          const size_t start_pos,
                                                          const size_t end_pos,
                                                          const size_t expected_size)
{
    knowledge_commitment_vector<T1, T2> res;

    res.values.reserve(expected_size);
    res.indices.reserve(expected_size);

    for (size_t pos = start_pos; pos != end_pos; ++pos)
    {
        if (!v[pos].is_zero())
        {
            res.values.emplace_back(knowledge_commitment<T1, T2>(windowed_exp(scalar_size, T1_window, T1_table, T1_coeff * v[pos]),
                                                                 windowed_exp(scalar_size, T2_window, T2_table, T2_coeff * v[pos])));
            res.indices.emplace_back(pos);
        }
    }

    return res;
}

template<typename T1, typename T2, typename FieldT>
knowledge_commitment_vector<T1, T2> kc_batch_exp(const size_t scalar_size,
                                                 const size_t T1_window,
                                                 const size_t T2_window,
                                                 const libff::window_table<T1> &T1_table,
                                                 const libff::window_table<T2> &T2_table,
                                                 const FieldT &T1_coeff,
                                                 const FieldT &T2_coeff,
                                                 const std::vector<FieldT> &v,
                                                 const size_t suggested_num_chunks)
{
    knowledge_commitment_vector<T1, T2> res;
    res.domain_size_ = v.size();

    size_t nonzero = 0;
    for (size_t i = 0; i < v.size(); ++i)
    {
        nonzero += (v[i].is_zero() ? 0 : 1);
    }

    const size_t num_chunks = std::max((size_t)1, std::min(nonzero, suggested_num_chunks));

    if (!libff::inhibit_profiling_info)
    {
        libff::print_indent(); printf("Non-zero coordinate count: %zu/%zu (%0.2f%%)\n", nonzero, v.size(), 100.*nonzero/v.size());
    }

    std::vector<knowledge_commitment_vector<T1, T2> > tmp(num_chunks);
    std::vector<size_t> chunk_pos(num_chunks+1);

    const size_t chunk_size = nonzero / num_chunks;
    const size_t last_chunk = nonzero - chunk_size * (num_chunks - 1);

    chunk_pos[0] = 0;

    size_t cnt = 0;
    size_t chunkno = 1;

    for (size_t i = 0; i < v.size(); ++i)
    {
        cnt += (v[i].is_zero() ? 0 : 1);
        if (cnt == chunk_size && chunkno < num_chunks)
        {
            chunk_pos[chunkno] = i;
            cnt = 0;
            ++chunkno;
        }
    }

    chunk_pos[num_chunks] = v.size();

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < num_chunks; ++i)
    {
        tmp[i] = kc_batch_exp_internal<T1, T2, FieldT>(scalar_size, T1_window, T2_window, T1_table, T2_table, T1_coeff, T2_coeff, v,
                                                       chunk_pos[i], chunk_pos[i+1], i == num_chunks - 1 ? last_chunk : chunk_size);
#ifdef USE_MIXED_ADDITION
        libff::batch_to_special<knowledge_commitment<T1, T2>>(tmp[i].values);
#endif
    }

    if (num_chunks == 1)
    {
        tmp[0].domain_size_ = v.size();
        return tmp[0];
    }
    else
    {
        for (size_t i = 0; i < num_chunks; ++i)
        {
            res.values.insert(res.values.end(), tmp[i].values.begin(), tmp[i].values.end());
            res.indices.insert(res.indices.end(), tmp[i].indices.begin(), tmp[i].indices.end());
        }
        return res;
    }
}

} // libsnark

#endif // KC_MULTIEXP_TCC_
