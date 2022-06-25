/** @file
 *****************************************************************************
 * @author     This file is part of libsnark, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef KC_MULTIEXP_TCC_
#define KC_MULTIEXP_TCC_
#include "cgbn_math.h"
#include "cgbn_fp.h"
#include "cgbn_fp2.h"
#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g2.h"
#include "low_func_gpu.h"
#include <cuda_runtime.h>


namespace libsnark {

template<typename T1, typename T2, mp_size_t n>
knowledge_commitment<T1,T2> opt_window_wnaf_exp(const knowledge_commitment<T1,T2> &base,
                                                const libff::bigint<n> &scalar, const size_t scalar_bits)
{
    return knowledge_commitment<T1,T2>(opt_window_wnaf_exp(base.g, scalar, scalar_bits),
                                       opt_window_wnaf_exp(base.h, scalar, scalar_bits));
}



template<typename T, typename FieldT, libff::multi_exp_method Method>
T gpu_kc_multi_exp_with_mixed_addition_g1(const sparse_vector<T> &vec,
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

    //std::vector<libff::bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    std::vector<libff::bigint<FieldT::num_limbs>> bn_exponents(scratch_exponents.size());
    if (bn_exponents.size() < scalar_length)
    {
        bn_exponents.resize(scalar_length);
    }

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);

    //libff::enter_block("find max index");
    std::vector<unsigned int> partial_max_indices(ranges.size(), 0xffffffff);
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

    //std::vector<T> partial(ranges.size(), T::zero());
    //std::vector<unsigned int> counters(ranges.size(), 0);
    //libff::enter_block("cpu reduce sum");
    int max_depth = 0, min_depth = 30130;
    for(int i = 0; i < ranges.size(); i++){
      max_depth = std::max(max_depth, (int)(ranges[i].second-ranges[i].first));
    }

    //libff::leave_block("cpu Process scalar vector");

    //printf("max_depth = %d, min_depth=%d\n", max_depth, min_depth);
    /***********start alt_bn128_g1 gpu reduce sum*****************/
    if(true){
      auto copy_t = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.x.mont_repr_data + offset, src.X.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.y.mont_repr_data + offset, src.Y.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.z.mont_repr_data + offset, src.Z.mont_repr.data, 32);
      };
      auto copy_t_h = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
        memcpy(dst.x.mont_repr_data + offset, src.X.mont_repr.data, 32);
        memcpy(dst.y.mont_repr_data + offset, src.Y.mont_repr.data, 32);
        memcpy(dst.z.mont_repr_data + offset, src.Z.mont_repr.data, 32);
      };
      auto copy_field = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
      auto copy_field_h = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        memcpy(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
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

      static gpu::alt_bn128_g1 h_values, d_values, d_partial, d_t_zero, d_t_one;
      static std::vector<gpu::alt_bn128_g1> d_values2(chunks), d_buckets(chunks), d_buckets2(chunks), d_block_sums(chunks), d_block_sums2(chunks);
      static gpu::Fp_model h_scalars, d_scalars, d_field_zero, d_field_one;

      static gpu::gpu_meta d_counters, d_counters2, d_index_it, d_firsts, d_seconds, d_bucket_counters, d_starts, d_indexs, d_ids, d_instance_bucket_ids, d_density, d_flags;
      static gpu::gpu_buffer max_value, dmax_value, d_bn_exponents, h_bn_exponents, d_modulus, d_field_modulus;
      static bool first_init = true;
      static cudaStream_t stream;

    //libff::enter_block("gpu malloc...");
      if(first_init){
          gpu::create_stream(&stream);
        d_t_zero.init(1);
        d_t_one.init(1);
        d_field_zero.init(1);
        d_field_one.init(1);
        max_value.resize_host(1);
        dmax_value.resize(1);
        d_modulus.resize(1);
        d_field_modulus.resize(1);
        for(int i = 0; i < BITS/32; i++){
          max_value.ptr->_limbs[i] = 0xffffffff;
        }
        dmax_value.copy_from_host(max_value);
        copy_t(T::zero(), d_t_zero, 0);
        copy_t(T::one(), d_t_one, 0);
        copy_field(zero, d_field_zero, 0);
        copy_field(one, d_field_one, 0);
        gpu::copy_cpu_to_gpu(d_modulus.ptr->_limbs, value_it[0].X.get_modulus().data, 32);
        gpu::copy_cpu_to_gpu(d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32);
        first_init = false;
      }
      h_values.resize_host(values_size);
      d_values.resize(values_size);
      h_scalars.resize_host(scalar_size);
      d_scalars.resize(scalar_size);
      //d_partial.resize(ranges_size * gpu::REDUCE_BLOCKS_PER_RANGE * gpu::INSTANCES_PER_BLOCK);
      d_partial.resize(values_size);
      d_bn_exponents.resize(bn_exponents.size());
      h_bn_exponents.resize_host(bn_exponents.size());
      for(int i = 0; i < chunks; i++){
        d_values2[i].resize(length);
        d_buckets[i].resize((1<<c) * instances);
        d_buckets2[i].resize((1<<c));
        d_block_sums[i].resize((1<<c) / 32);
        d_block_sums2[i].resize((1<<c) / 32/32);
      }
      //if(first_init){
        d_counters.resize(ranges_size * max_depth * sizeof(uint32_t));
        d_counters2.resize(ranges_size * sizeof(uint32_t));
        d_firsts.resize(ranges_size * sizeof(uint32_t));
        d_seconds.resize(ranges_size * sizeof(uint32_t));
        d_bucket_counters.resize((1<<c) * sizeof(int) * chunks);
        d_starts.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        d_indexs.resize((1<<c)*sizeof(int)*chunks*2);
        d_ids.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        d_instance_bucket_ids.resize((length+1) * sizeof(int) * chunks * 2);
        d_index_it.resize(indices_size * sizeof(size_t));
        d_density.resize(density.size());
        d_flags.resize(scalar_size);
      //}
      std::vector<uint32_t> firsts(ranges_size), seconds(ranges_size);
      for(int i = 0; i < ranges_size; i++){
        firsts[i] = ranges[i].first;
        seconds[i] = ranges[i].second;
      }
    //libff::leave_block("gpu malloc...");
    //libff::enter_block("gpu copy...");
      gpu::copy_cpu_to_gpu(d_firsts.ptr, firsts.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(d_seconds.ptr, seconds.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(d_index_it.ptr, vec.indices.data(), sizeof(size_t) * indices_size, stream);

      uint64_t const_inv = value_it[0].X.inv;
      uint64_t const_field_inv = scalar_start[0].inv;

      const auto& modu = value_it[0].X.get_modulus();
#pragma omp parallel for
      for(int i = 0; i < values_size; i++){
        copy_t_h(value_it[i], h_values, i);
      }
      d_values.copy_from_cpu(h_values);
#pragma omp parallel for
      for(int i = 0; i < scalar_size; i++){
        copy_field_h(scalar_start[i], h_scalars, i);
      }
      d_scalars.copy_from_cpu(h_scalars);
    //libff::leave_block("gpu copy...");

    //libff::enter_block("Compute evaluation to reduce", false);
      gpu::alt_bn128_g1_reduce_sum_one_range5(
          d_values, 
          d_scalars, 
          (size_t*)d_index_it.ptr, 
          d_partial, 
         (uint32_t*)d_counters.ptr, 
          (char*)d_flags.ptr,
          ranges_size, 
          (uint32_t*)d_firsts.ptr, (uint32_t*)d_seconds.ptr,
          dmax_value.ptr, d_t_zero, d_field_zero, d_field_one, (char*)d_density.ptr, d_bn_exponents.ptr, 
          d_modulus.ptr, const_inv, d_field_modulus.ptr, const_field_inv, max_depth, stream);

      //int gpu_total_count = 0;
      //gpu::copy_gpu_to_cpu(&gpu_total_count, d_counters, sizeof(int));

      auto copy_back = [&](T& dst, const gpu::alt_bn128_g1& src, const int offset){
        gpu::copy_gpu_to_cpu(dst.X.mont_repr.data, src.x.mont_repr_data + offset, 32, stream);
        gpu::copy_gpu_to_cpu(dst.Y.mont_repr.data, src.y.mont_repr_data + offset, 32, stream);
        gpu::copy_gpu_to_cpu(dst.Z.mont_repr.data, src.z.mont_repr_data + offset, 32, stream);
      };

      T gpu_acc;
      copy_back(gpu_acc, d_partial, 0);
    //libff::leave_block("Compute evaluation to reduce", false);

      //d_bn_exponents.copy_to_host(h_bn_exponents);
      //for(int i = 0; i < bn_exponents.size(); i++){
      //  memcpy(bn_exponents[i].data, h_bn_exponents.ptr + i, 32);
      //}
      //std::vector<char> h_density(density.size());
      //gpu::copy_gpu_to_cpu(h_density.data(), d_density.ptr, density.size());
      //for(int i = 0 ;i < density.size(); i++){
      //  density[i] = (h_density[i] == 1);
      //}
      //auto exp_out = libff::multi_exp_with_density<T, FieldT, true, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
      //libff::enter_block("gpu multi exp with density");
      T exp_out = libff::multi_exp_with_density_gpu<T, FieldT, false, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config, d_values, (char*)d_density.ptr, d_bn_exponents, dmax_value, d_modulus, d_t_zero, d_values2, d_buckets, d_buckets2, d_block_sums, d_block_sums2, (int*)d_bucket_counters.ptr, (int*)d_starts.ptr, (int*)d_indexs.ptr, (int*)d_ids.ptr, (int*)d_instance_bucket_ids.ptr, stream);
      
      auto tmp = gpu_acc + exp_out;
      //libff::leave_block("gpu multi exp with density");
      //libff::leave_block("call gpu");

      if(false){
        d_values.release();
        d_partial.release();
        for(int i = 0; i < chunks; i++){
          d_values2[i].release();
          d_buckets[i].release();
          d_buckets2[i].release();
          d_block_sums[i].release();
          d_block_sums2[i].release();
        }
        d_scalars.release();
        d_field_one.release();
        d_field_zero.release();
        d_density.release();
        d_flags.release();
        d_counters.release();
        d_counters2.release();
        d_index_it.release();
        d_firsts.release();
        d_seconds.release();
        d_bucket_counters.release();
        d_starts.release();
        d_indexs.release();
        d_ids.release();
        d_instance_bucket_ids.release();
        d_bn_exponents.release();
        dmax_value.release();
        d_modulus.release();
        d_field_modulus.release();
        d_t_zero.release();
      }

      return tmp;
    }

    /***********end alt_bn128_g1 gpu reduce sum*****************/

    //return tmp;
}

template<typename T, typename FieldT, libff::multi_exp_method Method>
T gpu_kc_multi_exp_with_mixed_addition_g2(const sparse_vector<T> &vec,
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

    //std::vector<libff::bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    std::vector<libff::bigint<FieldT::num_limbs>> bn_exponents(scratch_exponents.size());
    int bn_exponents_size = bn_exponents.size();
    if (bn_exponents.size() < scalar_length)
    {
        bn_exponents.resize(scalar_length);
        bn_exponents_size = scalar_length;
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

    //std::vector<T> partial(ranges.size(), T::zero());
    //std::vector<unsigned int> counters(ranges.size(), 0);
    int max_depth = 0, min_depth = 1000000;
    for(int i = 0; i < ranges.size(); i++){
      int depth = ranges[i].second - ranges[i].first;
      max_depth = std::max(max_depth, depth);
      min_depth = std::min(min_depth, depth);
    }
    //printf("max depth=%d, min depth=%d\n", max_depth, min_depth);

    /***********start alt_bn128_g2 gpu reduce sum*****************/
    if(true){
      auto copy_t = [](const T& src, gpu::alt_bn128_g2& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.x.c0.mont_repr_data + offset, src.X.c0.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.y.c0.mont_repr_data + offset, src.Y.c0.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.z.c0.mont_repr_data + offset, src.Z.c0.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.x.c1.mont_repr_data + offset, src.X.c1.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.y.c1.mont_repr_data + offset, src.Y.c1.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.z.c1.mont_repr_data + offset, src.Z.c1.mont_repr.data, 32);
      };
      auto copy_t_h = [](const T& src, gpu::alt_bn128_g2& dst, const int offset){
        memcpy(dst.x.c0.mont_repr_data + offset, src.X.c0.mont_repr.data, 32);
        memcpy(dst.y.c0.mont_repr_data + offset, src.Y.c0.mont_repr.data, 32);
        memcpy(dst.z.c0.mont_repr_data + offset, src.Z.c0.mont_repr.data, 32);
        memcpy(dst.x.c1.mont_repr_data + offset, src.X.c1.mont_repr.data, 32);
        memcpy(dst.y.c1.mont_repr_data + offset, src.Y.c1.mont_repr.data, 32);
        memcpy(dst.z.c1.mont_repr_data + offset, src.Z.c1.mont_repr.data, 32);
      };
      auto copy_field = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
      auto copy_field_h = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        memcpy(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };

      const int length = vec.values.size();
      const int ranges_size = ranges.size();
      const int values_size = vec.values.size();
      const int indices_size = vec.indices.size();

      static gpu::alt_bn128_g2 h_values, d_values, d_partial, d_t_zero, d_t_one;
      static gpu::Fp_model h_scalars, d_scalars, d_field_zero, d_field_one, h_non_residue, d_non_residue;

      unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
      //unsigned int chunks = config.num_threads;
      unsigned int chunks = 1;//(254 + c - 1) / c;
      const int instances = gpu::BUCKET_INSTANCES_G2;

      static std::vector<gpu::alt_bn128_g2> d_values2(chunks), d_buckets(chunks), d_buckets2(chunks), d_block_sums(chunks), d_block_sums2(chunks);
      static bool first_init = true;
      static gpu::gpu_meta d_counters, d_counters2, d_index_it, d_firsts, d_seconds, d_bucket_counters, d_starts, d_indexs, d_ids, d_instance_bucket_ids, d_density, d_flags;
      static gpu::gpu_buffer max_value, dmax_value, d_bn_exponents, h_bn_exponents, d_modulus, d_field_modulus;
      static cudaStream_t stream;

    //libff::enter_block("gpu init", false);
      if(first_init){
	gpu::create_stream(&stream);
        d_t_zero.init(1);
        d_t_one.init(1);
        d_field_zero.init(1);
        d_field_one.init(1);
        h_non_residue.init_host(1);
        d_non_residue.init(1);
        max_value.resize_host(1);
        dmax_value.resize(1);
        d_modulus.resize(1);
        d_field_modulus.resize(1);
        for(int i = 0; i < BITS/32; i++){
          max_value.ptr->_limbs[i] = 0xffffffff;
        }
        dmax_value.copy_from_host(max_value);
        copy_t(T::zero(), d_t_zero, 0);
        copy_t(T::one(), d_t_one, 0);
        copy_field(zero, d_field_zero, 0);
        copy_field(one, d_field_one, 0);
        memcpy(h_non_residue.mont_repr_data, value_it[0].X.non_residue.mont_repr.data, 32);
        d_non_residue.copy_from_cpu(h_non_residue);
        gpu::copy_cpu_to_gpu(d_modulus.ptr->_limbs, value_it[0].X.c0.get_modulus().data, 32);
        gpu::copy_cpu_to_gpu(d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32);
        first_init = false;
      }
      h_values.resize_host(values_size);
      d_values.resize(values_size);
      h_scalars.resize_host(scalar_size);
      d_scalars.resize(scalar_size);
      d_partial.resize(ranges_size * gpu::REDUCE_BLOCKS_PER_RANGE * gpu::INSTANCES_PER_BLOCK);
      d_bn_exponents.resize(bn_exponents_size);
      h_bn_exponents.resize_host(bn_exponents_size);
      for(int i = 0; i < chunks; i++){
        d_values2[i].resize(length);
        d_buckets[i].resize((1<<c) * instances);
        d_buckets2[i].resize((1<<c));
        d_block_sums[i].resize((1<<c) / 32);
        d_block_sums2[i].resize((1<<c) / 32/32);
      }
      //if(first_init){
        d_counters.resize(ranges_size * max_depth * sizeof(uint32_t));
        d_counters2.resize(ranges_size * sizeof(uint32_t));
        d_firsts.resize(ranges_size * sizeof(uint32_t));
        d_seconds.resize(ranges_size * sizeof(uint32_t));
        d_bucket_counters.resize((1<<c) * sizeof(int) * chunks);
        d_starts.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        d_indexs.resize((1<<c)*sizeof(int)*chunks * 2);
        d_ids.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        d_instance_bucket_ids.resize((length+1) * sizeof(int) * chunks * 2);
        d_index_it.resize(indices_size * sizeof(size_t));
        d_density.resize(density.size());
        d_flags.resize(scalar_size);
      //}
      std::vector<uint32_t> firsts(ranges_size), seconds(ranges_size);
      for(int i = 0; i < ranges_size; i++){
        firsts[i] = ranges[i].first;
        seconds[i] = ranges[i].second;
      }
    //libff::leave_block("gpu init", false);
    //libff::enter_block("gpu copy", false);
      gpu::copy_cpu_to_gpu(d_firsts.ptr, firsts.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(d_seconds.ptr, seconds.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(d_index_it.ptr, vec.indices.data(), sizeof(size_t) * indices_size, stream);

      uint64_t const_inv = value_it[0].X.c0.inv;
      uint64_t const_field_inv = scalar_start[0].inv;

#pragma omp parallel for
      for(int i = 0; i < values_size; i++){
        copy_t_h(value_it[i], h_values, i);
      }
      d_values.copy_from_cpu(h_values);
#pragma omp parallel for
      for(int i = 0; i < scalar_size; i++){
        copy_field_h(scalar_start[i], h_scalars, i);
      }
      d_scalars.copy_from_cpu(h_scalars);
    //libff::leave_block("gpu copy", false);

    //libff::enter_block("gpu reduce", false);
      gpu::alt_bn128_g2_reduce_sum_one_range(
          d_values, 
          d_scalars, 
          (size_t*)d_index_it.ptr, 
          d_partial, 
          (uint32_t*)d_counters.ptr, 
          (char*)d_flags.ptr,
          ranges_size, 
          (uint32_t*)d_firsts.ptr, (uint32_t*)d_seconds.ptr,
          dmax_value.ptr, d_t_zero, d_field_zero, d_field_one, d_non_residue, (char*)d_density.ptr, d_bn_exponents.ptr, 
          d_modulus.ptr, const_inv, d_field_modulus.ptr, const_field_inv, max_depth, stream);
      //d_bn_exponents.copy_to_host(h_bn_exponents);
      //for(int i = 0; i < bn_exponents.size(); i++){
      //  memcpy(bn_exponents[i].data, h_bn_exponents.ptr + i, 32);
      //}

      auto copy_back = [&](T& dst, const gpu::alt_bn128_g2& src, const int offset){
        gpu::copy_gpu_to_cpu(dst.X.c0.mont_repr.data, src.x.c0.mont_repr_data + offset, 32, stream);
        gpu::copy_gpu_to_cpu(dst.Y.c0.mont_repr.data, src.y.c0.mont_repr_data + offset, 32, stream);
        gpu::copy_gpu_to_cpu(dst.Z.c0.mont_repr.data, src.z.c0.mont_repr_data + offset, 32, stream);
        gpu::copy_gpu_to_cpu(dst.X.c1.mont_repr.data, src.x.c1.mont_repr_data + offset, 32, stream);
        gpu::copy_gpu_to_cpu(dst.Y.c1.mont_repr.data, src.y.c1.mont_repr_data + offset, 32, stream);
        gpu::copy_gpu_to_cpu(dst.Z.c1.mont_repr.data, src.z.c1.mont_repr_data + offset, 32, stream);
      };
      T gpu_acc;
      copy_back(gpu_acc, d_partial, 0);
    //libff::leave_block("gpu reduce", false);

      //auto tmp = gpu_acc + 
      //d_bn_exponents.copy_to_host(h_bn_exponents);
      //for(int i = 0; i < bn_exponents.size(); i++){
      //  memcpy(bn_exponents[i].data, h_bn_exponents.ptr + i, 32);
      //}
      //std::vector<char> h_density(density.size());
      //gpu::copy_gpu_to_cpu(h_density.data(), d_density.ptr, density.size());
      //for(int i = 0; i < density.size(); i++){
      //  density[i] = (h_density[i] == 1);
      //}
      //auto exp_out = libff::multi_exp_with_density<T, FieldT, true, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
      //gpu::copy_cpu_to_gpu(d_bn_exponents.ptr, bn_exponents.data(), 32 * bn_exponents.size());
      auto exp_out = libff::multi_exp_with_density_g2_gpu<T, FieldT, false, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config, d_values, (char*)d_density.ptr, d_bn_exponents, dmax_value, d_modulus, d_t_zero, d_non_residue, d_values2, d_buckets, d_buckets2, d_block_sums, d_block_sums2, (int*)d_bucket_counters.ptr, (int*)d_starts.ptr, (int*)d_indexs.ptr, (int*)d_ids.ptr, (int*)d_instance_bucket_ids.ptr, stream);
      auto tmp = gpu_acc + exp_out;

      //release gpu memory
      if(false){
        d_values.release();
        d_partial.release();
        for(int i = 0; i < chunks; i++){
          d_values2[i].release();
          d_buckets[i].release();
          d_buckets2[i].release();
          d_block_sums[i].release();
          d_block_sums2[i].release();
        }
        d_scalars.release();
        d_field_one.release();
        d_field_zero.release();
        d_density.release();
        d_flags.release();
        d_counters.release();
        d_counters2.release();
        d_index_it.release();
        d_firsts.release();
        d_seconds.release();
        d_bucket_counters.release();
        d_starts.release();
        d_indexs.release();
        d_ids.release();
        d_instance_bucket_ids.release();
        d_bn_exponents.release();
        dmax_value.release();
        d_modulus.release();
        d_field_modulus.release();
        d_t_zero.release();
      }
      return tmp;
    }

    //libff::leave_block("Process scalar vector");
    //return tmp;
}

template<typename T, typename FieldT, libff::multi_exp_method Method>
T gpu_kc_multi_exp_with_mixed_addition_g2_mcl(const sparse_vector<T> &vec,
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
                //result = result + value_it[i];
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

    //T acc = T::zero();
    //unsigned int totalCount = 0;
    //for (unsigned int i = 0; i < ranges.size(); i++)
    //{
    //    acc = acc + partial[i];
    //    totalCount += counters[i];
    //}
    //libff::leave_block("evaluation reduce");
    //libff::leave_block("reduce");

    //libff::leave_block("Process scalar vector");

    //return acc + libff::multi_exp_with_density<T, FieldT, true, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
    //call gpu
    if(true){
#ifdef MCL_GPU
        int max_depth = 0, min_depth = 30130;
        for(int i = 0; i < ranges.size(); i++){
            max_depth = std::max(max_depth, (int)(ranges[i].second-ranges[i].first));
        }

      auto copy_t = [](const T& src, gpu::mcl_bn128_g2& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.x.c0.mont_repr_data + offset, src.pt.x.a.getUnit(), 32);
        gpu::copy_cpu_to_gpu(dst.x.c1.mont_repr_data + offset, src.pt.x.b.getUnit(), 32);
        gpu::copy_cpu_to_gpu(dst.y.c0.mont_repr_data + offset, src.pt.y.a.getUnit(), 32);
        gpu::copy_cpu_to_gpu(dst.y.c1.mont_repr_data + offset, src.pt.y.b.getUnit(), 32);
        gpu::copy_cpu_to_gpu(dst.z.c0.mont_repr_data + offset, src.pt.z.a.getUnit(), 32);
        gpu::copy_cpu_to_gpu(dst.z.c1.mont_repr_data + offset, src.pt.z.b.getUnit(), 32);
      };
      auto copy_t_h = [](const T& src, gpu::mcl_bn128_g2& dst, const int offset){
        memcpy(dst.x.c0.mont_repr_data + offset, src.pt.x.a.getUnit(), 32);
        memcpy(dst.x.c1.mont_repr_data + offset, src.pt.x.b.getUnit(), 32);
        memcpy(dst.y.c0.mont_repr_data + offset, src.pt.y.a.getUnit(), 32);
        memcpy(dst.y.c1.mont_repr_data + offset, src.pt.y.b.getUnit(), 32);
        memcpy(dst.z.c0.mont_repr_data + offset, src.pt.z.a.getUnit(), 32);
        memcpy(dst.z.c1.mont_repr_data + offset, src.pt.z.b.getUnit(), 32);
        //dst.x.a.copy((uint64_t*)src.x.c0.mont_repr_data + offset);
      };
      auto copy_field = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
      auto copy_field_h = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        memcpy(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
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

      static gpu::mcl_bn128_g2 h_values, d_values, d_partial, d_t_zero, d_t_one;
      static std::vector<gpu::mcl_bn128_g2> d_values2(chunks), d_buckets(chunks), d_buckets2(chunks), d_block_sums(chunks), d_block_sums2(chunks);
      static gpu::Fp_model h_scalars, d_scalars, d_field_zero, d_field_one;

      static gpu::gpu_meta d_counters, d_counters2, d_index_it, d_firsts, d_seconds, d_bucket_counters, d_starts, d_indexs, d_ids, d_instance_bucket_ids, d_density, d_flags;
      static gpu::gpu_buffer max_value, dmax_value, d_bn_exponents, h_bn_exponents, d_modulus, d_field_modulus;
      static bool first_init = true;
      static cudaStream_t stream;
      static gpu::Fp_model d_one, d_p;
      static gpu::Fp_model2 d_a;

    //libff::enter_block("gpu malloc...");
      if(first_init){
          gpu::create_stream(&stream);
        d_t_zero.init(1);
        d_t_one.init(1);
        d_field_zero.init(1);
        d_field_one.init(1);
        d_modulus.resize(1);
        d_field_modulus.resize(1);
        copy_t(T::zero(), d_t_zero, 0);
        copy_t(T::one(), d_t_one, 0);
        copy_field(zero, d_field_zero, 0);
        copy_field(one, d_field_one, 0);
        gpu::copy_cpu_to_gpu(d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32);
        d_one.init(1);
        d_p.init(1);
        d_a.init(1);
        gpu::copy_cpu_to_gpu(d_one.mont_repr_data, value_it[0].pt.z.a.one().getUnit(), 32);
        gpu::copy_cpu_to_gpu(d_p.mont_repr_data, value_it[0].pt.z.getOp().p, 32);
        gpu::copy_cpu_to_gpu(d_a.c0.mont_repr_data, value_it[0].pt.a_.a.getUnit(), 32);
        gpu::copy_cpu_to_gpu(d_a.c1.mont_repr_data, value_it[0].pt.a_.b.getUnit(), 32);
        first_init = false;
      }
      h_values.resize_host(values_size);
      d_values.resize(values_size);
      h_scalars.resize_host(scalar_size);
      d_scalars.resize(scalar_size);
      d_partial.resize(ranges_size * gpu::REDUCE_BLOCKS_PER_RANGE * gpu::INSTANCES_PER_BLOCK);
      d_bn_exponents.resize(bn_exponents.size());
      h_bn_exponents.resize_host(bn_exponents.size());
      for(int i = 0; i < chunks; i++){
        d_values2[i].resize(length);
        d_buckets[i].resize((1<<c) * instances);
        d_buckets2[i].resize((1<<c));
        d_block_sums[i].resize((1<<c) / 32);
        d_block_sums2[i].resize((1<<c) / 32/32);
      }
      //if(first_init)
      {
        d_counters.resize(ranges_size * max_depth * sizeof(uint32_t));
        d_counters2.resize(ranges_size * sizeof(uint32_t));
        d_firsts.resize(ranges_size * sizeof(uint32_t));
        d_seconds.resize(ranges_size * sizeof(uint32_t));
        d_bucket_counters.resize((1<<c) * sizeof(int) * chunks);
        d_starts.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        d_indexs.resize((1<<c)*sizeof(int)*chunks*2);
        d_ids.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
        d_instance_bucket_ids.resize((length+1) * sizeof(int) * chunks * 2);
        d_index_it.resize(indices_size * sizeof(size_t));
        d_density.resize(density.size());
        d_flags.resize(scalar_size);
      }
      std::vector<uint32_t> firsts(ranges_size), seconds(ranges_size);
      for(int i = 0; i < ranges_size; i++){
        firsts[i] = ranges[i].first;
        seconds[i] = ranges[i].second;
      }
    //libff::leave_block("gpu malloc...");
    //libff::enter_block("gpu copy...");
      gpu::copy_cpu_to_gpu(d_firsts.ptr, firsts.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(d_seconds.ptr, seconds.data(), sizeof(uint32_t) * ranges_size, stream);
      gpu::copy_cpu_to_gpu(d_index_it.ptr, vec.indices.data(), sizeof(size_t) * indices_size, stream);

      //uint64_t const_inv = FieldT::getOp().rp;
      uint64_t const_field_inv = scalar_start[0].inv;

      //const auto& modu = value_it[0].X.get_modulus();
#pragma omp parallel for
      for(int i = 0; i < values_size; i++){
        copy_t_h(value_it[i], h_values, i);
      }
      d_values.copy_from_cpu(h_values);
#pragma omp parallel for
      for(int i = 0; i < scalar_size; i++){
        copy_field_h(scalar_start[i], h_scalars, i);
      }
      d_scalars.copy_from_cpu(h_scalars);

      gpu::mcl_bn128_g2_reduce_sum(
          d_values, 
          d_scalars, 
          (size_t*)d_index_it.ptr, 
          d_partial, 
         (uint32_t*)d_counters.ptr, 
          (char*)d_flags.ptr,
          ranges_size, 
          (uint32_t*)d_firsts.ptr, (uint32_t*)d_seconds.ptr,
          d_t_zero, d_field_zero, d_field_one, (char*)d_density.ptr, d_bn_exponents.ptr, 
          d_field_modulus.ptr, const_field_inv, 
          d_one, d_p, d_a, value_it[0].pt.specialA_, value_it[0].pt.mode_,value_it[0].pt.x.getOp().rp,
          max_depth, stream);
        gpu::sync(stream);

      auto copy_back = [&](T& dst, const gpu::mcl_bn128_g2& src, const int offset){
        uint64_t tmp[4];
        gpu::copy_gpu_to_cpu(tmp, src.x.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.x.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.x.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.x.b.copy(tmp);
        
        gpu::copy_gpu_to_cpu(tmp, src.y.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.y.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.y.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.y.b.copy(tmp);

        gpu::copy_gpu_to_cpu(tmp, src.z.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.z.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.z.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.z.b.copy(tmp);
      };

      T gpu_acc;
      copy_back(gpu_acc, d_partial, 0);

      T exp_out = libff::multi_exp_with_density_gpu_mcl_g2<T, FieldT, false, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config, d_values, (char*)d_density.ptr, d_bn_exponents, dmax_value, d_modulus, d_t_zero, d_values2, d_buckets, d_buckets2, d_block_sums, d_block_sums2, (int*)d_bucket_counters.ptr, (int*)d_starts.ptr, (int*)d_indexs.ptr, (int*)d_ids.ptr, (int*)d_instance_bucket_ids.ptr, d_one, d_p, d_a, stream);
      
      auto tmp = gpu_acc + exp_out;
      return tmp;
#endif
    }
}

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
template<typename T, typename FieldT, libff::multi_exp_method Method>
void kc_multi_exp_with_mixed_addition_mcl_preprocess(const sparse_vector<T> &vec,
                                    typename std::vector<FieldT>::const_iterator scalar_start,
                                    typename std::vector<FieldT>::const_iterator scalar_end,
                                    std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                    const Config& config,
                                    libff::GpuMclData& gpu_mcl_data)
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

#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        T result = T::zero();
        //unsigned int count = 0;
#pragma omp parallel for
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
            }
            else
            {
                density[i] = true;
                bn_exponents[i] = scalar.as_bigint();
                //++count;
                //++num_other;
            }
        }
        //partial[j] = result; 
        //counters[j] = count;
    }

    //call gpu
    if(true){
#ifdef MCL_GPU

        int max_depth = 0, min_depth = 30130;
        for(int i = 0; i < ranges.size(); i++){
            max_depth = std::max(max_depth, (int)(ranges[i].second-ranges[i].first));
        }

      auto copy_t = [](const T& src, gpu::mcl_bn128_g1& dst, const int offset, gpu::CudaStream stream){
        gpu::copy_cpu_to_gpu(dst.x.mont_repr_data + offset, src.pt.x.getUnit(), 32, stream);
        gpu::copy_cpu_to_gpu(dst.y.mont_repr_data + offset, src.pt.y.getUnit(), 32, stream);
        gpu::copy_cpu_to_gpu(dst.z.mont_repr_data + offset, src.pt.z.getUnit(), 32, stream);
      };
      auto copy_t_h = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
        memcpy(dst.x.mont_repr_data + offset, src.pt.x.getUnit(), 32);
        memcpy(dst.y.mont_repr_data + offset, src.pt.y.getUnit(), 32);
        memcpy(dst.z.mont_repr_data + offset, src.pt.z.getUnit(), 32);
      };
      auto copy_field = [](const FieldT& src, gpu::Fp_model& dst, const int offset, gpu::CudaStream stream){
        gpu::copy_cpu_to_gpu(dst.mont_repr_data + offset, src.mont_repr.data, 32, stream);
      };
      auto copy_field_h = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        memcpy(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
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

      if(true){
        copy_t(T::zero(), gpu_mcl_data.d_t_zero, 0, stream);
        copy_t(T::one(), gpu_mcl_data.d_t_one, 0, stream);
        copy_field(zero, gpu_mcl_data.d_field_zero, 0, stream);
        copy_field(one, gpu_mcl_data.d_field_one, 0, stream);
        gpu::copy_cpu_to_gpu(gpu_mcl_data.d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32, stream);
        gpu::copy_cpu_to_gpu(gpu_mcl_data.d_one.mont_repr_data, value_it[0].pt.z.one().getUnit(), 32, stream);
        gpu::copy_cpu_to_gpu(gpu_mcl_data.d_p.mont_repr_data, value_it[0].pt.z.getOp().p, 32, stream);
        gpu::copy_cpu_to_gpu(gpu_mcl_data.d_a.mont_repr_data, value_it[0].pt.a_.getUnit(), 32, stream);
      }
      gpu_mcl_data.h_values.resize_host(values_size);
      gpu_mcl_data.d_values.resize(values_size);
      gpu_mcl_data.h_scalars.resize_host(scalar_size);
      gpu_mcl_data.d_scalars.resize(scalar_size);
      gpu_mcl_data.d_partial.resize(values_size);
      gpu_mcl_data.d_bn_exponents.resize(bn_exponents.size());
      gpu_mcl_data.h_bn_exponents.resize_host(bn_exponents.size());
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
        copy_t_h(value_it[i], gpu_mcl_data.h_values, i);
      }
      gpu_mcl_data.d_values.copy_from_cpu(gpu_mcl_data.h_values, stream);
#pragma omp parallel for
      for(int i = 0; i < scalar_size; i++){
        copy_field_h(scalar_start[i], gpu_mcl_data.h_scalars, i);
      }
      gpu_mcl_data.d_scalars.copy_from_cpu(gpu_mcl_data.h_scalars, stream);
      for(int i = 0; i < bn_exponents.size(); i++){
        memcpy(gpu_mcl_data.h_bn_exponents.ptr + i, bn_exponents[i].data, 32);
      }
      gpu_mcl_data.d_bn_exponents.copy_from_host(gpu_mcl_data.h_bn_exponents, stream);

      std::vector<char> h_density(density.size());
      for(int i = 0; i<density.size(); i++){
        h_density[i] = density[i] ? 1 : 0;
      }
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_density.ptr, h_density.data(), density.size(), stream);
#endif
    }
}

template<typename T, typename FieldT, libff::multi_exp_method Method>
T kc_multi_exp_with_mixed_addition_mcl(const sparse_vector<T> &vec,
                                    typename std::vector<FieldT>::const_iterator scalar_start,
                                    typename std::vector<FieldT>::const_iterator scalar_end,
                                    std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                    const Config& config,
                                    libff::GpuMclData& gpu_mcl_data)
{
      const size_t scalar_length = vec.indices.size();
      auto ranges = libsnark::get_cpu_ranges(0, scalar_length);
      const int ranges_size = ranges.size();
      uint64_t const_field_inv = scalar_start[0].inv;
      auto index_it = vec.indices.begin();
      auto value_it = vec.values.begin();
      int max_depth = 0, min_depth = 30130;
      for(int i = 0; i < ranges.size(); i++){
          max_depth = std::max(max_depth, (int)(ranges[i].second-ranges[i].first));
      }
      gpu::CudaStream& stream = gpu_mcl_data.stream;
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
          max_depth, stream);
          gpu::sync(stream);

      auto copy_back = [&](T& dst, const gpu::mcl_bn128_g1& src, const int offset){
        uint64_t tmp[4];
        gpu::copy_gpu_to_cpu(tmp, src.x.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.x.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.y.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.y.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.z.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.z.copy(tmp);
      };

      T gpu_acc;
      copy_back(gpu_acc, gpu_mcl_data.d_partial, 0);

      //auto exp_out = libff::multi_exp_with_density_test<T, FieldT, true, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
      T exp_out = libff::multi_exp_with_density_gpu_mcl<T, FieldT, false, Method>(vec.values.begin(), vec.values.end(), config, gpu_mcl_data.d_values, (char*)gpu_mcl_data.d_density.ptr, gpu_mcl_data.d_bn_exponents, gpu_mcl_data.dmax_value, gpu_mcl_data.d_modulus, gpu_mcl_data.d_t_zero, gpu_mcl_data.d_values2, gpu_mcl_data.d_buckets, gpu_mcl_data.d_buckets2, gpu_mcl_data.d_block_sums, gpu_mcl_data.d_block_sums2, (int*)gpu_mcl_data.d_bucket_counters.ptr, (int*)gpu_mcl_data.d_starts.ptr, (int*)gpu_mcl_data.d_indexs.ptr, (int*)gpu_mcl_data.d_ids.ptr, (int*)gpu_mcl_data.d_instance_bucket_ids.ptr, gpu_mcl_data.d_one, gpu_mcl_data.d_p, gpu_mcl_data.d_a, stream);
      
      auto tmp = gpu_acc + exp_out;
      return tmp;
}

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
