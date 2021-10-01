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
#include "cgbn_alt_bn128_g1.h"


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
    libff::enter_block("Process scalar vector");
    auto index_it = vec.indices.begin();
    auto value_it = vec.values.begin();

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();

    //size_t num_skip = 0;
    //size_t num_add = 0;
    //size_t num_other = 0;

    const size_t scalar_size = std::distance(scalar_start, scalar_end);
    const size_t scalar_length = vec.indices.size();

    libff::enter_block("allocate density memory");
    std::vector<bool> density(scalar_length);
    libff::leave_block("allocate density memory");

    std::vector<libff::bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    if (bn_exponents.size() < scalar_length)
    {
        bn_exponents.resize(scalar_length);
    }

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);

    libff::enter_block("find max index");
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
    libff::leave_block("find max index");

    ranges = get_cpu_ranges(0, actual_max_idx);

    std::vector<T> partial(ranges.size(), T::zero());
    std::vector<unsigned int> counters(ranges.size(), 0);

#ifdef MULTICORE
    //#pragma omp parallel for
#endif
    /*
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
                //result = result.mixed_add(value_it);
#else
                result = result.gpu_add(value_it[i]);
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
    */
    libff::enter_block("cpu reduce sum");
    int max_depth = 30130, min_depth = 30130;
    const int local_instances = 64 * BlockDepth;
    const int blocks = (max_depth + local_instances - 1) / local_instances;
#pragma omp parallel for
      /*
    for (size_t r = 0; r < ranges.size(); r++)
    {
      //std::vector<T> results(instances);
      unsigned int count = 0;
      int first = ranges[r].first;
      int second = ranges[r].second;
      T result = T::zero();
      for(int b = 0; b < blocks; b++){
        T b_result = T::zero();
        int b_count = 0;
        for(int i = 0; i < 64; i++){
          T tmp_result = T::zero();
          int tmp_count = 0;
          
          for(int j = b * 64 + i; j < second - first; j += blocks * 64){
            const FieldT scalar = scalar_start[index_it[j + first]];
            if (scalar == zero)
            {
            }
            else if (scalar == one)
            {
              tmp_result = tmp_result + value_it[j + first];
            }
            else
            {
              density[first + j] = true;
              bn_exponents[first+j] = scalar.as_bigint();
              ++tmp_count;
            }
          }
          b_result = b_result + tmp_result;
          b_count += tmp_count;
        }
        result = result + b_result;
        count += b_count;
      }
      partial[r] = result;
      counters[r] = count;
      */
      /*
      for(int instance = 0; instance < instances; instance++){
        results[j * instances + instance] = T::zero(); 
        for(int i = instance + ranges[j].first; i < ranges[j].second; i+= instances){
          const FieldT scalar = scalar_start[index_it[i]];
          if (scalar == zero)
          {
          }
          else if (scalar == one)
          {
            results[j * instances + instance] = results[j * instances + instance] + value_it[i];
          }
          else
          {
            density[i] = true;
            bn_exponents[i] = scalar.as_bigint();
            ++count;
          }
        }
      }
      T result = T::zero();
      for(int instance = 0; instance < instances; instance++){
        result = result + results[j * instances + instance];
      }
      partial[j] = result;
      counters[j] = count;
      }
      */
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
                //result = result.mixed_add(value_it);
#else
                //result = result.gpu_add(value_it[i]);
                result = result + value_it[i];
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
    libff::leave_block("cpu reduce sum");
    T acc = T::zero();
    unsigned int totalCount = 0;
    for (unsigned int i = 0; i < ranges.size(); i++)
    {
        acc = acc + partial[i];
        totalCount += counters[i];
    }

    //printf("max_depth = %d, min_depth=%d\n", max_depth, min_depth);
    /***********start alt_bn128_g1 gpu reduce sum*****************/
    if(true){
      const int local_instances = BlockDepth * 64;
      const int blocks = (max_depth + local_instances - 1) / local_instances;
      gpu::alt_bn128_g1 d_values, d_partial, d_partial2, d_t_zero, d_t_one;
      gpu::Fp_model d_scalars, d_field_zero, d_field_one;
      const int ranges_size = ranges.size();
      const int values_size = vec.values.size();
      const int indices_size = vec.indices.size();
      {
        d_values.init(values_size);
        d_scalars.init(scalar_size);
        d_partial.init(ranges_size * blocks * 64);
        d_partial2.init(ranges_size);
        d_t_zero.init(1);
        d_t_one.init(1);
        d_field_zero.init(1);
        d_field_one.init(1);
      }

      uint32_t *d_counters, *d_counters2;
      size_t* d_index_it;
      uint32_t *d_firsts, *d_seconds;
      char *d_density;
      gpu::gpu_buffer max_value, dmax_value, d_bn_exponents, h_bn_exponents, d_modulus, d_field_modulus;
      {
        gpu::gpu_malloc((void**)&d_counters, ranges_size  * blocks * sizeof(uint32_t));
        gpu::gpu_malloc((void**)&d_counters2, ranges_size  * sizeof(uint32_t));
        gpu::gpu_malloc((void**)&d_index_it, indices_size * sizeof(size_t));
        gpu::gpu_malloc((void**)&d_firsts, ranges_size  * sizeof(uint32_t));
        gpu::gpu_malloc((void**)&d_seconds, ranges_size  * sizeof(uint32_t));
        max_value.resize_host(1);
        dmax_value.resize(1);
        d_modulus.resize(1);
        d_field_modulus.resize(1);
        for(int i = 0; i < BITS/32; i++){
          max_value.ptr->_limbs[i] = 0xffffffff;
        }
        dmax_value.copy_from_host(max_value);
        std::vector<uint32_t> firsts(ranges_size), seconds(ranges_size);
        for(int i = 0; i < ranges_size; i++){
          firsts[i] = ranges[i].first;
          seconds[i] = ranges[i].second;
        }
        gpu::gpu_malloc((void**)&d_density, density.size() * sizeof(char));
        d_bn_exponents.resize(bn_exponents.size());
        h_bn_exponents.resize_host(bn_exponents.size());
        gpu::copy_cpu_to_gpu(d_index_it, vec.indices.data(), sizeof(size_t) * indices_size);
        gpu::copy_cpu_to_gpu(d_firsts, firsts.data(), sizeof(uint32_t) * ranges_size);
        gpu::copy_cpu_to_gpu(d_seconds, seconds.data(), sizeof(uint32_t) * ranges_size);
      }


      auto copy_t = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.x.mont_repr_data + offset, src.X.mont_repr.data, 32);

        gpu::copy_cpu_to_gpu(dst.y.mont_repr_data + offset, src.Y.mont_repr.data, 32);

        gpu::copy_cpu_to_gpu(dst.z.mont_repr_data + offset, src.Z.mont_repr.data, 32);
      };
      auto copy_field = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
      gpu::copy_cpu_to_gpu(d_modulus.ptr->_limbs, value_it[0].X.get_modulus().data, 32);
      gpu::copy_cpu_to_gpu(d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32);
      uint64_t const_inv = value_it[0].X.inv;
      uint64_t const_field_inv = scalar_start[0].inv;

      const auto& modu = value_it[0].X.get_modulus();
      for(int i = 0; i < values_size; i++){
        copy_t(value_it[i], d_values, i);
      }
      for(int i = 0; i < scalar_size; i++){
        copy_field(scalar_start[i], d_scalars, i);
      }
      copy_t(T::zero(), d_t_zero, 0);
      copy_t(T::one(), d_t_one, 0);
      copy_field(zero, d_field_zero, 0);
      copy_field(one, d_field_one, 0);

      gpu::init_error_report();
      gpu::warm_up();
      libff::enter_block("gpu reduce sum");
      gpu::alt_bn128_g1_reduce_sum_one_range(
          d_values, 
          d_scalars, 
          d_index_it, 
          d_partial, 
          d_counters, 
          ranges_size, 
          d_firsts, d_seconds,
          dmax_value.ptr, d_t_zero, d_field_zero, d_field_one, d_density, d_bn_exponents.ptr, 
          d_modulus.ptr, const_inv, d_field_modulus.ptr, const_field_inv, max_depth);
      alt_bn128_g1_reduce_sum(
          d_partial, 
          d_counters, 
          d_partial2, 
          d_counters2, 
          ranges_size, 
          dmax_value.ptr, d_t_zero, d_modulus.ptr, const_inv, max_depth);
      alt_bn128_g1_reduce_sum_one_instance(
          d_partial2, 
          d_counters2, 
          d_partial, 
          d_counters, 
          dmax_value.ptr, d_t_zero, d_modulus.ptr, const_inv, ranges_size);

      //std::vector<int> h_counters(ranges_size);
      //gpu::copy_gpu_to_cpu(h_counters.data(), d_counters2, ranges_size * sizeof(int));
      //gpu::copy_gpu_to_cpu(counters.data(), d_counters, ranges_size * sizeof(int));
      int gpu_total_count = 0;
      gpu::copy_gpu_to_cpu(&gpu_total_count, d_counters, sizeof(int));
      if(gpu_total_count != totalCount){
        printf("compare total count failed...\n");
      }

      auto copy_back = [&](T& dst, const gpu::alt_bn128_g1& src, const int offset){
        gpu::copy_gpu_to_cpu(dst.X.mont_repr.data, src.x.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.Y.mont_repr.data, src.y.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.Z.mont_repr.data, src.z.mont_repr_data + offset, 32);
      };

      //for(int i = 0; i < ranges_size; i++){
      //  if(h_counters[i] != counters[i]){
      //    printf("compare counter failed..%d.\n", i);
      //    break;
      //  }
      //}
      //std::vector<T> h_partial(ranges.size(), T::zero());
      //for(int r = 0; r < ranges_size; r++){
      //  T value; 
      //  copy_back(value, d_partial2, r);
      //  //if(value != partial[r]){
      //  //  printf("compare partial failed..%d.\n", r);
      //  //  break;
      //  //}
      //  partial[r] = value;
      //}
      T gpu_acc;
      copy_back(gpu_acc, d_partial, 0);
      libff::leave_block("gpu reduce sum");
      if(gpu_acc != acc){
        printf("compare acc failed..\n");
      }

      char *h_density = new char[density.size()];
      gpu::copy_gpu_to_cpu(h_density, d_density, density.size() * sizeof(char));
      for(int i = 0; i < density.size(); i++){
        //density[i] = h_density[i];
        char a = 1;
        if(density[i] == false) a = 0;
        if(a != h_density[i]){
          printf("compare density failed %d\n", i);
        }
      }
      printf("cpu gpu compare success\n");
      d_bn_exponents.copy_to_host(h_bn_exponents);
      for(int i = 0; i < bn_exponents.size(); i++){
        if(density[i] && memcmp(h_bn_exponents.ptr[i]._limbs, bn_exponents[i].data, 32) != 0){
          printf("bn_exponents %d no equal..\n", i);
          break;
        }
        memcpy(bn_exponents[i].data, h_bn_exponents.ptr[i]._limbs, 32);
      }

      free(h_density);
      gpu::gpu_free(d_counters);
      gpu::gpu_free(d_index_it);
      gpu::gpu_free(d_firsts);
      gpu::gpu_free(d_seconds);
      gpu::gpu_free(d_density);
      dmax_value.release();
      max_value.release_host();
      d_values.release();
      d_partial.release();
      d_t_zero.release();
      d_t_one.release();
      d_field_zero.release();
      d_field_one.release();
      d_bn_exponents.release();
      h_bn_exponents.release_host();
    }

    /***********end alt_bn128_g1 gpu reduce sum*****************/

    libff::leave_block("Process scalar vector");

    return acc + libff::multi_exp_with_density<T, FieldT, false, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
}

template<typename T, typename FieldT, libff::multi_exp_method Method>
T kc_multi_exp_with_mixed_addition(const sparse_vector<T> &vec,
                                    typename std::vector<FieldT>::const_iterator scalar_start,
                                    typename std::vector<FieldT>::const_iterator scalar_end,
                                    std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                    const Config& config)
{
    libff::enter_block("Process scalar vector");
    auto index_it = vec.indices.begin();
    auto value_it = vec.values.begin();

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();

    //size_t num_skip = 0;
    //size_t num_add = 0;
    //size_t num_other = 0;

    const size_t scalar_size = std::distance(scalar_start, scalar_end);
    const size_t scalar_length = vec.indices.size();

    libff::enter_block("allocate density memory");
    std::vector<bool> density(scalar_length);
    libff::leave_block("allocate density memory");

    std::vector<libff::bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    if (bn_exponents.size() < scalar_length)
    {
        bn_exponents.resize(scalar_length);
    }

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);

    libff::enter_block("find max index");
    std::vector<unsigned int> partial_max_indices(ranges.size(), 0xffffffff);
    printf("ranges.size = %d\n", ranges.size());
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
    libff::leave_block("find max index");

    ranges = get_cpu_ranges(0, actual_max_idx);

    std::vector<T> partial(ranges.size(), T::zero());
    std::vector<unsigned int> counters(ranges.size(), 0);

#ifdef MULTICORE
    //#pragma omp parallel for
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

    libff::leave_block("Process scalar vector");

    return acc + libff::multi_exp_with_density<T, FieldT, true, Method>(vec.values.begin(), vec.values.end(), bn_exponents, density, config);
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
