/** @file
 *****************************************************************************

 Implementation of interfaces for a R1CS-to-QAP reduction.

 See r1cs_to_qap.hpp .

 *****************************************************************************
 * @author     This file is part of libsnark, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef R1CS_TO_QAP_TCC_
#define R1CS_TO_QAP_TCC_

#include <libff/common/profiling.hpp>
#include <libff/common/utils.hpp>
#include <libfqfft/evaluation_domain/get_evaluation_domain.hpp>

#include "cgbn_math.h"
#include "cgbn_fp.h"
//#include "cgbn_fp2.h"
#include "cgbn_alt_bn128_g1.h"
//#include "cgbn_alt_bn128_g2.h"
//#include "low_func_gpu.h"
#include <cuda_runtime.h>

namespace libsnark {

/**
 * Instance map for the R1CS-to-QAP reduction.
 *
 * Namely, given a R1CS constraint system cs, construct a QAP instance for which:
 *   A := (A_0(z),A_1(z),...,A_m(z))
 *   B := (B_0(z),B_1(z),...,B_m(z))
 *   C := (C_0(z),C_1(z),...,C_m(z))
 * where
 *   m = number of variables of the QAP
 * and
 *   each A_i,B_i,C_i is expressed in the Lagrange basis.
 */
template<typename FieldT>
qap_instance<FieldT> r1cs_to_qap_instance_map(const r1cs_constraint_system<FieldT> &cs)
{
    libff::enter_block("Call to r1cs_to_qap_instance_map");

    const std::shared_ptr<libfqfft::evaluation_domain<FieldT> > domain = libfqfft::get_evaluation_domain<FieldT>(cs.num_constraints() + cs.num_inputs() + 1);

    std::vector<std::map<size_t, FieldT> > A_in_Lagrange_basis(cs.num_variables()+1);
    std::vector<std::map<size_t, FieldT> > B_in_Lagrange_basis(cs.num_variables()+1);
    std::vector<std::map<size_t, FieldT> > C_in_Lagrange_basis(cs.num_variables()+1);

    libff::enter_block("Compute polynomials A, B, C in Lagrange basis");
    /**
     * add and process the constraints
     *     input_i * 0 = 0
     * to ensure soundness of input consistency
     */
    for (size_t i = 0; i <= cs.num_inputs(); ++i)
    {
        A_in_Lagrange_basis[i][cs.num_constraints() + i] = FieldT::one();
    }
    /* process all other constraints */
    for (size_t i = 0; i < cs.num_constraints(); ++i)
    {
        auto a = cs.constraints[i].a.getTerms();
        for (size_t j = 0; j < a.size(); ++j)
        {
            A_in_Lagrange_basis[a[j].index][i] += a[j].coeff;
        }

        auto b = cs.constraints[i].b.getTerms();
        for (size_t j = 0; j < cs.constraints[i].b.getTerms().size(); ++j)
        {
            B_in_Lagrange_basis[b[j].index][i] += b[j].coeff;
        }

        auto c = cs.constraints[i].c.getTerms();
        for (size_t j = 0; j < c.size(); ++j)
        {
            C_in_Lagrange_basis[c[j].index][i] += c[j].coeff;
        }
    }
    libff::leave_block("Compute polynomials A, B, C in Lagrange basis");

    libff::leave_block("Call to r1cs_to_qap_instance_map");

    return qap_instance<FieldT>(domain,
                                cs.num_variables(),
                                domain->m,
                                cs.num_inputs(),
                                std::move(A_in_Lagrange_basis),
                                std::move(B_in_Lagrange_basis),
                                std::move(C_in_Lagrange_basis));
}

static unsigned int roundUpToNearestPowerOf2(unsigned int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

/**
 * Instance map for the R1CS-to-QAP reduction followed by evaluation of the resulting QAP instance.
 *
 * Namely, given a R1CS constraint system cs and a field element t, construct
 * a QAP instance (evaluated at t) for which:
 *   At := (A_0(t),A_1(t),...,A_m(t))
 *   Bt := (B_0(t),B_1(t),...,B_m(t))
 *   Ct := (C_0(t),C_1(t),...,C_m(t))
 *   Ht := (1,t,t^2,...,t^n)
 *   Zt := Z(t) = "vanishing polynomial of a certain set S, evaluated at t"
 * where
 *   m = number of variables of the QAP
 *   n = degree of the QAP
 */
template<typename FieldT>
qap_instance_evaluation<FieldT> r1cs_to_qap_instance_map_with_evaluation(const r1cs_constraint_system<FieldT> &cs,
                                                                         const FieldT &t)
{
    libff::enter_block("Call to r1cs_to_qap_instance_map_with_evaluation");

    const std::shared_ptr<libfqfft::evaluation_domain<FieldT> > domain = libfqfft::get_evaluation_domain<FieldT>(roundUpToNearestPowerOf2(cs.num_constraints() + cs.num_inputs() + 1));

    std::vector<FieldT> At, Bt, Ct, Ht;

    At.resize(cs.num_variables()+1, FieldT::zero());
    Bt.resize(cs.num_variables()+1, FieldT::zero());
    Ct.resize(cs.num_variables()+1, FieldT::zero());
    Ht.reserve(domain->m+1);

    const FieldT Zt = domain->compute_vanishing_polynomial(t);

    libff::enter_block("Compute evaluations of A, B, C, H at t");
    const std::vector<FieldT> u = domain->evaluate_all_lagrange_polynomials(t);
    /**
     * add and process the constraints
     *     input_i * 0 = 0
     * to ensure soundness of input consistency
     */
    for (size_t i = 0; i <= cs.num_inputs(); ++i)
    {
        At[i] = u[cs.num_constraints() + i];
    }
    /* process all other constraints */
    for (size_t i = 0; i < cs.num_constraints(); ++i)
    {
        auto a = cs.constraints[i]->getA().getTerms();
        for (size_t j = 0; j < a.size(); ++j)
        {
            At[a[j].index] += u[i]*a[j].getCoeff();
        }

        auto b = cs.constraints[i]->getB().getTerms();
        for (size_t j = 0; j < b.size(); ++j)
        {
            Bt[b[j].index] += u[i]*b[j].getCoeff();
        }

        auto c = cs.constraints[i]->getC().getTerms();
        for (size_t j = 0; j < c.size(); ++j)
        {
            Ct[c[j].index] +=u[i]*c[j].getCoeff();
        }
    }

    FieldT ti = FieldT::one();
    for (size_t i = 0; i < domain->m+1; ++i)
    {
        Ht.emplace_back(ti);
        ti *= t;
    }
    libff::leave_block("Compute evaluations of A, B, C, H at t");

    libff::leave_block("Call to r1cs_to_qap_instance_map_with_evaluation");

    return qap_instance_evaluation<FieldT>(domain,
                                           cs.num_variables(),
                                           domain->m,
                                           cs.num_inputs(),
                                           t,
                                           std::move(At),
                                           std::move(Bt),
                                           std::move(Ct),
                                           std::move(Ht),
                                           Zt);
}


template<typename FieldT>
static void butterfly_2(std::vector<FieldT>& out, const std::vector<FieldT>& twiddles, unsigned int stride, unsigned int stage_length, unsigned int out_offset)
{
    unsigned int out_offset2 = out_offset + stage_length;

    FieldT t = out[out_offset2];
    out[out_offset2] = out[out_offset] - t;
    out[out_offset] += t;
    out_offset2++;
    out_offset++;

    for (unsigned int k = 1; k < stage_length; k++)
    {
        FieldT t = twiddles[k] * out[out_offset2];
        out[out_offset2] = out[out_offset] - t;
        out[out_offset] += t;
        out_offset2++;
        out_offset++;
    }
}

template<typename FieldT>
static void butterfly_4(std::vector<FieldT>& out, const std::vector<FieldT>& twiddles, unsigned int stride, unsigned int stage_length, unsigned int out_offset, bool flag=false)
{
    const FieldT j = twiddles[twiddles.size() - 1];
    unsigned int tw = 0;

    /* Case twiddle == one */
    {
        const unsigned i0  = out_offset;
        const unsigned i1  = out_offset + stage_length;
        const unsigned i2  = out_offset + stage_length*2;
        const unsigned i3  = out_offset + stage_length*3;

        const FieldT z0  = out[i0];
        const FieldT z1  = out[i1];
        const FieldT z2  = out[i2];
        const FieldT z3  = out[i3];

        const FieldT t1  = z0 + z2;
        const FieldT t2  = z1 + z3;
        const FieldT t3  = z0 - z2;
        const FieldT t4j = j * (z1 - z3);

        out[i0] = t1 + t2;
        out[i1] = t3 - t4j;
        out[i2] = t1 - t2;
        out[i3] = t3 + t4j;

        out_offset++;
        tw += 3;
    }

    auto f=[](const FieldT& a){
        for(int i = 0; i < 4; i++){
            printf("%lu ", a.mont_repr.data[i]);
        }
        printf("\n");
    };
    if(flag){
        printf("modulus:\n");
        //f(j.get_modulus());
        for(int i = 0; i < 4; i++){
            printf("%lu ", j.get_modulus().data[i]);
        }
        printf("\n");
        printf("inv=%lu\n", j.inv);
        printf("j:\n");
        f(j);
        printf("\n");
    }

    for (unsigned int k = 1; k < stage_length; k++)
    {
        const unsigned i0  = out_offset;
        const unsigned i1  = out_offset + stage_length;
        const unsigned i2  = out_offset + stage_length*2;
        const unsigned i3  = out_offset + stage_length*3;

        const FieldT z0  = out[i0];
        const FieldT z1  = out[i1] * twiddles[tw];
        const FieldT z2  = out[i2] * twiddles[tw+1];
        const FieldT z3  = out[i3] * twiddles[tw+2];
        if(flag && k == 4){
            printf("cpu data:\n");
            f(out[i1]);
            f(twiddles[tw]);
            printf("\n");
        }

        const FieldT t1  = z0 + z2;
        const FieldT t2  = z1 + z3;
        const FieldT t3  = z0 - z2;
        const FieldT t4 = z1-z3;
        //const FieldT t4j = j * (z1 - z3);
        const FieldT t4j = j * t4;

        out[i0] = t1 + t2;
        out[i1] = t3 - t4j;
        out[i2] = t1 - t2;
        out[i3] = t3 + t4j;
        if(flag && k == 4){
            printf("cpu data:\n");
            f(z0);
            f(z1);
            f(z2);
            f(z3);
            f(t1);
            f(t2);
            f(t3);
            f(t4);
            f(t4j);
            f(out[i0]);
            f(out[i1]);
            f(out[i2]);
            f(out[i3]);
            //f(out[i3].get_modulus());
            printf("\n");
        }

        out_offset++;
        tw += 3;
    }
}

template<typename FieldT>
void copy_field_to(const FieldT& src, gpu::Fp_model& dst, const int offset){
    memcpy(dst.mont_repr_data + offset, src.mont_repr.data, 32);
};

template<typename FieldT>
struct GpuData{
    std::vector<std::vector<libfqfft::Info>> infos;
    gpu::Fp_model h_itwiddles, h_ftwiddles, d_itwiddles, d_ftwiddles;
    gpu::gpu_buffer d_modulus, h_max_values, d_max_values;
    gpu::Fp_model h_in, d_in, d_out;
    std::vector<int> itwiddle_offsets, ftwiddle_offsets;
    GpuData(const std::shared_ptr<libfqfft::evaluation_domain<FieldT>> domain){
        h_max_values.resize_host(1);
        d_max_values.resize(1);
        for(int i = 0; i < BITS/32; i++){
            h_max_values.ptr->_limbs[i] = 0xffffffff;
        }
        d_max_values.copy_from_host(h_max_values);
        d_modulus.resize(1);

        auto copy_twiddle = [&](const std::vector<std::vector<FieldT>>& twiddles, gpu::Fp_model& h_twiddles, gpu::Fp_model& d_twiddles, std::vector<int>& twiddle_offsets){
            int total_twiddle = 0;
            for(int level = 0; level < twiddles.size(); level++){
                total_twiddle += twiddles[level].size();
            }
            h_twiddles.resize_host(total_twiddle);
            d_twiddles.resize(total_twiddle);

            total_twiddle = 0;
            int twiddle_offset = 0;
            twiddle_offsets.resize(twiddles.size());
            for(int level = 0; level < twiddles.size(); level++){
                for(size_t i = 0; i < twiddles[level].size(); i++){
                    copy_field_to(twiddles[level][i], h_twiddles, total_twiddle++);
                }
                twiddle_offsets[level] = twiddle_offset;
                twiddle_offset += twiddles[level].size();
            }
            d_twiddles.copy_from_cpu(h_twiddles);
        };
        copy_twiddle(domain->data.iTwiddles, h_itwiddles, d_itwiddles, itwiddle_offsets);
        copy_twiddle(domain->data.fTwiddles, h_ftwiddles, d_ftwiddles, ftwiddle_offsets);
    }
    ~GpuData(){
        h_max_values.release_host();
        d_max_values.release();
        d_modulus.release();
        h_itwiddles.release_host();
        h_ftwiddles.release_host();
        d_itwiddles.release();
        d_ftwiddles.release();
        d_in.release();
        h_in.release_host();
        d_out.release();
    }
};

template<typename FieldT>
void CallIFFT(
        const std::shared_ptr<libfqfft::evaluation_domain<FieldT>> domain,
        std::vector<FieldT>& in, 
        GpuData<FieldT>& gpu_data,
        bool inverse, bool need_mul_scalar, bool need_get_infos, bool calc_coset=false){
    auto copy_field_back = [](FieldT& dst, const gpu::Fp_model& src, const int offset){
        memcpy(dst.mont_repr.data, src.mont_repr_data + offset, 32);
    };

    //double t0 = omp_get_wtime();
    //static std::vector<std::vector<libfqfft::Info>> infos;
    //if(need_get_infos){
    //    infos.clear();
    //    domain->fft_internal(in, infos);
    //}
    //double t1 = omp_get_wtime();
    //printf("fft_internal: %f\n", t1-t0);

    //static gpu::Fp_model h_itwiddles, d_itwiddles, h_ftwiddles, d_ftwiddles;
    //static gpu::gpu_buffer d_modulus, h_max_values, d_max_values;
    uint64_t const_inv = in[0].inv;
    gpu_data.h_in.resize_host(in.size());
    gpu_data.d_in.resize(in.size());
    gpu_data.d_out.resize(in.size());
#ifdef debug
    static gpu::Fp_model h_copy, h_out; 
    h_copy.resize_host(in.size());
    std::vector<FieldT> out(in.size());
    auto f=[](const FieldT& a){
        for(int i = 0; i < 4; i++){
            printf("%lu ", a.mont_repr.data[i]);
        }
        printf("\n");
    };
#endif
    //printf("modulus:");
    //f(in[0].get_modulus());
    //printf("\n");
    for(size_t i = 0; i < in.size(); i++){
        copy_field_to(in[i], gpu_data.h_in, i);
    }
    gpu_data.d_in.copy_from_cpu(gpu_data.h_in);

    auto& twiddles =  inverse ? domain->data.iTwiddles : domain->data.fTwiddles;
    //static std::vector<int> itwiddle_offsets(infos.size()), ftwiddle_offsets(infos.size());
    static gpu::gpu_meta d_in_offsets, d_out_offsets, d_lengths, d_radixs, d_strides;

    gpu::Fp_model& d_twiddles = inverse ? gpu_data.d_itwiddles : gpu_data.d_ftwiddles;
    std::vector<int>& twiddle_offsets = inverse ? gpu_data.itwiddle_offsets : gpu_data.ftwiddle_offsets;

    auto& infos = gpu_data.infos;
    int max_len = infos[0].size();
    for(int i = 1; i < infos.size(); i++){
        if(max_len < infos[i].size()){
            max_len = infos[i].size();
        }
    }

    std::vector<int> in_offsets(max_len), out_offsets(max_len), lengths(max_len), radixs(max_len), strides(max_len);
    if(true){
        d_in_offsets.resize(max_len*sizeof(int));
        d_out_offsets.resize(max_len*sizeof(int));
        d_lengths.resize(max_len*sizeof(int));
        d_radixs.resize(max_len*sizeof(int));
        d_strides.resize(max_len*sizeof(int));
    }

    //double t2 = omp_get_wtime();
    //printf("befor calc time:%f\n", t2-t1);
    for(int i = infos.size()-1; i >= 0; i--){
        int info_len = infos[i].size();
        int max_len = 0, min_len = infos[i][0].length;
        const uint32_t stage_length = domain->data.stages[i].length;
        const uint32_t radix = domain->data.stages[i].radix;
        if(true){
            for(int j = 0; j < info_len; j++){
                in_offsets[j] = infos[i][j].in_offset;
                out_offsets[j] = infos[i][j].out_offset;
                lengths[j] = infos[i][j].length;
                radixs[j] = infos[i][j].radix;
                strides[j] = infos[i][j].stride;
                if(max_len < lengths[j]) max_len = lengths[j];
                if(min_len > lengths[j]) min_len = lengths[j];
            }
            //printf("%d max_len = %d, min_len = %d, info_len = %d, radix=%d\n", i, max_len, min_len, info_len, radix);
            //double tt0 = omp_get_wtime();
            {
                gpu::copy_cpu_to_gpu(d_in_offsets.ptr, in_offsets.data(), info_len*sizeof(int));
                gpu::copy_cpu_to_gpu(d_out_offsets.ptr, out_offsets.data(), info_len*sizeof(int));
                gpu::copy_cpu_to_gpu(d_lengths.ptr, lengths.data(), info_len*sizeof(int));
                gpu::copy_cpu_to_gpu(d_radixs.ptr, radixs.data(), info_len*sizeof(int));
                gpu::copy_cpu_to_gpu(d_strides.ptr, strides.data(), info_len*sizeof(int));
            }
        }
        //double tt1 = omp_get_wtime();
        //printf("%d copy time = %f\n", i, tt1-tt0);
        if(stage_length == 1){
            gpu::fft_copy(gpu_data.d_in, gpu_data.d_out, (int*)d_in_offsets.ptr, (int*)d_out_offsets.ptr, (int*)d_strides.ptr, info_len, radix);     
            //d_out.copy_to_cpu(h_copy);
        }
        if(radix == 2){
            gpu::butterfly_2(gpu_data.d_out, d_twiddles, twiddle_offsets[i], (int*)d_strides.ptr, stage_length, (int*)d_out_offsets.ptr, info_len, gpu_data.d_max_values.ptr, gpu_data.d_modulus.ptr, const_inv); 
            //d_out.copy_to_cpu(h_in);
        }
        if(radix == 4){
            gpu::butterfly_4(gpu_data.d_out, d_twiddles, twiddles[i].size(), twiddle_offsets[i], (int*)d_strides.ptr, stage_length, (int*)d_out_offsets.ptr, info_len, gpu_data.d_max_values.ptr, gpu_data.d_modulus.ptr, const_inv); 
            //d_out.copy_to_cpu(h_in);
        }
        //double tt2 = omp_get_wtime();
        //printf("%d calc time = %f\n", i, tt2-tt1);
#ifdef debug
        if(false){
            for(int j = 0; j < infos[i].size(); j++){
                int in_offset = infos[i][j].in_offset;
                int out_offset = infos[i][j].out_offset;
                int level = i;//infos[i][j].level;
                int length = infos[i][j].length;
                int radix = infos[i][j].radix;
                int stride = infos[i][j].stride;
                if(length == 1){
                    for(int k = 0; k <  radix; k++){
                        out[out_offset + k] = in[in_offset + k * stride]; 
                        FieldT a;
                        copy_field_back(a, h_copy, out_offset + k);
                        if(a != out[out_offset + k]){
                            printf("compare copy failed %d %d %d\n", i, j, k);
                            return;
                        }
                    }
                }
                bool flag = false;
                switch (infos[i][j].radix)
                {
                    case 2: 
                        butterfly_2(out, twiddles[level], 0, length, out_offset); 
                        for(int k = 0; k < length*2; k++){
                            FieldT a;
                            copy_field_back(a, h_in, out_offset + k);
                            if(a != out[out_offset + k]){
                                printf("compare radix=2 failed %d %d %d\n", i, j, k);
                                return;
                            }
                        }
                        break;
                    case 4: 
                        if(i == 9 && j == 60448) flag = true;
                        butterfly_4(out, twiddles[level], 0, length, out_offset, flag); 
                        for(int k = 0; k < length*2; k++){
                            FieldT a;
                            copy_field_back(a, h_in, out_offset + k);
                            if(a != out[out_offset + k]){
                                printf("compare radix=4 failed %d %d %d\n", i, j, k);
                                f(a);
                                f(out[out_offset+k]);
                                return;
                            }
                        }
                        break;
                    default: std::cout << "error" << std::endl; assert(false);
                }
            }
        }
#endif
    }
    //double t3 = omp_get_wtime();

    static gpu::Fp_model h_sconst, d_sconst;
    size_t m = domain->m;
    if(need_mul_scalar){
#ifdef debug
        if(true){
            for(size_t i = 0; i < in.size(); i++){
                copy_field_to(out[i], gpu_data.h_in, i);
            }
            d_out.copy_from_cpu(gpu_data.h_in);
        }
#endif
        h_sconst.resize_host(1);
        d_sconst.resize(1);
        const FieldT sconst = FieldT(domain->m).inverse();
        copy_field_to(sconst, h_sconst, 0);
        d_sconst.copy_from_cpu(h_sconst);
        gpu::alt_bn128_g1_elementwise_mul_scalar(gpu_data.d_out, d_sconst, m, gpu_data.d_modulus.ptr, const_inv);  
    }

    if(calc_coset){
        static gpu::Fp_model h_one, d_one, h_g, d_g;
        d_one.resize(1);
        h_one.resize_host(1);
        d_g.resize(1);
        h_g.resize_host(1);

        copy_field_to(FieldT::one(), h_one, 0);
        d_one.copy_from_cpu(h_one);
        copy_field_to(FieldT::multiplicative_generator, h_g, 0);
        d_g.copy_from_cpu(h_g);
        //call cpu
        if(false){
            gpu_data.d_out.copy_to_cpu(gpu_data.h_in);
            for(size_t i = 0; i < in.size(); i++){
                copy_field_back(in[i], gpu_data.h_in, i);
            }
            domain->cosetFFT(in, FieldT::multiplicative_generator);
        }

        gpu::multiply_by_coset_and_constant(gpu_data.d_out, m, d_g, d_one, d_one, gpu_data.d_modulus.ptr, const_inv, 64);

        //verify 
        if(false){
            gpu_data.d_out.copy_to_cpu(gpu_data.h_in);
            for(int i = 0; i < m; i++){
                FieldT a;
                copy_field_back(a, gpu_data.h_in, i);
                if(in[i] != a){
                    printf("compare %d failed\n", i);
                    break;
                }
            }
        }
    }

    gpu_data.d_out.copy_to_cpu(gpu_data.h_in);
    for(size_t i = 0; i < in.size(); i++){
        copy_field_back(in[i], gpu_data.h_in, i);
    }
    //double t4 = omp_get_wtime();
    //printf("gpu time: %f %f\n", t3-t2, t4-t3);
}

/**
 * Witness map for the R1CS-to-QAP reduction.
 *
 * The witness map takes zero knowledge into account when d1,d2,d3 are random.
 *
 * More precisely, compute the coefficients
 *     h_0,h_1,...,h_n
 * of the polynomial
 *     H(z) := (A(z)*B(z)-C(z))/Z(z)
 * where
 *   A(z) := A_0(z) + \sum_{k=1}^{m} w_k A_k(z) + d1 * Z(z)
 *   B(z) := B_0(z) + \sum_{k=1}^{m} w_k B_k(z) + d2 * Z(z)
 *   C(z) := C_0(z) + \sum_{k=1}^{m} w_k C_k(z) + d3 * Z(z)
 *   Z(z) := "vanishing polynomial of set S"
 * and
 *   m = number of variables of the QAP
 *   n = degree of the QAP
 *
 * This is done as follows:
 *  (1) compute evaluations of A,B,C on S = {sigma_1,...,sigma_n}
 *  (2) compute coefficients of A,B,C
 *  (3) compute evaluations of A,B,C on T = "coset of S"
 *  (4) compute evaluation of H on T
 *  (5) compute coefficients of H
 *  (6) patch H to account for d1,d2,d3 (i.e., add coefficients of the polynomial (A d2 + B d1 - d3) + d1*d2*Z )
 *
 * The code below is not as simple as the above high-level description due to
 * some reshuffling to save space.
 */
template<typename FieldT>
void r1cs_to_qap_witness_map(const std::shared_ptr<libfqfft::evaluation_domain<FieldT>> domain,
                             const r1cs_constraint_system<FieldT> &cs,
                             const std::vector<FieldT> &full_variable_assignment,
                             std::vector<FieldT> &aA,
                             std::vector<FieldT> &aB,
                             std::vector<FieldT> &aH)
{
    libff::enter_block("Call to r1cs_to_qap_witness_map");

    libff::enter_block("Compute evaluation of polynomials A, B, C on set S");
    std::vector<FieldT> &aC = aH;
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < cs.num_constraints(); ++i)
    {
        aA[i] = cs.constraints[i]->evaluateA(full_variable_assignment);
        aB[i] = cs.constraints[i]->evaluateB(full_variable_assignment);
        aC[i] = cs.constraints[i]->evaluateC(full_variable_assignment);
    }
    /* account for the additional constraints input_i * 0 = 0 */
    for (size_t i = 0; i <= cs.num_inputs(); ++i)
    {
        aA[i+cs.num_constraints()] = full_variable_assignment[i];
        aB[i+cs.num_constraints()] = FieldT::zero();
        aC[i+cs.num_constraints()] = FieldT::zero();
    }
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    /* zero initialize the remaining coefficients */
    for (size_t i = cs.num_constraints() + cs.num_inputs() + 1; i < domain->m; i++)
    {
        aA[i] = FieldT::zero();
        aB[i] = FieldT::zero();
        aC[i] = FieldT::zero();
    }
    libff::leave_block("Compute evaluation of polynomials A, B, C on set S");

    libff::enter_block("Compute the recursive Infos");
    GpuData<FieldT> gpu_data(domain);
    //gpu_data.infos.clear();
    domain->fft_internal(aA, gpu_data.infos);
    gpu::copy_cpu_to_gpu(gpu_data.d_modulus.ptr->_limbs, aA[0].get_modulus().data, 32);
    libff::leave_block("Compute the recursive Infos");

    libff::enter_block("Compute coefficients of polynomial A");
    if(true){
        CallIFFT<FieldT>(domain, aA, gpu_data, true, true, true);
    }else{
        domain->iFFT(aA);
    }
    libff::leave_block("Compute coefficients of polynomial A");

    libff::enter_block("Compute evaluation of polynomial A on set T");
    domain->cosetFFT(aA, FieldT::multiplicative_generator);
    CallIFFT<FieldT>(domain, aA, gpu_data, false, false, false);
    libff::leave_block("Compute evaluation of polynomial A on set T");

    libff::enter_block("Compute coefficients of polynomial B");
    if(true){
        CallIFFT<FieldT>(domain, aB, gpu_data, true, true, false);
    }else{
        domain->iFFT(aB);
    }
    libff::leave_block("Compute coefficients of polynomial B");

    libff::enter_block("Compute evaluation of polynomial B on set T coset");
    domain->cosetFFT(aB, FieldT::multiplicative_generator);
    libff::leave_block("Compute evaluation of polynomial B on set T coset");
    libff::enter_block("Compute evaluation of polynomial B on set T ifft");
    CallIFFT<FieldT>(domain, aB, gpu_data, false, false, false);
    libff::leave_block("Compute evaluation of polynomial B on set T ifft");

    libff::enter_block("Compute coefficients of polynomial C");
    if(true){
        CallIFFT<FieldT>(domain, aC, gpu_data, true, true, false);
    }else{
        domain->iFFT(aC);
    }
    libff::leave_block("Compute coefficients of polynomial C");

    libff::enter_block("Compute evaluation of polynomial C on set T");
    domain->cosetFFT(aC, FieldT::multiplicative_generator);
    CallIFFT<FieldT>(domain, aC, gpu_data, false, false, false);
    libff::leave_block("Compute evaluation of polynomial C on set T");

    libff::enter_block("Compute evaluation of polynomial H on set T");
#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        aH[i] = (aA[i] * aB[i]) - aC[i];
    }
    aH[domain->m] = FieldT::zero();

    libff::enter_block("Divide by Z on set T");
    domain->divide_by_Z_on_coset(aH);
    libff::leave_block("Divide by Z on set T");

    libff::leave_block("Compute evaluation of polynomial H on set T");

    libff::enter_block("Compute coefficients of polynomial H");
    CallIFFT<FieldT>(domain, aH, gpu_data, true, false, false);
    domain->icosetFFT(aH, FieldT::multiplicative_generator);
    libff::leave_block("Compute coefficients of polynomial H");

    libff::leave_block("Call to r1cs_to_qap_witness_map");
}

} // libsnark

#endif // R1CS_TO_QAP_TCC_
