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


//for debug
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

//for debug
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

#ifdef USE_GPU
template<typename FieldT>
void copy_field_to(const FieldT& src, gpu::Buffer<>& dst, const int offset){
    memcpy(dst.ptr_ + offset * gpu::N, src.mont_repr.data, 32);
};

template<typename FieldT>
struct GpuData{
    std::vector<std::vector<libfqfft::Info>> infos;
    gpu::Buffer<gpu::Int, gpu::N> h_itwiddles, h_ftwiddles, d_itwiddles, d_ftwiddles;
    gpu::Buffer<gpu::Int, gpu::N> d_modulus;
    gpu::Buffer<gpu::Int, gpu::N> h_in, d_A, d_B, d_C, d_in, d_out;
    gpu::Buffer<gpu::Int, gpu::N> xor_results, d_one, d_g, d_c, d_sconst;
    std::vector<int> itwiddle_offsets, ftwiddle_offsets;
    std::vector<int> in_offsets, out_offsets, lengths, radixs, strides;
    std::vector<int> info_offsets;
    gpu::Buffer<int, 1> d_in_offsets, d_out_offsets, d_strides;
    bool calc_xor = true;
    uint64_t const_inv; 
    gpu::GPUContext *gpu_ctx_;
    gpu::CPUContext *cpu_ctx_ = new gpu::CPUContext();
    gpu::GPUStream stream_;

    GpuData(const std::shared_ptr<libfqfft::evaluation_domain<FieldT>> domain, gpu::GPUContext* gpu_ctx){
        gpu_ctx_ = gpu_ctx;
        stream_.create();
        d_modulus.resize(gpu_ctx, 1);
        xor_results.resize(gpu_ctx, domain->m+1);
        d_one.resize(gpu_ctx, 1);
        d_g.resize(gpu_ctx, 1);
        d_c.resize(gpu_ctx, 1);
        d_sconst.resize(gpu_ctx, 1);

        auto copy_twiddle = [&](const std::vector<std::vector<FieldT>>& twiddles, gpu::Buffer<>& h_twiddles, gpu::Buffer<>& d_twiddles, std::vector<int>& twiddle_offsets){
            int total_twiddle = 0;
            for(int level = 0; level < twiddles.size(); level++){
                total_twiddle += twiddles[level].size();
            }
            h_twiddles.resize(cpu_ctx_, total_twiddle);
            d_twiddles.resize(gpu_ctx_, total_twiddle);

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
            d_twiddles.copy(h_twiddles, &stream_);
        };
        copy_twiddle(domain->data.iTwiddles, h_itwiddles, d_itwiddles, itwiddle_offsets);
        copy_twiddle(domain->data.fTwiddles, h_ftwiddles, d_ftwiddles, ftwiddle_offsets);
        calc_xor = true;
    }
    ~GpuData(){
        h_itwiddles.release();
        h_ftwiddles.release();
        d_itwiddles.release();
        d_ftwiddles.release();
        d_modulus.release();
        h_in.release();
        d_A.release();
        d_B.release();
        d_C.release();
        d_in.release();
        d_out.release();
        xor_results.release();
        d_one.release();
        d_g.release();
        d_c.release();
        d_sconst.release();
        d_in_offsets.release();
        d_out_offsets.release();
        d_strides.release();
    }

    void InitOffsets(){
      info_offsets.resize(infos.size());
      int total_len = 0;
      for(int i = infos.size()-1; i >= 0; i--){
        info_offsets[i] = total_len;
        total_len += infos[i].size();
      }
      in_offsets.resize(total_len);
      out_offsets.resize(total_len);
      strides.resize(total_len);
      d_in_offsets.resize(gpu_ctx_, total_len);
      d_out_offsets.resize(gpu_ctx_, total_len);
      d_strides.resize(gpu_ctx_, total_len);

//#pragma omp parallel for
      for(int i = infos.size()-1; i >= 0; i--){
        int info_len = infos[i].size();
        const int info_offset = info_offsets[i];
        for(int j = 0; j < info_len; j++){
          in_offsets[info_offset + j] = infos[i][j].in_offset;
          out_offsets[info_offset + j] = infos[i][j].out_offset;
          strides[info_offset + j] = infos[i][j].stride;
        }
      }
      gpu::copy_cpu_to_gpu(d_in_offsets.ptr_, in_offsets.data(), total_len*sizeof(int), stream_.stream_);
      gpu::copy_cpu_to_gpu(d_out_offsets.ptr_, out_offsets.data(), total_len*sizeof(int), stream_.stream_);
      gpu::copy_cpu_to_gpu(d_strides.ptr_, strides.data(), total_len*sizeof(int), stream_.stream_);
    }
};

template<typename FieldT>
void GpuMultiplyByCosetAndConstant(GpuData<FieldT>& gpu_data, const int n, gpu::Buffer<gpu::Int, gpu::N>& d_out){
    auto& stream = gpu_data.stream_.stream_;
    if(gpu_data.calc_xor){
        gpu_data.calc_xor = false;
        gpu::calc_xor(gpu_data.xor_results, n, 1, gpu_data.d_g, gpu_data.d_one, gpu_data.d_modulus, gpu_data.const_inv, 64, stream);
    }
    gpu::multiply(d_out, gpu_data.xor_results, n, 1, gpu_data.d_c, gpu_data.d_modulus, gpu_data.const_inv, stream);
}

template<typename FieldT>
inline void copy_field_back(FieldT& dst, const gpu::Buffer<>& src, const int offset){
    memcpy(dst.mont_repr.data, src.ptr_ + offset * gpu::N, 32);
};

template<typename FieldT>
void CallIFFT(
        const std::shared_ptr<libfqfft::evaluation_domain<FieldT>> domain,
        std::vector<FieldT>& in, 
        GpuData<FieldT>& gpu_data,
        gpu::Buffer<>& h_in,
        gpu::Buffer<>& d_in,
        gpu::Buffer<>& d_out,
        bool copy_out,
        bool inverse, bool need_mul_scalar, bool need_get_infos, bool calc_coset=false, bool copy_in=true){

    uint64_t const_inv = in[0].inv;
    h_in.resize(gpu_data.cpu_ctx_, in.size());
    d_out.resize(gpu_data.gpu_ctx_, in.size());
    if(copy_in){
        d_in.resize(gpu_data.gpu_ctx_, in.size());
#pragma omp parallel
        for(size_t i = 0; i < in.size(); i++){
            copy_field_to(in[i], h_in, i);
        }
        d_in.copy(h_in, &gpu_data.stream_);
    }

    auto& twiddles =  inverse ? domain->data.iTwiddles : domain->data.fTwiddles;

    gpu::Buffer<>& d_twiddles = inverse ? gpu_data.d_itwiddles : gpu_data.d_ftwiddles;
    std::vector<int>& twiddle_offsets = inverse ? gpu_data.itwiddle_offsets : gpu_data.ftwiddle_offsets;

    auto& infos = gpu_data.infos;
    auto& stream = gpu_data.stream_.stream_;

    for(int i = infos.size()-1; i >= 0; i--){
        int info_len = infos[i].size();
        int max_len = 0, min_len = infos[i][0].length;
        const uint32_t stage_length = domain->data.stages[i].length;
        const uint32_t radix = domain->data.stages[i].radix;
        const int info_offset = gpu_data.info_offsets[i];
        if(stage_length == 1){
            gpu::fft_copy(d_in, d_out, (int*)gpu_data.d_in_offsets.ptr_ + info_offset, (int*)gpu_data.d_out_offsets.ptr_ + info_offset, (int*)gpu_data.d_strides.ptr_ + info_offset, info_len, radix, stream);     
        }
        if(radix == 2){
            gpu::butterfly_2(d_out, d_twiddles, twiddle_offsets[i], (int*)gpu_data.d_strides.ptr_ + info_offset, stage_length, (int*)gpu_data.d_out_offsets.ptr_ + info_offset, info_len, gpu_data.d_modulus, const_inv, stream); 
        }
        if(radix == 4){
            gpu::butterfly_4(d_out, d_twiddles, twiddles[i].size(), twiddle_offsets[i], (int*)gpu_data.d_strides.ptr_ + info_offset, stage_length, (int*)gpu_data.d_out_offsets.ptr_ + info_offset, info_len, gpu_data.d_modulus, const_inv, stream); 
        }
    }

    size_t m = domain->m;
    if(need_mul_scalar){
        gpu::alt_bn128_g1_elementwise_mul_scalar(d_out, gpu_data.d_sconst, m, gpu_data.d_modulus, const_inv, stream);  
    }

    if(copy_out){
        //d_out.copy_to_cpu(h_in);
        h_in.copy(d_out, &gpu_data.stream_);
#pragma omp parallel
        for(size_t i = 0; i < in.size(); i++){
            copy_field_back(in[i], h_in, i);
        }
    }
}
#endif  //end USE_GPU

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
 #ifdef USE_GPU
template<typename FieldT>
void r1cs_to_qap_witness_map(const std::shared_ptr<libfqfft::evaluation_domain<FieldT>> domain,
                             const r1cs_constraint_system<FieldT> &cs,
                             const std::vector<FieldT> &full_variable_assignment,
                             std::vector<FieldT> &aA,
                             std::vector<FieldT> &aB,
                             std::vector<FieldT> &aH,
                             gpu::Buffer<gpu::Int, gpu::N>& d_H)
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
    static gpu::GPUContext gpu_ctx;
    GpuData<FieldT> gpu_data(domain, &gpu_ctx);
    auto& stream = gpu_data.stream_.stream_;
    //gpu_data.infos.clear();
    domain->fft_internal(aA, gpu_data.infos);
    gpu_data.InitOffsets();
    gpu::copy_cpu_to_gpu(gpu_data.d_modulus.ptr_, aA[0].get_modulus().data, 32);
    gpu::copy_cpu_to_gpu(gpu_data.d_one.ptr_, FieldT::one().mont_repr.data, 32);
    gpu::copy_cpu_to_gpu(gpu_data.d_g.ptr_, FieldT::multiplicative_generator.mont_repr.data, 32);
    gpu::copy_cpu_to_gpu(gpu_data.d_c.ptr_, FieldT::one().mont_repr.data, 32);
    gpu_data.const_inv = aA[0].inv;
    const FieldT sconst = FieldT(domain->m).inverse();
    gpu::copy_cpu_to_gpu(gpu_data.d_sconst.ptr_, sconst.mont_repr.data, 32);
    libff::leave_block("Compute the recursive Infos");

    libff::enter_block("Compute coefficients of polynomial A");
    CallIFFT<FieldT>(domain, aA, gpu_data, gpu_data.h_in, gpu_data.d_in, gpu_data.d_out, false, true, true, true, true);
    GpuMultiplyByCosetAndConstant<FieldT>(gpu_data, domain->m, gpu_data.d_out);
    libff::leave_block("Compute coefficients of polynomial A");

    libff::enter_block("Compute evaluation of polynomial A on set T");
    //domain->cosetFFT(aA, FieldT::multiplicative_generator);
    CallIFFT<FieldT>(domain, aA, gpu_data, gpu_data.h_in, gpu_data.d_out, gpu_data.d_A, false, false, false, false, false, false);
    libff::leave_block("Compute evaluation of polynomial A on set T");

    libff::enter_block("Compute coefficients of polynomial B");
    CallIFFT<FieldT>(domain, aB, gpu_data, gpu_data.h_in, gpu_data.d_in, gpu_data.d_out, false, true, true, false, true);
    GpuMultiplyByCosetAndConstant<FieldT>(gpu_data, domain->m, gpu_data.d_out);
    libff::leave_block("Compute coefficients of polynomial B");

    libff::enter_block("Compute evaluation of polynomial B on set T coset");
    //domain->cosetFFT(aB, FieldT::multiplicative_generator);
    libff::leave_block("Compute evaluation of polynomial B on set T coset");
    libff::enter_block("Compute evaluation of polynomial B on set T ifft");
    CallIFFT<FieldT>(domain, aB, gpu_data, gpu_data.h_in, gpu_data.d_out, gpu_data.d_B, false, false, false, false, false, false);
    libff::leave_block("Compute evaluation of polynomial B on set T ifft");

    libff::enter_block("Compute coefficients of polynomial C");
    CallIFFT<FieldT>(domain, aC, gpu_data, gpu_data.h_in, gpu_data.d_in, gpu_data.d_out, false, true, true, false, true);
    GpuMultiplyByCosetAndConstant<FieldT>(gpu_data, domain->m, gpu_data.d_out);
    libff::leave_block("Compute coefficients of polynomial C");

    libff::enter_block("Compute evaluation of polynomial C on set T");
    //domain->cosetFFT(aC, FieldT::multiplicative_generator);
    CallIFFT<FieldT>(domain, aC, gpu_data, gpu_data.h_in, gpu_data.d_out, gpu_data.d_C, false, false, false, false, false, false);
    libff::leave_block("Compute evaluation of polynomial C on set T");
    //gpu::Fp_model& d_H = d_H2; //gpu_data.d_H;

    libff::enter_block("Compute evaluation of polynomial H on set T");
    {
        const FieldT coset = FieldT::multiplicative_generator;
        const FieldT Z_inverse_at_coset = domain->compute_vanishing_polynomial(coset).inverse();
        gpu::Buffer<gpu::Int, gpu::N> d_coset;
        d_coset.resize(&gpu_ctx, 1);
        //gpu_data.d_H.resize(domain->m+1);
        d_H.resize(&gpu_ctx, domain->m+1);
        gpu::copy_cpu_to_gpu(d_coset.ptr_, Z_inverse_at_coset.mont_repr.data, 32, stream);
        gpu::calc_H(gpu_data.d_A, gpu_data.d_B, gpu_data.d_C, d_H, d_coset, domain->m, gpu_data.d_modulus, aA[0].inv, stream); 
        gpu::copy_cpu_to_gpu(d_H.ptr_ + domain->m * gpu::N, FieldT::zero().mont_repr.data, 32, stream); 
    }

    libff::leave_block("Compute evaluation of polynomial H on set T");

    libff::enter_block("Compute coefficients of polynomial H");
    CallIFFT<FieldT>(domain, aH, gpu_data, gpu_data.h_in, d_H, gpu_data.d_out, false, true, false, false, false, false);
    //calc xor is slow, so call host icosetFFT 
    //domain->icosetFFT(aH, FieldT::multiplicative_generator);
    if(true){
        const FieldT sconst = FieldT(domain->m).inverse();
        const FieldT g = FieldT::multiplicative_generator.inverse();
        gpu::copy_cpu_to_gpu(gpu_data.d_g.ptr_, g.mont_repr.data, 32, stream);
        gpu::copy_cpu_to_gpu(gpu_data.d_c.ptr_, sconst.mont_repr.data, 32, stream);
        //need calc xor because g is change
        gpu_data.calc_xor = true;
        GpuMultiplyByCosetAndConstant<FieldT>(gpu_data, domain->m, gpu_data.d_out);
        d_H.copy(gpu_data.d_out, &gpu_data.stream_);
        gpu_ctx.wait(&gpu_data.stream_);
    }
    libff::leave_block("Compute coefficients of polynomial H");

    libff::leave_block("Call to r1cs_to_qap_witness_map");
}

#else
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

    libff::enter_block("Compute coefficients of polynomial A");
    domain->iFFT(aA);
    libff::leave_block("Compute coefficients of polynomial A");

    libff::enter_block("Compute evaluation of polynomial A on set T");
    domain->cosetFFT(aA, FieldT::multiplicative_generator);
    libff::leave_block("Compute evaluation of polynomial A on set T");

    libff::enter_block("Compute coefficients of polynomial B");
    domain->iFFT(aB);
    libff::leave_block("Compute coefficients of polynomial B");

    libff::enter_block("Compute evaluation of polynomial B on set T");
    domain->cosetFFT(aB, FieldT::multiplicative_generator);
    libff::leave_block("Compute evaluation of polynomial B on set T");

    libff::enter_block("Compute coefficients of polynomial C");
    domain->iFFT(aC);
    libff::leave_block("Compute coefficients of polynomial C");

    libff::enter_block("Compute evaluation of polynomial C on set T");
    domain->cosetFFT(aC, FieldT::multiplicative_generator);
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
    domain->icosetFFT(aH, FieldT::multiplicative_generator);
    libff::leave_block("Compute coefficients of polynomial H");

    libff::leave_block("Call to r1cs_to_qap_witness_map");
}
#endif

} // libsnark

#endif // R1CS_TO_QAP_TCC_
