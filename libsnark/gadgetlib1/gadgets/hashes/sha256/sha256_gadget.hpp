/** @file
 *****************************************************************************

 Declaration of interfaces for top-level SHA256 gadgets.

 *****************************************************************************
 * @author     This file is part of libsnark, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef SHA256_GADGET_HPP_
#define SHA256_GADGET_HPP_

#include <libsnark/common/data_structures/merkle_tree.hpp>
#include <libsnark/gadgetlib1/gadgets/basic_gadgets.hpp>
#include <libsnark/gadgetlib1/gadgets/hashes/hash_io.hpp>
#include <libsnark/gadgetlib1/gadgets/hashes/sha256/sha256_components.hpp>

namespace libsnark {

/**
 * Gadget for the SHA256 compression function.
 */
template<typename FieldT>
class sha256_compression_function_gadget : public gadget<FieldT> {
private:
    pb_variable_array<FieldT> packed_W;
    std::shared_ptr<sha256_message_schedule_gadget<FieldT> > message_schedule;
    std::vector<sha256_round_function_gadget<FieldT> > round_functions;

    pb_variable_array<FieldT> unreduced_output;
    pb_variable_array<FieldT> reduced_output;
    std::vector<lastbits_gadget<FieldT> > reduce_output;

    pb_linear_combination_array<FieldT> prev_output;
    digest_variable<FieldT> output;

public:
    sha256_compression_function_gadget(protoboard<FieldT> &pb,
                                       const pb_linear_combination_array<FieldT> &prev_output,
                                       const pb_variable_array<FieldT> &new_block,
                                       const digest_variable<FieldT> &output,
                                       const std::string &annotation_prefix);
    void generate_r1cs_constraints();
    void generate_r1cs_witness();
};

/**
 * Gadget for the SHA256 compression function, viewed as a 2-to-1 hash
 * function, and using the same initialization vector as in SHA256
 * specification. Thus, any collision for
 * sha256_two_to_one_hash_gadget trivially extends to a collision for
 * full SHA256 (by appending the same padding).
 */
template<typename FieldT>
class sha256_two_to_one_hash_gadget : public gadget<FieldT> {
public:
    typedef libff::bit_vector hash_value_type;
    typedef merkle_authentication_path merkle_authentication_path_type;

    std::shared_ptr<sha256_compression_function_gadget<FieldT> > f;

    sha256_two_to_one_hash_gadget(protoboard<FieldT> &pb,
                                  const digest_variable<FieldT> &left,
                                  const digest_variable<FieldT> &right,
                                  const digest_variable<FieldT> &output,
                                  const std::string &annotation_prefix);
    sha256_two_to_one_hash_gadget(protoboard<FieldT> &pb,
                                  const size_t block_length,
                                  const block_variable<FieldT> &input_block,
                                  const digest_variable<FieldT> &output,
                                  const std::string &annotation_prefix);

    void generate_r1cs_constraints(const bool ensure_output_bitness=true); // TODO: ignored for now
    void generate_r1cs_witness();

    static size_t get_block_len();
    static size_t get_digest_len();
    static libff::bit_vector get_hash(const libff::bit_vector &input);

    static size_t expected_constraints(const bool ensure_output_bitness=true); // TODO: ignored for now
};

} // libsnark

#include <libsnark/gadgetlib1/gadgets/hashes/sha256/sha256_gadget.tcc>

#endif // SHA256_GADGET_HPP_
