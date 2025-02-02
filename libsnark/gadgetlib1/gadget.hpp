/** @file
 *****************************************************************************
 * @author     This file is part of libsnark, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef GADGET_HPP_
#define GADGET_HPP_

#include <libsnark/gadgetlib1/protoboard.hpp>

namespace libsnark {

template<typename FieldT>
class gadget {
public:
    protoboard<FieldT> &pb;
    const std::string annotation_prefix;
    gadget(protoboard<FieldT> &pb, const std::string &annotation_prefix="");
};

} // libsnark
#include <libsnark/gadgetlib1/gadget.tcc>

#endif // GADGET_HPP_
