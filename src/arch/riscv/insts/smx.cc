/*
 * Copyright (c) 2023 Max Xing
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "arch/riscv/insts/smx.hh"

#include <sstream>
#include <string>

#include "arch/riscv/regs/int.hh"
#include "arch/riscv/utility.hh"
#include "base/compiler.hh"
#include "cpu/static_inst.hh"

namespace gem5
{

namespace RiscvISA
{

void
SmxOp::setIndvarSrcs()
{
    for (const auto &iv : IndvarRegs) {
        setSrcRegIdx(_numSrcRegs++, iv);
    }
}

void
SmxOp::setIndvarDests()
{
    for (const auto &iv : IndvarRegs) {
        setDestRegIdx(_numDestRegs++, iv);
    }
}

std::string
SmxOp::generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const
{
    return mnemonic;
}

std::string
SmxImmOp::generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const
{
    std::stringstream ss;
    ss << mnemonic << ' ' << registerName(destRegIdx(0)) << ", " <<
        stream << ", " << imm;
    return ss.str();
}

std::string
SmxLoadOp::generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const
{
    std::stringstream ss;
    ss << mnemonic << ' ' << registerName(destRegIdx(0)) << ", " <<
        stream << ", " << sel;
    return ss.str();
}

std::string
SmxStoreOp::generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const
{
    std::stringstream ss;
    ss << mnemonic << ' ' << stream << ", " <<
        registerName(srcRegIdx(0)) << ", " << sel;
    return ss.str();
}

std::unique_ptr<PCStateBase>
SmxBranchOp::branchTarget(const PCStateBase &branch_pc) const
{
    auto &rpc = branch_pc.as<RiscvISA::PCState>();
    return std::make_unique<PCState>(rpc.pc() + imm);
}

} // namespace RiscvISA
} // namespace gem5
