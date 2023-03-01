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

#ifndef __ARCH_RISCV_INSTS_SMX_HH__
#define __ARCH_RISCV_INSTS_SMX_HH__

#include <string>

#include "arch/riscv/insts/bitfields.hh"
#include "arch/riscv/insts/static_inst.hh"
#include "cpu/static_inst.hh"

namespace gem5
{

namespace RiscvISA
{

/**
 * Base class for all SMX operations
 */
class SmxOp : public RiscvStaticInst
{
  protected:
    SmxOp(const char *mnem, MachInst _machInst, OpClass __opClass)
        : RiscvStaticInst(mnem, _machInst, __opClass)
    {}

    void setIndvarSrcs();
    void setIndvarDests();

    std::string generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const override;
};

/**
 * Base class for SMX operations with immediates
 */
class SmxImmOp : public SmxOp
{
  protected:
    uint64_t stream;
    int64_t imm;

    SmxImmOp(const char *mnem, MachInst _machInst, OpClass __opClass)
        : SmxOp(mnem, _machInst, __opClass),
            stream(RS1), imm(sext<12>(FUNCT12))
    {}

    std::string generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const override;
};

/**
 * Base class for SMX memory operations
 */
class SmxMemOp : public SmxOp
{
  protected:
    Request::Flags memAccessFlags;
    uint64_t stream;
    uint64_t width;
    uint64_t sel;

    SmxMemOp(const char *mnem, MachInst _machInst, OpClass __opClass,
            uint64_t _width)
        : SmxOp(mnem, _machInst, __opClass),
            stream(RS1), width(_width), sel(MMSEL)
    {}
};

/**
 * Base class for SMX load operations
 */
class SmxLoadOp : public SmxMemOp
{
  protected:
    bool is_unsigned;

    SmxLoadOp(const char *mnem, MachInst machInst, OpClass __opClass)
        : SmxMemOp(mnem, machInst, __opClass, LDWIDTH),
            is_unsigned(!!LDUNSIGNED)
    {}

    std::string generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const override;
};

/**
 * Base class for SMX store operations
 */
class SmxStoreOp : public SmxMemOp
{
  protected:
    SmxStoreOp(const char *mnem, MachInst machInst, OpClass __opClass)
        : SmxMemOp(mnem, machInst, __opClass, STWIDTH)
    {}

    std::string generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const override;
};

/**
 * Base class for SMX branch operations
 */
class SmxBranchOp : public SmxImmOp
{
  protected:
    SmxBranchOp(const char *mnem, MachInst _machInst, OpClass __opClass)
        : SmxImmOp(mnem, _machInst, __opClass)
    {
        imm <<= 1;
    }

    std::unique_ptr<PCStateBase> branchTarget(
            const PCStateBase &branch_pc) const override;
    
    using StaticInst::branchTarget;
};

} // namespace RiscvISA
} // namespace gem5

#endif // __ARCH_RISCV_INSTS_SMX_HH__
