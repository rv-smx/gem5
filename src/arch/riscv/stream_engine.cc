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

#include "arch/riscv/stream_engine.hh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <sstream>

#include "arch/riscv/isa.hh"
#include "arch/riscv/regs/int.hh"
#include "base/bitfield.hh"
#include "base/compiler.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "cpu/exec_context.hh"
#include "cpu/thread_context.hh"
#include "cpu/o3/dyn_inst.hh"
#include "debug/StreamEngine.hh"
#include "sim/probe/probe.hh"

namespace
{

using namespace gem5;

constexpr unsigned MAX_INDVAR_NUM = RiscvISA::IndvarRegNum;
constexpr unsigned MAX_MEMORY_NUM = 32;
constexpr unsigned MAX_ADDR_NUM = 4;

constexpr unsigned MAX_PC_MEM_ID_PAIRS = 32;
constexpr unsigned NUM_RUNAHEAD_STEPS = 32;

/**
 * Listener for O3 CPU commit events.
 */
class CommitProbeListener : public ProbeListenerArgBase<o3::DynInstPtr>
{
  public:
    using ProbeListenerArgBase::ProbeListenerArgBase;

    void
    notify(const o3::DynInstPtr &inst) override
    {
        auto isa = inst->tcBase()->getIsaPtr();
        auto &se = static_cast<RiscvISA::ISA *>(isa)->streamEngine();
        auto op = dynamic_cast<RiscvISA::SmxOp *>(inst->staticInst.get());
        if (!op) return;
        auto name = op->getName();

        // Configurations.
        if (name == "smx_cfg_iv")
            return se.commitIndvarConfig(inst.get(), op);
        if (name == "smx_cfg_ms")
            return se.commitMemoryConfig(inst.get(), op);
        if (name == "smx_cfg_addr")
            return se.commitAddrConfig(inst.get(), op);

        // Hints.
        if (name == "smx_ready") return se.commitReady(inst.get());
        if (name == "smx_end") return se.commitEnd();

        // Step instructions.
        if (name.rfind("smx_step", 0) == 0)
            return se.commitStep(inst.get(), op);
    }
};

RegVal
applyWidthUnsigned(RegVal val, unsigned width, bool is_unsigned)
{
    switch (width) {
      case 0b00:
        return is_unsigned ? bits(val, 7, 0) : szext<8>(val);
      case 0b01:
        return is_unsigned ? bits(val, 15, 0) : szext<16>(val);
      case 0b10:
        return is_unsigned ? bits(val, 31, 0) : szext<32>(val);
      case 0b11:
        return val;
      default:
        GEM5_UNREACHABLE;
    }
}

std::string
indvarsToString(const std::vector<RegVal> &indvars)
{
    std::ostringstream oss;
    for (unsigned i = 0; i < indvars.size(); ++i) {
        if (i) oss << ", ";
        oss << indvars[i];
    }
    return oss.str();
}

} // namespace

namespace gem5
{

namespace RiscvISA
{

void
StreamEngine::addAddrConfigForLastMem(RegVal stride, unsigned dep,
        SmxStreamKind kind)
{
    auto &addrs = mems.back().addrs;
    unsigned memory_id = mems.size() - 1;

    addrs.push_back({stride, dep, kind});
    const char *kind_str;
    if (kind == SMX_KIND_IV) {
        kind_str = "induction variable";
    } else {
        kind_str = "memory";
    }
    DPRINTF(StreamEngine,
        "Added address factor stride=%llu, dependents %s"
        " stream %u for memory stream %u\n",
        stride, kind_str, dep, memory_id);
}

void
StreamEngine::addIndvarConfig(RegVal _init_val, RegVal _step_val,
        RegVal _final_val, SmxStopCond cond, unsigned width, bool is_unsigned)
{
    auto init_val = applyWidthUnsigned(_init_val, width, is_unsigned);
    auto step_val = applyWidthUnsigned(_step_val, width, is_unsigned);
    auto final_val = applyWidthUnsigned(_final_val, width, is_unsigned);
    DPRINTF(StreamEngine,
        "Induction variable stream %u: init=%llu, step=%llu, final=%llu\n",
        (unsigned)ivs.size(), init_val, step_val, final_val);
    ivs.push_back(
        {init_val, step_val, final_val, cond, width, is_unsigned});
    ++committedConfigs;
}

void
StreamEngine::addMemoryConfig(RegVal base, RegVal stride1, unsigned dep1,
        SmxStreamKind kind1, bool prefetch, unsigned width)
{
    auto mem_id = static_cast<unsigned>(mems.size());
    DPRINTF(StreamEngine, "Memory stream %u: base=0x%llx\n", mem_id, base);
    mems.push_back({base, prefetch, width, {}});
    addAddrConfigForLastMem(stride1, dep1, kind1);
    ++committedConfigs;
}

void
StreamEngine::addAddrConfig(
        RegVal stride1, unsigned dep1, SmxStreamKind kind1,
        RegVal stride2, unsigned dep2, SmxStreamKind kind2)
{
    addAddrConfigForLastMem(stride1, dep1, kind1);
    if (stride2) addAddrConfigForLastMem(stride2, dep2, kind2);
    ++committedConfigs;
}

Addr
StreamEngine::getMemoryAddrWithIndvars(unsigned memory_id,
        IndvarQuerier iq) const
{
    const auto &mem = mems[memory_id];
    Addr vaddr = mem.base;
    for (const auto &addr : mem.addrs) {
        assert(addr.kind == SMX_KIND_IV);
        vaddr += iq(addr.dep) * addr.stride;
    }
    return vaddr;
}

bool
StreamEngine::stepIndvars(std::vector<RegVal> &indvars,
        unsigned indvar_id, IndvarQuerier iq) const
{
    bool should_step = true;
    for (unsigned id = ivs.size() - 1; id <= ivs.size() - 1; --id) {
        auto &iv = ivs[id];
        if (id > indvar_id) {
            indvars[id] = iv.initVal;
        } else {
            auto value = iq(id);
            if (should_step) {
                value = applyWidthUnsigned(value + iv.stepVal, iv.width,
                    iv.isUnsigned);
                if (isNotInLoop(indvar_id, value)) {
                    value = iv.initVal;
                } else {
                    should_step = false;
                }
            }
            indvars[id] = value;
        }
    }
    return should_step;
}

void
StreamEngine::removeCommitListener()
{
    if (tc && commitListener) {
        auto cpu = tc->getCpuPtr();
        auto listener = static_cast<CommitProbeListener *>(commitListener);
        cpu->getProbeManager()->removeListener("Commit", *listener);
        delete listener;
    }
}

void
StreamEngine::clear()
{
    ivs.clear();
    mems.clear();
    committedConfigs = 0;
    isReady = false;
    pcMemIdPairs.clear();
    currentIndvars.clear();
}

void
StreamEngine::initThreadContext(ThreadContext *_tc)
{
    // Update thread context.
    tc = _tc;
    // Create a new commit listener.
    auto cpu = tc->getCpuPtr();
    if (dynamic_cast<o3::CPU *>(cpu)) {
        commitListener = new CommitProbeListener(
            cpu->getProbeManager(), "Commit");
    }
}

void
StreamEngine::serialize(CheckpointOut &cp) const
{
    // TODO
    panic("Unimplemented!");
}

void
StreamEngine::unserialize(CheckpointIn &cp)
{
    // TODO
    panic("Unimplemented!");
}

bool
StreamEngine::configIndvar(ExecContext *xc, RegVal init_val,
        RegVal step_val, RegVal final_val, SmxStopCond cond,
        unsigned width, bool is_unsigned)
{
    if (cond < SMX_COND_GT || cond > SMX_COND_NE) {
        DPRINTF(StreamEngine, "Unsupported stop condition %u "
            "for induction variable stream at PC=0x%llx\n",
            cond, xc->pcState().instAddr());
        return false;
    }
    // Add configuration if is not O3 (i.e. not speculating).
    if (!dynamic_cast<o3::CPU *>(xc->tcBase()->getCpuPtr())) {
        addIndvarConfig(init_val, step_val, final_val, cond, width,
            is_unsigned);
    }
    return true;
}

void
StreamEngine::configMemory(ExecContext *xc, RegVal base, RegVal stride1,
        unsigned dep1, SmxStreamKind kind1, bool prefetch, unsigned width)
{
    // Add configuration if is not O3 (i.e. not speculating).
    if (!dynamic_cast<o3::CPU *>(xc->tcBase()->getCpuPtr())) {
        addMemoryConfig(base, stride1, dep1, kind1, prefetch, width);
    }
}

void
StreamEngine::configAddr(ExecContext *xc,
        RegVal stride1, unsigned dep1, SmxStreamKind kind1,
        RegVal stride2, unsigned dep2, SmxStreamKind kind2)
{
    // Add configuration if is not O3 (i.e. not speculating).
    if (!dynamic_cast<o3::CPU *>(xc->tcBase()->getCpuPtr())) {
        addAddrConfig(stride1, dep1, kind1, stride2, dep2, kind2);
    }
}

bool
StreamEngine::ready(ExecContext *xc, const SmxOp *op, unsigned conf_num,
        bool &wait)
{
    // Check committed configurations.
    if (committedConfigs < conf_num) {
        wait = true;
        return true;
    } else if (committedConfigs == conf_num) {
        wait = false;
    } else {
        DPRINTF(StreamEngine, "Unmatched configuration number\n");
        return false;
    }

    // Check configurations.
    if (ivs.empty()) {
        DPRINTF(StreamEngine, "No induction variable stream configured\n");
        return false;
    }
    if (ivs.size() > MAX_INDVAR_NUM) {
        DPRINTF(StreamEngine, "Induction variable stream number exceeded! "
            "Currently supports %u induction variable streams\n",
            MAX_INDVAR_NUM);
        return false;
    }
    if (mems.size() > MAX_MEMORY_NUM) {
        DPRINTF(StreamEngine, "Memory stream number exceeded! "
            "Currently supports %u memory streams\n",
            MAX_MEMORY_NUM);
        return false;
    }
    for (unsigned memory_id = 0; memory_id < mems.size(); ++memory_id) {
        const auto &addrs = mems[memory_id].addrs;
        if (addrs.size() > MAX_ADDR_NUM) {
            DPRINTF(StreamEngine,
                "Address factor number of memory stream %u exceeded! "
                "Currently supports %u address factors\n",
                memory_id, MAX_ADDR_NUM);
            return false;
        }
        for (const auto &addr : addrs) {
            if (!isValidStream(addr.dep, addr.kind)) return false;
            if (addr.kind == SMX_KIND_MS) {
                DPRINTF(StreamEngine,
                    "Try to configure indirect memory acess for "
                    "memory stream %u! Currently does not support "
                    "indirect memory access\n",
                    memory_id);
                return false;
            }
        }
    }

    // Initialize induction variables.
    for (unsigned i = 0; i < ivs.size(); ++i) {
        setIndvarDestReg(xc, op, i, ivs[i].initVal);
    }

    // Enable prefetch if is not O3 (i.e. not speculating).
    if (!dynamic_cast<o3::CPU *>(xc->tcBase()->getCpuPtr())) {
        commitReady(xc);
    }
    return true;
}

void
StreamEngine::end(ExecContext *xc)
{
    // Add configuration if is not O3 (i.e. not speculating).
    if (!dynamic_cast<o3::CPU *>(xc->tcBase()->getCpuPtr())) commitEnd();
}

RegVal
StreamEngine::step(ExecContext *xc, const SmxOp *op, unsigned indvar_id)
{
    RegVal ret = 0;
    for (unsigned id = 0; id < ivs.size(); ++id) {
        auto &iv = ivs[id];
        auto value = getIndvarSrcReg(xc, op, id);
        // Update induction variable registers.
        if (id < indvar_id) {
            // No change in value.
            value = value;
        } else if (id == indvar_id) {
            // Step the current induction variable.
            value = applyWidthUnsigned(value + iv.stepVal, iv.width,
                iv.isUnsigned);
            ret = value;
        } else {
            // Reset to initial value.
            value = iv.initVal;
        }
        setIndvarDestReg(xc, op, id, value);
        DPRINTF(StreamEngine, "Updated induction variable stream %u = %llu\n",
            id, value);
    }
    // Update prefetch queue if the current CPU is not O3
    // (i.e. not speculating).
    if (!dynamic_cast<o3::CPU *>(xc->tcBase()->getCpuPtr())) {
        commitStep(xc, op);
    }
    return ret;
}

RegVal
StreamEngine::getIndvarSrcReg(ExecContext *xc, const SmxOp *op,
        unsigned indvar_id) const
{
    if (op->hasIndvarSrcs() && indvar_id < MAX_INDVAR_NUM) {
        auto value = xc->getRegOperand(op,
            op->numSrcRegs() - IndvarRegNum + indvar_id);
        DPRINTF(StreamEngine, "Got induction variable stream %u = %llu\n",
            indvar_id, value);
        return value;
    }
    GEM5_UNREACHABLE;
}

void
StreamEngine::setIndvarDestReg(ExecContext *xc, const SmxOp *op,
        unsigned indvar_id, RegVal value)
{
    if (op->hasIndvarDests() && indvar_id < MAX_INDVAR_NUM) {
        DPRINTF(StreamEngine, "Set induction variable stream %u = %llu\n",
            indvar_id, value);
        return xc->setRegOperand(op,
            op->numDestRegs() - IndvarRegNum + indvar_id, value);
    }
    GEM5_UNREACHABLE;
}

bool
StreamEngine::isValidStream(unsigned id, SmxStreamKind kind) const
{
    if (kind == SMX_KIND_IV) {
        if (id >= ivs.size()) {
            DPRINTF(StreamEngine, "Invalid induction variable stream %u\n",
                id);
            return false;
        }
        return true;
    } else if (kind == SMX_KIND_MS) {
        if (id >= mems.size()) {
            DPRINTF(StreamEngine, "Invalid memory stream %u\n", id);
            return false;
        }
        return true;
    } else {
        DPRINTF(StreamEngine, "Unsupported stream kind %u\n", kind);
        return false;
    }
}

Addr
StreamEngine::getMemoryAddr(ExecContext *xc, const SmxOp *op,
        unsigned memory_id)
{
    // Calculate virtual address.
    auto vaddr = getMemoryAddrWithIndvars(memory_id,
        [this, xc, op](unsigned id) {
            return getIndvarSrcReg(xc, op, id);
        });
    DPRINTF(StreamEngine, "Got memory address 0x%llx from stream %u\n",
        vaddr, memory_id);

    // Update PC-memory ID pairs.
    if (mems[memory_id].prefetch) {
        auto pc = xc->pcState().instAddr();
        auto it = std::find_if(pcMemIdPairs.begin(), pcMemIdPairs.end(),
            [pc](const auto &p) { return p.first == pc; });
        if (it != pcMemIdPairs.end()) {
            // Check if ID is same.
            assert(it->second == memory_id);
        } else {
            // Insert a new pair.
            pcMemIdPairs.push_back({pc, memory_id});
            if (pcMemIdPairs.size() > MAX_PC_MEM_ID_PAIRS) {
                pcMemIdPairs.pop_front();
            }
        }
    }
    return vaddr;
}

bool
StreamEngine::isNotInLoop(unsigned indvar_id, RegVal value) const
{
    const auto &iv = ivs[indvar_id];
    switch (iv.cond) {
      case SMX_COND_GT:
        return (int64_t)value > (ino64_t)iv.finalVal;
      case SMX_COND_GE:
        return (int64_t)value >= (ino64_t)iv.finalVal;
      case SMX_COND_LT:
        return (int64_t)value < (ino64_t)iv.finalVal;
      case SMX_COND_LE:
        return (int64_t)value <= (ino64_t)iv.finalVal;
      case SMX_COND_GTU:
        return value > iv.finalVal;
      case SMX_COND_GEU:
        return value >= iv.finalVal;
      case SMX_COND_LTU:
        return value < iv.finalVal;
      case SMX_COND_LEU:
        return value <= iv.finalVal;
      case SMX_COND_EQ:
        return value == iv.finalVal;
      case SMX_COND_NE:
        return value != iv.finalVal;
      default:
        GEM5_UNREACHABLE;
    }
}

void
StreamEngine::commitIndvarConfig(ExecContext *xc, const SmxOp *op)
{
    // Read operands.
    auto rs1 = xc->getRegOperand(op, 0);
    auto rs2 = xc->getRegOperand(op, 1);
    auto rs3 = xc->getRegOperand(op, 2);
    auto cond = static_cast<SmxStopCond>(bits(op->machInst, 10, 7));
    auto width = bits(op->machInst, 26, 25);
    auto is_unsigned = bits(op->machInst, 11);

    // Add configuration.
    addIndvarConfig(rs1, rs2, rs3, cond, width, is_unsigned);
}

void
StreamEngine::commitMemoryConfig(ExecContext *xc, const SmxOp *op)
{
    // Read operands.
    auto base = xc->getRegOperand(op, 0);
    auto stride1 = xc->getRegOperand(op, 1);
    auto dep1 = bits(op->machInst, 11, 7);
    auto kind1 = static_cast<SmxStreamKind>(bits(op->machInst, 25));
    auto prefetch = bits(op->machInst, 26);
    auto width = bits(op->machInst, 29, 27);

    // Add configuration.
    addMemoryConfig(base, stride1, dep1, kind1, prefetch, width);
}

void
StreamEngine::commitAddrConfig(ExecContext *xc, const SmxOp *op)
{
    // Read operands.
    auto stride1 = xc->getRegOperand(op, 0);
    auto dep1 = bits(op->machInst, 11, 7);
    auto kind1 = static_cast<SmxStreamKind>(bits(op->machInst, 25));
    auto stride2 = xc->getRegOperand(op, 1);
    auto dep2 = bits(op->machInst, 30, 26);
    auto kind2 = static_cast<SmxStreamKind>(bits(op->machInst, 31));

    // Add configuration.
    addAddrConfig(stride1, dep1, kind1, stride2, dep2, kind2);
}

void
StreamEngine::commitReady(ExecContext *xc)
{
    // Check if is not ready.
    if (xc->pcState().branching()) return;
    isReady = true;
    committedConfigs = 0;

    // Initialize current induction variables.
    for (unsigned i = 0; i < ivs.size(); ++i) {
        currentIndvars.push_back(ivs[i].initVal);
    }

    DPRINTF(StreamEngine, "Stream memory access is ready\n");
}

void
StreamEngine::commitEnd()
{
    // Reset state.
    clear();
    DPRINTF(StreamEngine, "Stream memory access ended\n");
}

void
StreamEngine::commitStep(ExecContext *xc, const SmxOp *op)
{
    auto indvar_id = bits(op->machInst, 19, 15);
    // Update the current induction variables.
    stepIndvars(currentIndvars, indvar_id,
        [this, xc, op](unsigned id) {
            return getIndvarSrcReg(xc, op, id);
        });
}

bool
StreamEngine::getRunaheadAddrForPc(Addr pc, Addr &vaddr)
{
    // Bail out if the stream engine is not ready.
    if (!isReady) return false;

    // Find for a PC-memory ID pair.
    auto it = std::find_if(pcMemIdPairs.begin(), pcMemIdPairs.end(),
        [pc](const auto &p) { return p.first == pc; });
    if (it == pcMemIdPairs.end()) {
        DPRINTF(StreamEngine, "PC=0x%llx is not found in stream engine\n",
            pc);
        return false;
    }

    // Debug for the current induction variables.
    auto iv_str = indvarsToString(currentIndvars);
    DPRINTF(StreamEngine, "Current induction variables: %s\n",
        iv_str.c_str());

    // Perform runahead.
    auto indvars = currentIndvars;
    auto iq = [&indvars](unsigned id) { return indvars[id]; };
    unsigned step = 0;
    for (step = 0; step < NUM_RUNAHEAD_STEPS; ++step) {
        if (stepIndvars(indvars, indvars.size() - 1, iq)) {
            break;
        }
    }

    // Bail out if runahead steps reached the EOL.
    if (step < NUM_RUNAHEAD_STEPS) {
        DPRINTF(StreamEngine, "Runahead steps=%u reached the EOL\n", step);
        return false;
    }

    // Debug for the induction variables after runahead.
    iv_str = indvarsToString(indvars);
    DPRINTF(StreamEngine, "Induction variables after runahead: %s\n",
        iv_str.c_str());

    // Get virtual address.
    vaddr = getMemoryAddrWithIndvars(it->second, iq);
    return true;
}

} // namespace RiscvISA
} // namespace gem5
