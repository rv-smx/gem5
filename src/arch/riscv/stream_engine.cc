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

#include <cassert>
#include <cstdint>

#include "arch/riscv/regs/int.hh"
#include "base/bitfield.hh"
#include "base/compiler.hh"
#include "base/logging.hh"
#include "cpu/base.hh"
#include "cpu/exec_context.hh"
#include "debug/StreamEngine.hh"
#include "sim/eventq.hh"

namespace
{

constexpr unsigned MAX_INDVAR_NUM = gem5::RiscvISA::IndvarRegNum;
constexpr unsigned MAX_MEMORY_NUM = 32;
constexpr unsigned MAX_ADDR_NUM = 4;

constexpr unsigned MAX_PREF_QUEUE_ENTRIES = 16;
constexpr unsigned MAX_PREF_QUEUE_CONSUME_ENTRIES = 2;

gem5::RegVal
applyWidthUnsigned(gem5::RegVal val, unsigned width, bool is_unsigned)
{
    switch (width) {
      case 0b00:
        return is_unsigned ? gem5::bits(val, 7, 0) : gem5::szext<8>(val);
      case 0b01:
        return is_unsigned ? gem5::bits(val, 15, 0) : gem5::szext<16>(val);
      case 0b10:
        return is_unsigned ? gem5::bits(val, 31, 0) : gem5::szext<32>(val);
      case 0b11:
        return val;
      default:
        GEM5_UNREACHABLE;
    }
}

} // namespace

namespace gem5
{

namespace RiscvISA
{

bool
StreamEngine::addAddrConfigForLastMem(RegVal stride, unsigned dep,
        SmxStreamKind kind)
{
    if (!isValidStream(dep, kind)) return false;
    auto &addrs = mems.back().addrs;
    unsigned memory_id = mems.size() - 1;

    if (addrs.size() >= MAX_ADDR_NUM) {
        DPRINTF(StreamEngine,
            "Address factor number of memory stream %u exceeded! "
            "Currently supports %u address factors\n",
            memory_id, MAX_ADDR_NUM);
        return false;
    }
    if (kind == SMX_KIND_MS) {
        DPRINTF(StreamEngine,
            "Try to configure indirect memory acess for memory stream %u! "
            "Currently does not support indirect memory access\n",
            memory_id);
        return false;
    }

    addrs.push_back({stride, dep, kind});
    const char *kind_str;
    if (kind == SMX_KIND_IV) {
        ivs[dep].users.push_back(memory_id);
        kind_str = "induction variable";
    } else {
        mems[dep].users.push_back(memory_id);
        kind_str = "memory";
    }
    DPRINTF(StreamEngine,
        "Added address factor stride=%llu, dependents %s"
        " stream %u for memory stream %u\n",
        stride, kind_str, dep, memory_id);
    return true;
}

void
StreamEngine::schedulePrefetch(BaseCPU *cpu)
{
    auto event = new EventFunctionWrapper(
        [this, cpu] { prefetchNext(cpu); }, "stream_engine", true);
    cpu->schedule(event, cpu->clockEdge(Cycles(1))); // TODO: 0?
}

void
StreamEngine::prefetchNext(BaseCPU *cpu)
{
    switch (prefetcherState) {
      case Stopped: 
        DPRINTF(StreamEngine, "Prefetcher stopped\n");
        // Do not schedule the next event, just return.
        return;
      case Running:
        DPRINTF(StreamEngine, "Prefetcher running\n");
        // Send prefetch request to LSU.
        sendPrefetchReq();
        // Check if all memory streams are prefetched
        if (prefetchMemStreamIdx < mems.size()) {
            break;
        } else {
            prefetchMemStreamIdx = 0;
        }
        // Prefetch complete, update queue and state.
        prefetchQueue.push(prefetchIndvars);
        if (prefetchQueue.size() >= MAX_PREF_QUEUE_ENTRIES) {
            prefetcherState = Full;
        }
        // Update induction variables and state.
        if (stepPrefetchIndvars()) prefetcherState = Stopped;
        break;
      case Full:
        DPRINTF(StreamEngine,
            "Prefetcher stalled due to prefetch queue is full\n");
        // Check if we can continue.
        if (prefetchQueue.size() < MAX_PREF_QUEUE_ENTRIES) {
            prefetcherState = Running;
        }
        break;
      default:
        GEM5_UNREACHABLE;
    }
    schedulePrefetch(cpu);
}

void
StreamEngine::sendPrefetchReq()
{
    // Skip unprefetchable streams.
    while (!mems[prefetchMemStreamIdx].prefetch) {
        ++prefetchMemStreamIdx;
        if (prefetchMemStreamIdx >= mems.size()) return;
    }
    // Get memory address for prefetch.
    const auto &mem = mems[prefetchMemStreamIdx];
    Addr vaddr = mem.base;
    for (const auto &addr : mem.addrs) {
        assert(addr.kind == SMX_KIND_IV);
        vaddr += prefetchIndvars[addr.dep] * addr.stride;
    }
    // Send prefetch request.
    // TODO
    DPRINTF(StreamEngine, "Sent prefetch request for address 0x%llx\n",
        vaddr);
    // Update memory stream index, skip unprefetchable streams.
    ++prefetchMemStreamIdx;
    while (prefetchMemStreamIdx < mems.size() &&
        !mems[prefetchMemStreamIdx].prefetch)
    {
        ++prefetchMemStreamIdx;
    }
}

void
StreamEngine::consumePrefetchQueue(const std::vector<RegVal> &cur_indvars)
{
    if (prefetcherState == Stopped) return;
    // Check the first few entries of the prefetch queue.
    for (unsigned i = 0; i < MAX_PREF_QUEUE_CONSUME_ENTRIES; ++i) {
        if (prefetchQueue.empty()) break;
        bool same = cur_indvars == prefetchQueue.front();
        prefetchQueue.pop();
        if (same) return;
    }
    // The current induction variables and the first few entries
    // of the queue are not the same, clear the queue and
    // reset induction variables.
    while (!prefetchQueue.empty()) prefetchQueue.pop();
    prefetchIndvars = cur_indvars;
    prefetchMemStreamIdx = 0;
    DPRINTF(StreamEngine, "Prefetch queue cleared and indvars reset\n");
}

bool
StreamEngine::stepPrefetchIndvars()
{
    unsigned id = ivs.size() - 1;
    for (auto it = prefetchIndvars.rbegin();
        it != prefetchIndvars.rend(); ++it, --id)
    {
        const auto &iv = ivs[id];
        *it = applyWidthUnsigned(*it + iv.stepVal, iv.width,
            iv.isUnsigned);
        if (isNotInLoop(id, *it)) {
            *it = iv.initVal;
            if (!id) return true;
        } else {
            break;
        }
    }
    return false;
}

void
StreamEngine::clear()
{
    ivs.clear();
    mems.clear();
    prefetcherState = Stopped;
    prefetchIndvars.clear();
    prefetchMemStreamIdx = 0;
    while (!prefetchQueue.empty()) prefetchQueue.pop();
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
StreamEngine::addIndvarConfig(RegVal _init_val, RegVal _step_val,
        RegVal _final_val, SmxStopCond cond, unsigned width, bool is_unsigned)
{
    if (ivs.size() >= MAX_INDVAR_NUM) {
        DPRINTF(StreamEngine, "Induction variable stream number exceeded! "
            "Currently supports %u induction variable streams\n",
            MAX_INDVAR_NUM);
        return false;
    }
    if (cond < SMX_COND_GT || cond > SMX_COND_NE) {
        DPRINTF(StreamEngine, "Unsupported stop condition %u "
            "for induction variable stream %u\n", cond, (unsigned)ivs.size());
        return false;
    }

    auto init_val = applyWidthUnsigned(_init_val, width, is_unsigned);
    auto step_val = applyWidthUnsigned(_step_val, width, is_unsigned);
    auto final_val = applyWidthUnsigned(_final_val, width, is_unsigned);
    DPRINTF(StreamEngine,
        "Induction variable stream %u: init=%llu, step=%llu, final=%llu\n",
        (unsigned)ivs.size(), init_val, step_val, final_val);
    ivs.push_back(
        {init_val, step_val, final_val, cond, width, is_unsigned, {}});
    return true;
}

bool
StreamEngine::addMemoryConfig(RegVal base, RegVal stride1, unsigned dep1,
        SmxStreamKind kind1, bool prefetch, unsigned width)
{
    if (mems.size() >= MAX_MEMORY_NUM) {
        DPRINTF(StreamEngine, "Memory stream number exceeded! "
            "Currently supports %u memory streams\n",
            MAX_MEMORY_NUM);
        return false;
    }
    DPRINTF(StreamEngine, "Memory stream %u: base=0x%llx\n",
        (unsigned)mems.size(), base);
    mems.push_back({base, prefetch, width, {}, {}});
    return addAddrConfigForLastMem(stride1, dep1, kind1);
}

bool
StreamEngine::addAddrConfig(
        RegVal stride1, unsigned dep1, SmxStreamKind kind1,
        RegVal stride2, unsigned dep2, SmxStreamKind kind2)
{
    if (!addAddrConfigForLastMem(stride1, dep1, kind1)) return false;
    return !stride2 || addAddrConfigForLastMem(stride2, dep2, kind2);
}

bool
StreamEngine::ready(ExecContext *xc, const SmxOp *op)
{
    if (ivs.empty()) {
        DPRINTF(StreamEngine, "No induction variable stream configured\n");
        return false;
    }
    for (unsigned i = 0; i < ivs.size(); ++i) {
        setIndvarDestReg(xc, op, i, ivs[i].initVal);
        prefetchIndvars.push_back(ivs[i].initVal);
    }
    prefetcherState = Running;
    auto cpu = xc->tcBase()->getCpuPtr();
    schedulePrefetch(cpu);
    DPRINTF(StreamEngine, "Stream memory access is ready\n");
    return true;
}

bool
StreamEngine::end()
{
    clear();
    DPRINTF(StreamEngine, "Stream memory access ended\n");
    return true;
}

RegVal
StreamEngine::step(ExecContext *xc, const SmxOp *op, unsigned indvar_id)
{
    // Update induction variable registers.
    std::vector<RegVal> cur_indvars;
    RegVal ret = 0;
    for (unsigned id = 0; id < ivs.size(); ++id) {
        auto &iv = ivs[id];
        auto value = getIndvarSrcReg(xc, op, id);
        // Record current value of all induction variables.
        if (id <= indvar_id) {
            cur_indvars.push_back(value);
        } else {
            cur_indvars.push_back(iv.initVal);
        }
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
    // Update prefetch queue.
    consumePrefetchQueue(cur_indvars);
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
        unsigned memory_id) const
{
    const auto &mem = mems[memory_id];
    Addr vaddr = mem.base;
    for (const auto &addr : mem.addrs) {
        assert(addr.kind == SMX_KIND_IV);
        vaddr += getIndvarSrcReg(xc, op, addr.dep) * addr.stride;
    }
    DPRINTF(StreamEngine, "Got memory address 0x%llx from stream %u\n",
        vaddr, memory_id);
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

} // namespace RiscvISA
} // namespace gem5
