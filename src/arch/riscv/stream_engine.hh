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

#ifndef __ARCH_RISCV_STREAM_ENGINE_HH__
#define __ARCH_RISCV_STREAM_ENGINE_HH__

#include <deque>
#include <queue>
#include <vector>

#include "arch/riscv/insts/smx.hh"
#include "base/types.hh"
#include "mem/request.hh"
#include "sim/serialize.hh"

namespace gem5
{

class ExecContext;
class ThreadContext;

namespace RiscvISA
{

enum SmxStopCond
{
  SMX_COND_GT = 0,
  SMX_COND_GE = 1,
  SMX_COND_LT = 2,
  SMX_COND_LE = 3,
  SMX_COND_GTU = 4,
  SMX_COND_GEU = 5,
  SMX_COND_LTU = 6,
  SMX_COND_LEU = 7,
  SMX_COND_EQ = 8,
  SMX_COND_NE = 9,
};

enum SmxStreamKind
{
  SMX_KIND_IV = 0,
  SMX_KIND_MS = 1,
};

/**
 * Stream engine of the SMX ISA extension.
 *
 * @todo Support prefetch.
 */
class StreamEngine
{
  private:
    struct IndvarConfig
    {
        RegVal initVal;
        RegVal stepVal;
        RegVal finalVal;
        SmxStopCond cond;
        unsigned width;
        bool isUnsigned;
        std::vector<unsigned> users;
    };

    struct AddrConfig
    {
        RegVal stride;
        unsigned dep;
        SmxStreamKind kind;
    };

    struct MemoryConfig
    {
        RegVal base;
        bool prefetch;
        unsigned width;
        std::vector<AddrConfig> addrs;
        std::vector<unsigned> users;
    };

    std::vector<IndvarConfig> ivs;
    std::vector<MemoryConfig> mems;

    bool addAddrConfigForLastMem(RegVal stride, unsigned dep,
            SmxStreamKind kind);

    enum RequestState
    {
        Ready,
        Pending,
        Finished,
    };

    struct PrefetchRequest
    {
        RequestState state;
        RequestPtr request;
    };

    bool prefetchEnable;
    std::vector<RegVal> prefetchIndvars;
    unsigned prefetchMemStreamIdx;
    std::queue<std::vector<RegVal>> prefetchQueue;
    std::deque<PrefetchRequest> requestQueue;
    void *commitListener;

    void schedulePrefetch(ThreadContext *tc);
    void prefetchNext(ThreadContext *tc);
    void enqueuePrefetchReq(ThreadContext *tc);
    void handlePrefetchReq(ThreadContext *tc, unsigned req_id);

    /**
     * Called when committing `step` instructions.
     * @param indvars Induction variables after step (must be valid).
     */
    void consumePrefetchQueue(const std::vector<RegVal> &indvars);

    /**
     * Steps prefetch induction variables.
     * @return `true` if all induction variables
     *         are reset to initial value.
     */
    bool stepPrefetchIndvars();

  public:
    StreamEngine()
    {
        clear();
    }

    void clear();
    void serialize(CheckpointOut &cp) const;
    void unserialize(CheckpointIn &cp);

    bool addIndvarConfig(RegVal init_val, RegVal step_val, RegVal final_val,
            SmxStopCond cond, unsigned width, bool is_unsigned);

    bool addMemoryConfig(RegVal base, RegVal stride1, unsigned dep1,
            SmxStreamKind kind1, bool prefetch, unsigned width);

    bool addAddrConfig(RegVal stride1, unsigned dep1, SmxStreamKind kind1,
            RegVal stride2, unsigned dep2, SmxStreamKind kind2);

    bool ready(ExecContext *xc, const SmxOp *op);

    bool end(ExecContext *xc);

    RegVal step(ExecContext *xc, const SmxOp *op, unsigned indvar_id);

    RegVal getIndvarSrcReg(ExecContext *xc, const SmxOp *op,
            unsigned indvar_id) const;

    void setIndvarDestReg(ExecContext *xc, const SmxOp *op,
            unsigned indvar_id, RegVal value);

    bool isValidStream(unsigned id, SmxStreamKind kind) const;

    Addr getMemoryAddr(ExecContext *xc, const SmxOp *op,
            unsigned memory_id) const;

    bool isNotInLoop(unsigned indvar_id, RegVal value) const;

    /**
     * @ingroup Used by commit listener.
     * @{
     */

    void commitStep(ExecContext *xc, const SmxOp *step);

    /** @} */

    /**
     * @ingroup Used by prefetch requests.
     * @{
     */

    void finishAddrTranslation(unsigned req_id, bool has_fault);

    /** @} */
};

} // namespace RiscvISA
} // namespace gem5

#endif // __ARCH_RISCV_STREAM_ENGINE_HH__
