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

#include "mem/cache/prefetch/smx.hh"

#include "arch/riscv/isa.hh"
#include "arch/riscv/stream_engine.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/StreamEngine.hh"
#include "params/SMXPrefetcher.hh"
#include "sim/system.hh"

namespace gem5
{

namespace prefetch
{

RiscvISA::StreamEngine *
SMX::getSE()
{
    if (se) return se;

    // Get system.
    fatal_if(System::systemList.size() != 1,
        "There must be only one system");
    auto sys = System::systemList.front();

    // Get thread context.
    fatal_if(sys->threads.size() != 1,
        "There must be only one thread");
    auto tc = *sys->threads.begin();
    if (!tlb) tlb = tc->getMMUPtr()->dtb;

    // Get RISC-V ISA.
    auto isa = dynamic_cast<RiscvISA::ISA *>(tc->getIsaPtr());
    fatal_if(!isa, "ISA must be RISC-V");

    // Update stream engine pointer.
    se = &isa->streamEngine();
    return se;
}

SMX::SMX(const SMXPrefetcherParams &p) : Queued(p), se(nullptr)
{
}

void
SMX::calculatePrefetch(const PrefetchInfo &pfi,
                       std::vector<AddrPriority> &addresses)
{
    // PC is required.
    if (!pfi.hasPC()) {
        DPRINTF(StreamEngine, "Ignoring request with no PC\n");
        return;
    }

    auto se = getSE();

    // Try get virtual address for the current request.
    Addr vaddr;
    if (!se->getRunaheadAddrForPc(pfi.getPC(), vaddr)) return;

    addresses.push_back({vaddr, 0});
    DPRINTF(StreamEngine, "Calculated prefetch request vaddr=0x%llx\n",
        vaddr);
}

} // namespace prefetch
} // namespace gem5
