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
#include <sstream>
#include <utility>

#include "arch/generic/mmu.hh"
#include "arch/riscv/isa.hh"
#include "arch/riscv/regs/int.hh"
#include "base/bitfield.hh"
#include "base/compiler.hh"
#include "base/logging.hh"
#include "cpu/exec_context.hh"
#include "cpu/thread_context.hh"
#include "cpu/o3/dyn_inst.hh"
#include "debug/StreamEngine.hh"
#include "mem/port.hh"
#include "sim/eventq.hh"
#include "sim/probe/probe.hh"

namespace
{

using namespace gem5;

constexpr unsigned MAX_INDVAR_NUM = RiscvISA::IndvarRegNum;
constexpr unsigned MAX_MEMORY_NUM = 32;
constexpr unsigned MAX_ADDR_NUM = 4;

constexpr unsigned MAX_PREF_QUEUE_ENTRIES = 64;
constexpr unsigned NUM_REQS_PER_PREF = 8;
constexpr unsigned MAX_PREF_REQ_QUEUE_ENTRIES =
    MAX_PREF_QUEUE_ENTRIES * NUM_REQS_PER_PREF;
constexpr unsigned MAX_PREF_QUEUE_CONSUME_ENTRIES = 2;

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

        if (op->getName().rfind("smx_step", 0) == 0)
            return se.commitStep(inst.get(), op);
    }
};

/**
 * Address translation request.
 */
class AddrTranslationReq : public BaseMMU::Translation
{
  private:
    unsigned reqId;

  public:
    AddrTranslationReq(unsigned _reqId) : reqId(_reqId) {}

    void markDelayed() override { /* Nothing to do. */ }

    void
    finish(const Fault &fault, const RequestPtr &req,
        ThreadContext *tc, BaseMMU::Mode mode)
    {
        auto isa = static_cast<RiscvISA::ISA *>(tc->getIsaPtr());
        auto &se = isa->streamEngine();

        // Tell stream engine to update request's state.
        se.finishAddrTranslation(reqId, fault != NoFault);
    }
};

/**
 * Inject memory channel between CPU and cache/memory.
 */
class MemoryChannelInjector : public Packet::SenderState
{
  private:
    class ChannelResponse : public ResponsePort
    {
      private:
        MemoryChannelInjector &mci;

      protected:
        Tick
        recvAtomicBackdoor(PacketPtr pkt, MemBackdoorPtr &backdoor) override
        {
            return mci.request.sendAtomicBackdoor(pkt, backdoor);
        }

        bool
        tryTiming(PacketPtr pkt) override
        {
            return mci.request.tryTiming(pkt);
        }

        bool
        recvTimingSnoopResp(PacketPtr pkt) override
        {
            return mci.request.sendTimingSnoopResp(pkt);
        }

        Tick
        recvAtomic(PacketPtr pkt) override
        {
            return mci.request.sendAtomic(pkt);
        }

        bool
        recvTimingReq(PacketPtr pkt) override
        {
            return mci.request.sendTimingReq(pkt);
        }

        void
        recvRespRetry() override
        {
            mci.request.sendRetryResp();
        }

        void
        recvFunctional(PacketPtr pkt) override
        {
            mci.request.sendFunctional(pkt);
        }

      public:
        ChannelResponse(MemoryChannelInjector &_mci)
            : ResponsePort("stream_engine.mci.response", nullptr),
                mci(_mci)
        {
        }

        AddrRangeList
        getAddrRanges() const override
        {
            return mci.request.getAddrRanges();
        }
    };

    friend ChannelResponse;

    class ChannelRequest : public RequestPort
    {
      private:
        MemoryChannelInjector &mci;

      protected:
        void
        recvRangeChange() override
        {
            mci.response.sendRangeChange();
        }

        Tick
        recvAtomicSnoop(PacketPtr pkt) override
        {
            return mci.response.sendAtomicSnoop(pkt);
        }

        void
        recvFunctionalSnoop(PacketPtr pkt) override
        {
            mci.response.sendFunctionalSnoop(pkt);
        }

        void
        recvTimingSnoopReq(PacketPtr pkt) override
        {
            mci.response.sendTimingSnoopReq(pkt);
        }

        void
        recvRetrySnoopResp() override
        {
            mci.response.sendRetrySnoopResp();
        }

        bool
        recvTimingResp(PacketPtr pkt) override
        {
            if (pkt->senderState ==
                    static_cast<Packet::SenderState *>(&mci)) {
                // The packet is a prefetch request, delete it.
                delete pkt;
                return true;
            } else {
                return mci.response.sendTimingResp(pkt);
            }
        }

        void
        recvReqRetry() override
        {
            mci.retryPrefetch();
            mci.response.sendRetryReq();
        }

      public:
        ChannelRequest(MemoryChannelInjector &_mci)
            : RequestPort("stream_engine.mci.request", nullptr),
                mci(_mci)
        {
        }

        bool
        isSnooping() const override
        {
            return mci.response.isSnooping();
        }
    };

    friend ChannelRequest;

    ChannelResponse response;
    ChannelRequest request;
    RiscvISA::StreamEngine &se;
    RequestPort &cpuReq;
    std::queue<std::pair<unsigned, PacketPtr>> pendingRetries;

    void
    retryPrefetch()
    {
        for (unsigned i = 0; i < NUM_REQS_PER_PREF; ++i) {
            if (pendingRetries.empty()) return;

            // Pick a pending packet from queue and send.
            auto [id, packet] = pendingRetries.front();
            if (!request.sendTimingReq(packet)) {
                // Cache blocked, wait for next retry.
                return;
            } else {
                // Inform the stream engine that retry finished.
                se.finishRetry(id);
                // Remove the packet from the queue.
                pendingRetries.pop();
            }
        }
    }

  public:
    MemoryChannelInjector(RiscvISA::StreamEngine &_se, RequestPort &_cpuReq)
        : response(*this), request(*this), se(_se), cpuReq(_cpuReq)
    {
        auto &peer = cpuReq.getPeer();
        cpuReq.bind(response);
        request.bind(peer);
    }

    ~MemoryChannelInjector()
    {
        auto &peer = request.getPeer();
        request.unbind();
        cpuReq.bind(peer);
    }

    bool
    sendPrefetchReq(unsigned req_id, const RequestPtr &req)
    {
        // Create a new packet.
        auto packet = Packet::createRead(req);
        packet->allocate();
        packet->senderState = this;
        // Send packet.
        if (!request.sendTimingReq(packet)) {
            // Failed, push to retry queue.
            pendingRetries.push({req_id, packet});
            return false;
        }
        return true;
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
        kind_str = "induction variable";
    } else {
        kind_str = "memory";
    }
    DPRINTF(StreamEngine,
        "Added address factor stride=%llu, dependents %s"
        " stream %u for memory stream %u\n",
        stride, kind_str, dep, memory_id);
    return true;
}

void
StreamEngine::schedulePrefetch(ThreadContext *tc)
{
    auto event = new EventFunctionWrapper(
        [this, tc] { prefetchNext(tc); }, "stream_engine", true);
    auto cpu = tc->getCpuPtr();
    cpu->schedule(event, cpu->clockEdge(Cycles(1)));
}

void
StreamEngine::prefetchNext(ThreadContext *tc)
{
    if (!prefetchEnable) {
        DPRINTF(StreamEngine, "Prefetcher stopped\n");
        // Do not schedule the next event, just return.
        return;
    }

    // Remove finished prefetch requests.
    while (!requestQueue.empty() &&
        requestQueue.front().state == Finished) {
        requestQueue.pop_front();
    }

    // Update prefetch queue.
    if (prefetchQueue.size() >= MAX_PREF_QUEUE_ENTRIES) {
        DPRINTF(StreamEngine,
            "Prefetcher stalled due to prefetch queue is full\n");
    } else {
        DPRINTF(StreamEngine, "Prefetcher running\n");
        // Send prefetch request to LSU.
        enqueuePrefetchReq(tc);
        // Check if all memory streams are prefetched
        if (prefetchMemStreamIdx >= mems.size()) {
            prefetchMemStreamIdx = 0;
            // Prefetch complete, update queue and state.
            prefetchQueue.push(prefetchIndvars);
            // Update induction variables and state.
            if (stepPrefetchIndvars()) prefetchEnable = false;
        }
    }

    // Update request queue.
    for (unsigned i = 0;
        i < NUM_REQS_PER_PREF && i < requestQueue.size(); ++i)
    {
        handlePrefetchReq(tc, i);
    }

    // Schedule for the next cycle.
    schedulePrefetch(tc);
}

void
StreamEngine::enqueuePrefetchReq(ThreadContext *tc)
{
    // Skip unprefetchable streams.
    while (!mems[prefetchMemStreamIdx].prefetch) ++prefetchMemStreamIdx;

    for (unsigned i = 0;
        i < NUM_REQS_PER_PREF && prefetchMemStreamIdx < mems.size();
        ++i)
    {
        // Check if we can enqueue new request.
        if (requestQueue.size() >= MAX_PREF_REQ_QUEUE_ENTRIES) {
            DPRINTF(StreamEngine, "Can not enqueue prefetch request "
                "due to request queue full\n");
            return;
        }

        // Get memory address for prefetch.
        const auto &mem = mems[prefetchMemStreamIdx];
        Addr vaddr = mem.base;
        for (const auto &addr : mem.addrs) {
            assert(addr.kind == SMX_KIND_IV);
            vaddr += prefetchIndvars[addr.dep] * addr.stride;
        }

        // Enqueue prefetch request.
        auto request = std::make_shared<Request>(
            vaddr, 1 << mem.width, Request::PREFETCH,
            tc->getCpuPtr()->dataRequestorId(), -1, -1);
        requestQueue.push_back({Ready, std::move(request)});
        DPRINTF(StreamEngine,
            "Enqueued prefetch request for address 0x%llx\n", vaddr);

        // Update memory stream index, skip unprefetchable streams.
        ++prefetchMemStreamIdx;
        while (prefetchMemStreamIdx < mems.size() &&
            !mems[prefetchMemStreamIdx].prefetch)
        {
            ++prefetchMemStreamIdx;
        }
    }
}

void
StreamEngine::handlePrefetchReq(ThreadContext *tc, unsigned req_id)
{
    auto &req = requestQueue[req_id];
    switch (req.state) {
      case Ready: {
        // Send address translation request to MMU.
        auto trans = new AddrTranslationReq(req_id);
        tc->getMMUPtr()->translateTiming(
            req.request, tc, trans, BaseMMU::Read);
        break;
      }
      case Pending:
        // Wait for the callback to do the rest things.
        break;
      case Finished:
      default:
        GEM5_UNREACHABLE;
    }
}

void
StreamEngine::consumePrefetchQueue(const std::vector<RegVal> &indvars)
{
    if (!prefetchEnable) return;
    // Check the first few entries of the prefetch queue.
    for (unsigned i = 0; i < MAX_PREF_QUEUE_CONSUME_ENTRIES; ++i) {
        if (prefetchQueue.empty()) break;
        if (indvars == prefetchQueue.front()) {
            std::string s;
            DPRINTF(StreamEngine,
                "Consumed prefetch queue entry before indvars=(%s)\n",
                (s = indvarsToString(indvars)).c_str());
            return;
        }
        prefetchQueue.pop();
    }
    // The current induction variables and the first few entries
    // of the queue are not the same, clear the queue and
    // reset induction variables.
    while (!prefetchQueue.empty()) prefetchQueue.pop();
    prefetchIndvars = indvars;
    prefetchMemStreamIdx = 0;
    {
        std::string s;
        DPRINTF(StreamEngine, "Prefetch queue cleared and indvars reset, "
            "indvars (%s) not found\n",
            (s = indvarsToString(indvars)).c_str());
    }
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
    prefetchEnable = false;
    prefetchIndvars.clear();
    prefetchMemStreamIdx = 0;
    while (!prefetchQueue.empty()) prefetchQueue.pop();
    requestQueue.clear();
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
    // Initialize induction variables.
    for (unsigned i = 0; i < ivs.size(); ++i) {
        setIndvarDestReg(xc, op, i, ivs[i].initVal);
        prefetchIndvars.push_back(ivs[i].initVal);
    }
    // Start prefetch.
    prefetchEnable = true;
    schedulePrefetch(xc->tcBase());
    // Setup commit listener for O3 CPU.
    auto cpu = xc->tcBase()->getCpuPtr();
    if (dynamic_cast<o3::CPU *>(cpu)) {
        commitListener = new CommitProbeListener(
            cpu->getProbeManager(), "Commit");
    }
    // Inject memory channel.
    memChannelInjector = new MemoryChannelInjector(
        *this, *dynamic_cast<RequestPort *>(&cpu->getDataPort()));
    DPRINTF(StreamEngine, "Stream memory access is ready\n");
    return true;
}

bool
StreamEngine::end(ExecContext *xc)
{
    // Reset state.
    clear();
    // Remove commit listener.
    auto cpu = xc->tcBase()->getCpuPtr();
    if (dynamic_cast<o3::CPU *>(cpu)) {
        auto listener = static_cast<CommitProbeListener *>(commitListener);
        cpu->getProbeManager()->removeListener("Commit", *listener);
        delete listener;
    }
    // Remove memory channel injector.
    auto mci = static_cast<MemoryChannelInjector *>(memChannelInjector);
    delete mci;
    DPRINTF(StreamEngine, "Stream memory access ended\n");
    return true;
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

void
StreamEngine::commitStep(ExecContext *xc, const SmxOp *step)
{
    auto indvar_id = bits(step->machInst, 19, 15);
    // Get stepped induction variables.
    std::vector<RegVal> indvars;
    indvars.resize(ivs.size());
    bool should_step = true;
    for (unsigned id = ivs.size() - 1; id <= ivs.size() - 1; --id) {
        auto &iv = ivs[id];
        if (id > indvar_id) {
            indvars[id] = iv.initVal;
        } else {
            auto value = getIndvarSrcReg(xc, step, id);
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
    // Update prefetch queue.
    if (!should_step) consumePrefetchQueue(indvars);
}

void
StreamEngine::finishAddrTranslation(unsigned req_id, bool has_fault)
{
    auto &req = requestQueue[req_id];

    // Ignore any kind of faults.
    if (has_fault) {
        DPRINTF(StreamEngine,
            "Address translation for vaddr=0x%llx has fault, ignored\n",
            req.request->getVaddr());
        req.state = Finished;
        return;
    }

    // Send load request to cache.
    DPRINTF(StreamEngine,
        "Address translation vaddr=0x%llx -> paddr=0x%llx done, "
        "sending load request to cache\n",
        req.request->getVaddr(), req.request->getPaddr());
    auto mci = static_cast<MemoryChannelInjector *>(memChannelInjector);
    if (mci->sendPrefetchReq(req_id, req.request)) {
        req.state = Finished;
    } else {
        DPRINTF(StreamEngine,
            "Load request %u failed due to cache blocked, retry pended\n",
            req_id);
        req.state = Retrying;
    }
}

void
StreamEngine::finishRetry(unsigned req_id)
{
    DPRINTF(StreamEngine, "Load request %u finished retry\n", req_id);
    requestQueue[req_id].state = Finished;
}

} // namespace RiscvISA
} // namespace gem5
