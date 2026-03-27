# CAF-CUDA Scheduler Starvation: Root Cause Analysis and Design

*Based on inspection of `libcaf_cuda`, confirmed empirically by example_10 and example_11.*

---

## 1. The Observed Behaviour

With `caf.scheduler.max-threads = 1`:

```
[AsyncTestActor] run_async() returned immediately.
[GPU] Kernel started. Delaying for 4500000000 cycles...
[GPU] Kernel finished delay.
[AsyncTestActor] CPU ping #1   ← first ping, after ~3 seconds
[BusyActor]      Tick #1       ← BusyActor also completely frozen
```

- `run_async()` prints its confirmation line before the kernel output → it does return to the handler body.
- Zero pings and zero ticks fire during the ~3-second kernel window.
- Both GPU `printf` messages flush together, immediately followed by a burst of backlogged pings/ticks.
- This is **not** a stdout-ordering artefact: the CAF worker thread was genuinely unavailable.

---

## 2. Execution Path Trace

```
caf_main()
  └─ actor handler for `int` message fires on the single worker thread
       │
       ├─ mgr.create_program_from_cubin(...)
       │    └─ program::program() → program::load_kernels()
       │         └─ cuCtxPushCurrent(ctx)
       │         └─ cuModuleLoadData(&module, binary)   ← blocking file+JIT load
       │         └─ cuCtxPopCurrent()
       │              └─ returns program_ptr (refcount = 1, held by local var)
       │
       ├─ runner.run_async(program, dim, actor_id, arg1, arg2)
       │    └─ [ copies program_ptr into base_command, refcount → 2 ]
       │    └─ base_command::base_enqueue()
       │         └─ device::launch_kernel_mem_ref(...)
       │              ├─ get_stream_for_actor()     ← acquires mutex, returns CUstream
       │              ├─ scratch_argument(out<int>) → cuMemAlloc (non-blocking)
       │              ├─ cuCtxPushCurrent(ctx)
       │              ├─ cuLaunchKernel(...)         ← ASYNC: kernel enqueued to GPU.
       │              │                                 Returns immediately.
       │              └─ cuCtxPopCurrent()
       │    └─ base_command destructs when cmd goes out of scope
       │         └─ program_ptr refcount → 1 (local handler var still holds it)
       │
       ├─ println("run_async() returned immediately.")   ← PRINTED HERE
       ├─ result_ptr_ = std::get<1>(result_tuple)
       ├─ schedule "fetch_result" for T+4 s
       │
       └─ *** HANDLER LAMBDA EXITS ***
            └─ local `program` variable destructs
                 └─ program_ptr refcount → 0
                      └─ program::~program()
                           └─ cuModuleUnload(module)   ◄══ BLOCKING POINT
```

---

## 3. Primary Theory — `cuModuleUnload` Implicit GPU Synchronisation

### What happens

`cuModuleUnload()` is called in `program::~program()`. At the time it is called, the GPU is still executing the delay kernel: the kernel draws instructions directly from the loaded module binary. On Maxwell hardware (sm_52, used here) the CUDA driver **implicitly calls `cuCtxSynchronize()`** inside `cuModuleUnload` as a safety guard before releasing the module pages, because releasing a code page that a running kernel is fetching from would corrupt the SM instruction cache.

The CUDA specification says the behaviour is *undefined* if a kernel from the module is still in flight when the module is unloaded. Undefined means the driver is free to do whatever it needs to do — on Maxwell it chooses to wait.

### Why it blocks the whole scheduler

The CAF scheduler is a work-stealing thread pool. With `max-threads=1` there is exactly one worker thread. That thread is the one executing the actor's message handler. While the handler's local-variable destructor is running (`cuModuleUnload` → implicit `cuCtxSynchronize`), the thread is stuck. No other actor — including `BusyActor` — can be scheduled.

### Why the GPU `printf` messages appear together, after `run_async()`

CUDA's device-side `printf` writes to an on-device ring buffer. That buffer is only flushed to the host `stdout` when the CUDA runtime performs a context synchronisation (`cuCtxSynchronize` or `cuStreamSynchronize`). Because the implicit sync inside `cuModuleUnload` is the *first* such synchronisation in this run, both GPU messages (`Kernel started` and `Kernel finished`) are flushed simultaneously right at that point — which explains why they appear together, directly before the burst of accumulated pings.

### Confirming evidence in the source

| File | Line(s) | Detail |
|---|---|---|
| [program.hpp](../actor-framework/libcaf_cuda/caf/cuda/program.hpp) | `program::program()` | Calls `load_kernels()` → `cuModuleLoadData` on every construction |
| [program.cpp](../actor-framework/libcaf_cuda/src/program.cpp) | `program::~program()` | Calls `cuModuleUnload(module)` for every loaded device |
| [command_runner.hpp](../actor-framework/libcaf_cuda/caf/cuda/command_runner.hpp) | `run_async()` | Passes `program_ptr` by value → copy goes into `base_command`, destructs at end of `run_async()` |
| [command.hpp](../actor-framework/libcaf_cuda/caf/cuda/command.hpp) | `base_enqueue()` | `cuLaunchKernel` is the only CUDA call that is async; everything else before it is blocking |
| [device.hpp](../actor-framework/libcaf_cuda/caf/cuda/device.hpp) | `launch_kernel_mem_ref` | No `cuStreamSynchronize` is called after launch — the launch itself is correct |

---

## 4. Secondary Theory — `CU_CTX_SCHED_AUTO` Spin-Wait Amplifies the Block

### The flag

In `platform::platform()` ([platform.cpp](../actor-framework/libcaf_cuda/src/platform.cpp)):
```cpp
cuCtxCreate(&contexts_[i], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, cuda_device);
```

`CU_CTX_SCHED_AUTO` tells the driver to auto-select the wait strategy. The heuristic is:

> If (number of active CUDA contexts C) ≤ (number of logical CPU processors P): use **SPIN**.
> If C > P: use **YIELD**.

On a typical development machine with 1 CUDA context and > 1 logical cores, C ≤ P, so the driver defaults to **SPIN** (busy-wait). When the implicit `cuCtxSynchronize` inside `cuModuleUnload` waits for the GPU, it does so by spinning — consuming 100% of the worker thread's CPU time rather than yielding the thread to the OS scheduler.

### Effect on CAF

CAF's work-stealing scheduler uses OS yield/sleep to cooperate with other threads. A CUDA spin-wait pre-empts this: the OS sees the thread as "running" and will not donate its time slice to other worker threads. Even if there were a second worker thread, on a loaded machine the spinlock degrades throughput. With `max-threads=1` and spinning, the result is total scheduler starvation.

---

## 5. Tertiary Theory — `actor_facade` Sync Path (Different Code Path)

For completeness: the `actor_facade` class ([actor_facade.hpp](../actor-framework/libcaf_cuda/caf/cuda/actor_facade.hpp)) uses `command::enqueue()` which calls `collect_output_buffers()` → `mem_ref::copy_to_host()` → **`cuStreamSynchronize()`**. This is an *unconditionally blocking* call inside the actor's message handler regardless of the scheduling flag. It is not the cause of the starvation in example_10/11 (which use `command_runner::run_async()`), but it means that any future user who switches to `actor_facade` faces the same problem on the very first kernel invocation, with no workaround short of not using `actor_facade`.

---

## 6. The `command_runner` Is Not the Problem — The Usage Pattern Is

To be precise: `base_enqueue()` itself is correct. After `cuLaunchKernel` returns, the GPU is running the kernel asynchronously and the CPU is free. The flaw is in the *lifetime management* of `program`:

```
actor handler body
├── local `program`   ← created here
├── run_async(program)
│    └── cuLaunchKernel  ← GPU starts; CPU returns from here
└── handler exits
     └── ~program()   ← cuModuleUnload called HERE while GPU is still running
```

The `program` object is a transient local variable. Its destructor races with the running kernel.

---

## 7. Design Solutions

### 7.1 Immediate Fix: Store `program_ptr` as an Actor Member (fixes Theory 1)

Move `program_ptr` to be a member of the actor state instead of a local variable in the handler. The destructor then only fires when the actor itself is torn down — at which point no kernel is running because the actor has already called `quit()`.

```cpp
class AsyncTestActor {
    caf::cuda::program_ptr program_;   // ← member, not local variable

    behavior make_behavior() {
        return {
            [this](int delay_seconds) {
                auto& mgr = self_->system().cuda_manager();
                program_ = mgr.create_program_from_cubin("delay.cubin", "myKernel");
                // ... run_async, etc.
                // program_ stays alive for the actor's entire lifetime
                // cuModuleUnload only happens when the actor is destroyed
            }
        };
    }
};
```

This is the minimum change needed to confirm or deny Theory 1 experimentally.

### 7.2 Fix Spin-Wait: Use `CU_CTX_SCHED_BLOCKING_SYNC` (mitigates Theory 2)

Change the context creation flag in `platform::platform()`:

```cpp
// Before:
cuCtxCreate(&contexts_[i], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, cuda_device);

// After:
cuCtxCreate(&contexts_[i], CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, cuda_device);
```

`CU_CTX_SCHED_BLOCKING_SYNC` makes any CUDA wait call sleep (OS-managed) rather than spin. With `max-threads=1` this still blocks the thread, but with `max-threads=2+` the sleeping thread yields its time slice and the second worker thread can run other actors during the wait. This alone does NOT solve the starvation with a single worker thread, but it is important hygiene regardless.

### 7.3 Non-Blocking Polling: `cuStreamQuery` in a CAF Timer

Replace the blocking synchronisation with periodic non-blocking polling. After launching:

```
actor launches kernel → saves CUstream → returns handler immediately
     ↓
CAF timer fires every ~5 ms
     ↓
handler calls cuStreamQuery(stream)
     ├── CUDA_ERROR_NOT_READY → reschedule timer, yield thread
     └── CUDA_SUCCESS         → stream done, safe to cuMemcpyDtoH, notify caller
```

`cuStreamQuery` is a non-blocking call that returns immediately. The worker thread is therefore never held captive. All other actors continue to run between polls. The resolution is only as coarse as the polling interval, but 5 ms is negligible for GPU workloads that take hundreds of milliseconds.

### 7.4 CUDA Stream Callbacks: `cuLaunchHostFunc` (True Async, Cleanest Design)

CUDA 10+ provides `cuLaunchHostFunc(stream, func, userData)` which enqueues a host-side function on the stream. When the GPU finishes all preceding work on that stream, the CUDA runtime calls `func(userData)` from its own internal thread — never from a CAF worker thread.

```
cuLaunchKernel(...)           // enqueue GPU work
cuLaunchHostFunc(stream, cb, actor_handle)  // enqueue completion callback
// worker thread returns immediately

// CUDA runtime internal thread:
// (after kernel finishes)
cb(actor_handle):
    anon_mail("gpu_done").send(actor_handle);   // wake up the actor

// actor handles "gpu_done" message:
[this](const std::string& msg) {
    if (msg == "gpu_done") {
        // cuStreamSynchronize here is instantaneous (stream already idle)
        auto result = result_ptr_->copy_to_host();
        // ...
    }
}
```

No CAF worker thread is ever blocked waiting for the GPU. The actor is woken exactly when the GPU is done. This is the correct long-term architectural pattern.

### 7.5 Dedicated CUDA Synchronisation Thread

For maximum portability and driver compatibility (including older sm_52 hardware that may not support `cuLaunchHostFunc` well):

Spawn one dedicated OS thread (not a CAF actor) whose sole responsibility is to watch GPU streams and post CAF messages on completion:

```
GPU actor:
    cuLaunchKernel(stream)
    post (stream, reply_actor) to the sync-thread's work queue
    return from handler immediately

sync-thread loop:
    for each (stream, reply_actor) in queue:
        cuStreamSynchronize(stream)         // blocks THIS dedicated thread only
        anon_mail("gpu_done").send(reply_actor)  // notifies actor via CAF
```

The worker thread(s) of the CAF scheduler are never touched. The dedicated thread can block indefinitely without affecting actor throughput. This is analogous to CAF's own `io_module` which uses dedicated network I/O threads for the same reason.

---

## 8. Can CUDA Truly Support Asynchronous Execution?

**Yes — but only if the application never calls a blocking CUDA synchronisation (cuStreamSynchronize, cuCtxSynchronize, cuDeviceSynchronize) on a CAF worker thread.**

The GPU itself is a fully asynchronous device. `cuLaunchKernel` returns to the host immediately; the kernel runs concurrently with host code on its own hardware. CUDA streams are specifically designed to support overlapping CPU and GPU execution. The GPU hardware is not the problem.

The problem is entirely in the *host-side bridge*:

| Mechanism | Blocks worker thread? | Notes |
|---|---|---|
| `cuLaunchKernel` | **No** | Fully async |
| `cuMemcpyHtoDAsync` | **No** | Async with stream |
| `cuStreamQuery` | **No** | Non-blocking poll, returns immediately |
| `cuLaunchHostFunc` callback | **No** | Fires on CUDA internal thread |
| `cuStreamSynchronize` | **Yes** | Spins or sleeps waiting for GPU |
| `cuCtxSynchronize` | **Yes** | Spins or sleeps, whole context |
| `cuModuleUnload` (sm_52) | **Yes** | Implicit sync before releasing module |
| `command::enqueue()` | **Yes** | Calls `copy_to_host()` → `cuStreamSynchronize` |

True async GPU use from within CAF requires routing all blocking CUDA calls through one of:
a) A non-blocking poll mechanism (Solutions 7.3)
b) CUDA stream callbacks that post CAF messages (Solution 7.4)
c) A dedicated sync thread that handles blocking and re-enters CAF via `anon_mail` (Solution 7.5)

The current architecture has the right shape — `base_enqueue()` correctly issues the async launch and returns `mem_ptr` handles — but it is undermined by the `program_ptr` lifetime issue (Theory 1) and the synchronous `copy_to_host()` in `mem_ref` which must only ever be called once the stream is known to be idle.

---

## 9. Summary of Theories

| # | Name | Location in code | Severity |
|---|---|---|---|
| **1** | `cuModuleUnload` implicit GPU sync | `program::~program()`, triggered by handler-local `program` variable | **Critical** — blocks entire scheduler for kernel duration |
| **2** | `CU_CTX_SCHED_AUTO` spin-wait | `platform::platform()` context creation flag | **Amplifier** — turns a blocking wait into 100% CPU spin, worsening starvation |
| **3** | `actor_facade::enqueue()` sync path | `actor_facade.hpp` → `command::enqueue()` → `copy_to_host()` | **Structural** — affects all `actor_facade` users; always synchronous |
