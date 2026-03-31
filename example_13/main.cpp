// example_13: actor_facade is non-blocking — verified with 1 CPU thread.
//
// Demonstrates that actor_facade now uses the run_async_notify() pattern
// internally.  Before the fix, actor_facade::make_behavior() called
// cmd->enqueue() which called cuStreamSynchronize() — blocking the CAF
// worker thread for the full duration of the GPU kernel.
//
// After the fix the facade:
//   1. Calls base_enqueue() (returns immediately after cuLaunchKernel).
//   2. Registers a cuLaunchHostFunc callback on the same CUDA stream.
//   3. Returns to CAF with a response_promise (worker is FREE).
//   4. When the GPU stream goes idle, CUDA fires the callback which sends
//      gpu_done_atom to the facade; the facade then copies results to host
//      and delivers the response_promise.
//
// With a single CAF worker thread:
//   - BusyActor ticks MUST appear DURING the GPU delay if the fix works.
//   - If actor_facade were still blocking, BusyActor ticks would only appear
//     AFTER the kernel finishes (the scheduler would be frozen).
//
// Compare with example_9 (blocking actor_facade, 1 thread — BusyActor silent
// during GPU work) to see the difference the fix makes.

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "../common/kernel_paths.hpp"
#include <vector>
#include <iostream>
#include <chrono>

using namespace caf;
using namespace caf::cuda;

// ─────────────────────────────────────────────────────────────────────────────
// BusyActor: independent actor that only prints ticks.
// It has no knowledge of GPU work — it is purely a canary for scheduler health.
// If ticks appear DURING the GPU delay the scheduler is not blocked.
// ─────────────────────────────────────────────────────────────────────────────
class BusyActor {
    event_based_actor* self_;
    int tick_count_ = 0;

public:
    BusyActor(event_based_actor* self) : self_(self) {}

    behavior make_behavior() {
        return {
            [this](const std::string& msg) {
                if (msg == "start") {
                    self_->println("[BusyActor] Started tick loop (200 ms period).");
                    self_->mail(std::string("tick"))
                        .delay(std::chrono::milliseconds(200))
                        .send(self_);
                } else if (msg == "tick") {
                    tick_count_++;
                    self_->println("[BusyActor] Tick #{}", tick_count_);
                    self_->mail(std::string("tick"))
                        .delay(std::chrono::milliseconds(200))
                        .send(self_);
                } else if (msg == "stop") {
                    self_->println("[BusyActor] Stopping at tick #{}.", tick_count_);
                    self_->quit();
                }
            }
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// FacadeTestActor
//
// Spawns an actor_facade (via mgr.spawnFromCUBIN), sends it a kernel request,
// pings itself every 200 ms, and waits for the GPU result.
//
// With the fixed actor_facade the ping loop AND BusyActor ticks should both
// appear while the GPU kernel is running.
// ─────────────────────────────────────────────────────────────────────────────
class FacadeTestActor {
    event_based_actor* self_;
    int ping_count_ = 0;
    actor busy_actor_;

public:
    FacadeTestActor(event_based_actor* self) : self_(self) {}

    behavior make_behavior() {
        return {
            [this](int delay_seconds, actor busy) {
                busy_actor_ = busy;
                self_->println("[FacadeTestActor] Spawning actor_facade and sending kernel request...");

                nd_range dim(1, 1, 1, 1, 1, 1);

                // Spawn the GPU actor through the manager — this calls
                // actor_facade::create() which now uses run_async_notify()
                // internally.
                actor gpu_actor =
                    self_->system().cuda_manager().spawnFromCUBIN(
                        actor_tests::paths::delay_cubin, "delayKernel", dim,
                        in<int>{}, out<int>{});

                // Send the typed kernel arguments.  The facade will return
                // std::vector<output_buffer> asynchronously via response promise.
                self_->mail(create_in_arg(delay_seconds),
                            create_out_arg_with_size<int>(1))
                    .request(gpu_actor, caf::infinite)
                    .then(
                        [this](const std::vector<output_buffer>& result) {
                            std::vector<int> output = extract_vector<int>(result);
                            self_->println(
                                "[FacadeTestActor] GPU result={}. "
                                "CPU got to ping #{} while GPU was running.",
                                output[0], ping_count_);
                            self_->mail(std::string("stop")).send(busy_actor_);
                            self_->quit();
                        },
                        [this](const caf::error& err) {
                            self_->println("[FacadeTestActor] Error: {}",
                                          to_string(err));
                            self_->mail(std::string("stop")).send(busy_actor_);
                            self_->quit();
                        });

                // CPU ping loop — fires every 200 ms to show the scheduler
                // is alive and accepting messages throughout the GPU delay.
                self_->mail(std::string("ping"))
                    .delay(std::chrono::milliseconds(200))
                    .send(self_);

                self_->println(
                    "[FacadeTestActor] Kernel request sent. "
                    "Worker thread is now FREE — expecting pings below...");
            },

            [this](const std::string& msg) {
                if (msg == "ping") {
                    ping_count_++;
                    self_->println("[FacadeTestActor] CPU ping #{}", ping_count_);
                    self_->mail(std::string("ping"))
                        .delay(std::chrono::milliseconds(200))
                        .send(self_);
                }
            }
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Config: deliberately keep to 1 worker thread.
// If actor_facade were still blocking there would be no CPU pings at all
// during the GPU delay because the single worker would be stuck in
// cuStreamSynchronize.
// ─────────────────────────────────────────────────────────────────────────────
class config : public actor_system_config {
public:
    config() {
        set("caf.scheduler.max-threads", 1u);
    }
};

void caf_main(actor_system& sys, const config& cfg) {
    scoped_actor self{sys};

    self->println("=== example_13: actor_facade non-blocking verification ===");
    self->println("Scheduler max-threads: {}",
                  get_or(sys.config(), "caf.scheduler.max-threads", 0));
    self->println("Expected: BusyActor and FacadeTestActor ticks appear DURING "
                  "the 3-second GPU delay.");
    self->println("If no ticks appear until after the GPU finishes, "
                  "actor_facade is still blocking.\n");

    auto busy = self->spawn(actor_from_state<BusyActor>);
    self->mail(std::string("start")).send(busy);

    auto test_actor = self->spawn(actor_from_state<FacadeTestActor>);
    self->mail(3, busy).send(test_actor); // 3-second GPU delay

    self->await_all_other_actors_done();
}

CAF_MAIN(caf::cuda::manager)
