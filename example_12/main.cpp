// example_12: Actor-native GPU completion via cuLaunchHostFunc.
//
// Demonstrates run_async_notify(): instead of the actor polling with a
// fixed-delay timer, the CUDA runtime calls a host-function callback when
// the kernel stream becomes idle.  The callback sends gpu_done_atom to this
// actor from CUDA's internal thread — no CAF worker thread is ever blocked.
//
// Compare with example_10 (run_async + fixed 4-second timer).  Here the
// actor wakes immediately when the GPU is ready, and there is no timer logic
// at all in the application code.

#include <caf/all.hpp>
#include <caf/actor_cast.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <iostream>
#include <chrono>

using namespace caf;

using delay_command = caf::cuda::command_runner<
    in<int>, // delay_seconds
    out<int> // output
>;

// ─────────────────────────────────────────────────────────────────────────────
// BusyActor: independent actor that only prints ticks.
// Used to confirm the scheduler keeps running during GPU work.
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
                    self_->println("[BusyActor] Stopping at tick #{}", tick_count_);
                    self_->quit();
                }
            }
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// AsyncTestActor: uses run_async_notify() — no polling timer.
// The actor handler for gpu_done_atom fires the instant the GPU is done.
// ─────────────────────────────────────────────────────────────────────────────
class AsyncTestActor {
    event_based_actor* self_;
    int ping_count_ = 0;
    actor busy_actor_;
    caf::cuda::mem_ptr<int> result_ptr_ = nullptr;

public:
    AsyncTestActor(event_based_actor* self) : self_(self) {}

    behavior make_behavior() {
        return {
            [this](int delay_seconds, actor busy) {
                busy_actor_ = busy;
                self_->println("[AsyncTestActor] Spawned. Launching GPU kernel via run_async_notify()...");

                // CPU ping loop — should fire continuously throughout GPU work.
                self_->mail(std::string("ping"))
                    .delay(std::chrono::milliseconds(200))
                    .send(self_);

                auto& mgr = self_->system().cuda_manager();
                auto program = mgr.create_program_from_cubin("delay_12.cubin", "delayKernel");

                caf::cuda::nd_range dim(1, 1, 1, 1, 1, 1);
                auto arg1 = caf::cuda::create_in_arg(delay_seconds);
                auto arg2 = caf::cuda::create_out_arg_with_size<int>(1);

                delay_command runner;

                // run_async_notify: kernel is launched async; when the stream
                // becomes idle the CUDA runtime sends gpu_done_atom to this
                // actor from its own internal thread.  No fixed timer needed.
                auto result_tuple = runner.run_async_notify(
                    program,
                    dim,
                    caf::actor_cast<caf::actor>(self_),
                    arg1, arg2);

                result_ptr_ = std::get<1>(result_tuple);
                self_->println("[AsyncTestActor] run_async_notify() returned. "
                               "Waiting for gpu_done_atom...");
            },

            // ── GPU completion callback (actor-native) ────────────────────
            // Fired immediately when the kernel stream goes idle.
            // No timer, no fixed delay — the GPU drives the actor forward.
            [this](caf::cuda::gpu_done_atom) {
                self_->println("[AsyncTestActor] gpu_done_atom received! "
                               "Stream is idle — fetching result.");
                std::vector<int> output = result_ptr_->copy_to_host();
                self_->println(
                    "[AsyncTestActor] GPU result={}. CPU got to ping #{} "
                    "while GPU was running.",
                    output[0], ping_count_);
                self_->mail(std::string("stop")).send(busy_actor_);
                self_->quit();
            },

            [this](const std::string& msg) {
                if (msg == "ping") {
                    ping_count_++;
                    self_->println("[AsyncTestActor] CPU ping #{}", ping_count_);
                    self_->mail(std::string("ping"))
                        .delay(std::chrono::milliseconds(200))
                        .send(self_);
                }
            }
        };
    }
};

class config : public actor_system_config {
public:
    config() {
        set("caf.scheduler.max-threads", 1u);
    }
};

void caf_main(actor_system& sys, const config& cfg) {
    scoped_actor self{sys};
    self->println("CAF-CUDA actor-native GPU completion (example 12).");
    self->println("Scheduler max-threads: {}",
                  get_or(sys.config(), "caf.scheduler.max-threads", 0));
    self->println("Key difference vs example 10/11: NO polling timer.");
    self->println("The GPU wakes the actor the moment its stream goes idle.\n");

    auto busy = self->spawn(actor_from_state<BusyActor>);
    self->mail(std::string("start")).send(busy);

    auto test_actor = self->spawn(actor_from_state<AsyncTestActor>);
    self->mail(3, busy).send(test_actor); // 3-second GPU delay

    self->await_all_other_actors_done();
}

CAF_MAIN(caf::cuda::manager)
