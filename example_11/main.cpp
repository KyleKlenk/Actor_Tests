// example_11: Two actors - one does GPU work via run_async(), one does CPU busy work.
// With 1 worker thread, the question is: does the BusyActor also get blocked
// while the GPU kernel is running inside the CAF control layer?
//
// Expected finding: YES — run_async() returns to the calling actor immediately,
// but the CAF control-layer actor that actually launches and waits on the CUDA
// stream blocks the single worker thread.  BusyActor ticks will therefore only
// appear AFTER the GPU kernel finishes, demonstrating that the blocking is
// scheduler-wide, not just local to the actor that called run_async().

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <iostream>
#include <chrono>

using namespace caf;

using delay_command = caf::cuda::command_runner<
    in<int>,  // delay_seconds
    out<int>  // output
>;

// ─────────────────────────────────────────────────────────────────────────────
// BusyActor: a completely independent actor that does nothing but print ticks.
// It has no knowledge of GPU work; it is purely a canary for scheduler health.
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
                    self_->println("[BusyActor] Started, beginning tick loop (every 200 ms)...");
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
// AsyncTestActor: fires GPU work via run_async(), then pings itself while
// waiting for the GPU result.  Holds a handle to BusyActor so it can shut it
// down when done.
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
                self_->println("[AsyncTestActor] Spawned, initiating GPU work (run_async())...");

                // Schedule a ping loop so we can see if THIS actor is also blocked.
                self_->mail(std::string("ping"))
                    .delay(std::chrono::milliseconds(200))
                    .send(self_);

                auto& mgr = self_->system().cuda_manager();
                auto program = mgr.create_program_from_cubin("delay_11.cubin", "delayKernel");

                caf::cuda::nd_range dim(1, 1, 1, 1, 1, 1);
                auto arg1 = caf::cuda::create_in_arg(delay_seconds);
                auto arg2 = caf::cuda::create_out_arg_with_size<int>(1);

                delay_command runner;
                auto result_tuple = runner.run_async(program, dim, self_->id(), arg1, arg2);
                self_->println("[AsyncTestActor] run_async() returned immediately.");

                result_ptr_ = std::get<1>(result_tuple);

                // Fetch the result well after the kernel should have finished.
                self_->mail(std::string("fetch_result"))
                    .delay(std::chrono::seconds(5))
                    .send(self_);
            },
            [this](const std::string& msg) {
                if (msg == "ping") {
                    ping_count_++;
                    self_->println("[AsyncTestActor] CPU ping #{}", ping_count_);
                    self_->mail(std::string("ping"))
                        .delay(std::chrono::milliseconds(200))
                        .send(self_);
                } else if (msg == "fetch_result") {
                    self_->println("[AsyncTestActor] Fetching result from GPU memory...");
                    std::vector<int> output = result_ptr_->copy_to_host();
                    self_->println(
                        "[AsyncTestActor] GPU result={}. CPU got to ping #{}.",
                        output[0], ping_count_);
                    // Shut down the busy actor so the system can exit cleanly.
                    self_->mail(std::string("stop")).send(busy_actor_);
                    self_->quit();
                }
            }
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Config: deliberately keep to 1 worker thread so we can observe the
// scheduler-level blocking caused by the CUDA control-layer actor.
// ─────────────────────────────────────────────────────────────────────────────
class config : public actor_system_config {
public:
    config() {
        set("caf.scheduler.max-threads", 1u);
    }
};

void caf_main(actor_system& sys, const config& cfg) {
    scoped_actor self{sys};
    self->println("Hello, CAF Async GPU Test (example 11)!");
    self->println("Scheduler max-threads config is: {}",
                  get_or(sys.config(), "caf.scheduler.max-threads", 0));
    self->println("Two actors: [AsyncTestActor] (GPU work) + [BusyActor] (CPU ticks).");
    self->println("Hypothesis: with 1 worker thread, BOTH actors are blocked during the GPU kernel.");
    self->println("");

    // Spawn BusyActor first and kick off its tick loop.
    auto busy = self->spawn(actor_from_state<BusyActor>);
    self->mail(std::string("start")).send(busy);

    // Spawn the GPU actor, passing the BusyActor handle so it can stop it later.
    auto test_actor = self->spawn(actor_from_state<AsyncTestActor>);
    self->mail(3, busy).send(test_actor);

    self->await_all_other_actors_done();
}

CAF_MAIN(caf::cuda::manager)
