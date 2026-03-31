#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "../common/kernel_paths.hpp"
#include <vector>
#include <iostream>
#include <chrono>

using namespace caf;

using delay_command = caf::cuda::command_runner<
    in<int>, // delay seconds
    out<int> // output
>;

class AsyncTestActor {
    event_based_actor* self_;
    int ping_count_ = 0;

public:
    AsyncTestActor(event_based_actor* self) : self_(self) {};

    behavior make_behavior() {
        return {
            [this](int delay_seconds) {
                self_->println("AsyncTestActor spawned, initiating GPU work (using run()...)");

                // Start a ping loop to show CPU is free (or in this case, blocked)
                self_->mail(std::string("ping")).delay(std::chrono::milliseconds(200)).send(self_);
                self_->println("Background tasks started. Now calling blocking run()...");

                auto& mgr = self_->system().cuda_manager();
                auto program = mgr.create_program_from_cubin(
                    actor_tests::paths::delay_cubin, "delayKernel");

                caf::cuda::nd_range dim(1, 1, 1, 1, 1, 1);

                auto arg1 = caf::cuda::create_in_arg(delay_seconds);
                auto arg2 = caf::cuda::create_out_arg_with_size<int>(1);

                delay_command runner;

                // .run() is expected to block the actor's execution until the GPU finishes
                auto result_buffer = runner.run(program, dim, self_->id(), arg1, arg2);

                std::vector<int> output = caf::cuda::extract_vector<int>(result_buffer);
                self_->println("Received result from GPU! Out={}. CPU got to ping #{}", output[0], ping_count_);
                self_->quit();
            },
            [this](const std::string& msg) {
                if (msg == "ping") {
                    ping_count_++;
                    self_->println("CPU doing work... (ping #{})", ping_count_);
                    self_->mail(std::string("ping")).delay(std::chrono::milliseconds(200)).send(self_);
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
    self->println("Hello, CAF Sync GPU Test (example 9)!");
    self->println("Scheduler max-threads config is: {}", get_or(sys.config(), "caf.scheduler.max-threads", 0));
    
    auto test_actor = self->spawn(actor_from_state<AsyncTestActor>);
    self->mail(3).send(test_actor); // 3 seconds delay
    
    self->await_all_other_actors_done();
}

CAF_MAIN(caf::cuda::manager)
