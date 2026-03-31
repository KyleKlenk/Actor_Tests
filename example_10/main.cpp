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
    
    // Store the memory pointer so we can fetch it asynchronously later
    caf::cuda::mem_ptr<int> result_ptr_ = nullptr;

public:
    AsyncTestActor(event_based_actor* self) : self_(self) {};

    behavior make_behavior() {
        return {
            [this](int delay_seconds) {
                self_->println("AsyncTestActor spawned, initiating GPU work (using run_async()...)");

                // Start a ping loop to show CPU is free
                self_->mail(std::string("ping")).delay(std::chrono::milliseconds(200)).send(self_);

                auto& mgr = self_->system().cuda_manager();
                auto program = mgr.create_program_from_cubin(
                    actor_tests::paths::delay_cubin, "delayKernel");

                caf::cuda::nd_range dim(1, 1, 1, 1, 1, 1);

                auto arg1 = caf::cuda::create_in_arg(delay_seconds);
                auto arg2 = caf::cuda::create_out_arg_with_size<int>(1);

                delay_command runner;

                // .run_async() does not block the actor's CPU thread, just returning the mem pointers immediately!
                auto result_tuple = runner.run_async(program, dim, self_->id(), arg1, arg2);
                self_->println("Background tasks started. run_async() returned immediately.");
                
                // Save the result ptr
                result_ptr_ = std::get<1>(result_tuple);
                
                // Ask ourselves to check the result after 4 seconds (after the 3-second GPU kernel definitely finished)
                self_->mail(std::string("fetch_result")).delay(std::chrono::seconds(4)).send(self_);
            },
            [this](const std::string& msg) {
                if (msg == "ping") {
                    ping_count_++;
                    self_->println("CPU doing work... (ping #{})", ping_count_);
                    self_->mail(std::string("ping")).delay(std::chrono::milliseconds(200)).send(self_);
                } else if (msg == "fetch_result") {
                    self_->println("Attempting to fetch the result from GPU memory...");
                    std::vector<int> output = result_ptr_->copy_to_host();
                    self_->println("Received result from GPU! Out={}. CPU got to ping #{}", output[0], ping_count_);
                    self_->quit();
                }
            }
        };
    }
};

class config : public actor_system_config {
public:
  config() {
      // With run_async(), even with a single thread we should see the CPU pinging!
      // But let's keep it to 2 as the user had in example_8
      set("caf.scheduler.max-threads", 1u);
  }
};

void caf_main(actor_system& sys, const config& cfg) {
    scoped_actor self{sys};
    self->println("Hello, CAF Async GPU Test (example 10)!");
    self->println("Scheduler max-threads config is: {}", get_or(sys.config(), "caf.scheduler.max-threads", 0));
    
    auto test_actor = self->spawn(actor_from_state<AsyncTestActor>);
    self->mail(3).send(test_actor); // 3 seconds delay
    
    self->await_all_other_actors_done();
}

CAF_MAIN(caf::cuda::manager)
