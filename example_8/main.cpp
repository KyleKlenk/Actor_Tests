#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <iostream>
#include <chrono>

using namespace caf;




class AsyncTestActor {
    event_based_actor* self_;
    int ping_count_ = 0;

public:
    AsyncTestActor(event_based_actor* self) : self_(self) {};

    behavior make_behavior() {
        return {
            [this](int delay_seconds) {
                self_->println("AsyncTestActor spawned, initiating GPU work...");

                caf::cuda::nd_range dim(1, 1, 1, 1, 1, 1);

                auto gpuActor = self_->system().cuda_manager().spawnFromCUBIN(
                    "delay_8.cubin", "delayKernel", dim,
                    in<int>{}, out<int>{});

                self_->mail(caf::cuda::create_in_arg(delay_seconds), 
                            caf::cuda::create_out_arg_with_size<int>(1))
                    .send(gpuActor);
                
                // Start a ping loop to show CPU is free
                self_->mail(std::string("ping")).delay(std::chrono::milliseconds(200)).send(self_);
                self_->println("Background tasks started.");
            },
            [this](const std::string& msg) {
                if (msg == "ping") {
                    ping_count_++;
                    self_->println("CPU doing work... (ping #{})", ping_count_);
                    self_->mail(std::string("ping")).delay(std::chrono::milliseconds(200)).send(self_);
                }
            },
            [this](const std::vector<output_buffer>& result) {
                std::vector<int> output = caf::cuda::extract_vector<int>(result);
                self_->println("Received result from GPU! Out={}. CPU got to ping #{}", output[0], ping_count_);
                self_->quit();
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
    self->println("Hello, CAF Async GPU Test!");
    self->println("Scheduler max-threads config is: {}", get_or(sys.config(), "caf.scheduler.max-threads", 0));
    
    auto test_actor = self->spawn(actor_from_state<AsyncTestActor>);
    self->mail(3).send(test_actor); // 3 seconds delay
    
    self->await_all_other_actors_done();
}

CAF_MAIN(caf::cuda::manager)
