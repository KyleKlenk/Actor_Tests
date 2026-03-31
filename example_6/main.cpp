// Test 6: Multi-GPU Parallel Matrix Multiplication
//
// Spawns one GpuWorker actor per detected GPU device. Each independently
// computes a 4×4 matrix product on its assigned device using the
// command_runner device_number overload.
//
// Key API exercised:
//   command_runner::run(program, dim, actor_id, shared_mem, device_number, ...)
//
// The program_ptr loads kernels on ALL devices at construction time inside
// program::load_kernels(), so a single create_program_from_cubin() call
// produces a handle that is valid for any device_number.
//
// With ≥2 GPUs the jobs run concurrently on separate hardware.
// With 1 GPU both jobs target device 0, exercising the same API path.
//
// Input matrices (4×4, row-major):
//   A = {1..16},  B = {16..1}
// Expected result: C[0][0] = 1*16 + 2*12 + 3*8 + 4*4 = 80

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "../common/kernel_paths.hpp"
#include <vector>
#include <chrono>

using mmul_command = caf::cuda::command_runner<
    in<int>,   // A  (read-only)
    in<int>,   // B  (read-only)
    out<int>,  // C  (write-only, output)
    in<int>    // N  (matrix dimension, scalar)
>;

// ===========================================================================
// GpuWorker
//
// Accepts a single integer message (N = matrix dimension), dispatches the
// kernel to its assigned device, prints the first row of the result, then
// quits.
// ===========================================================================
class GpuWorker {
    caf::event_based_actor* self_;
    int device_id_;

    // 4×4 input matrices (row-major)
    std::vector<int> A_ = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<int> B_ = {
        16, 15, 14, 13,
        12, 11, 10,  9,
         8,  7,  6,  5,
         4,  3,  2,  1
    };

public:
    GpuWorker(caf::event_based_actor* self, int device_id)
        : self_(self), device_id_(device_id) {}

    caf::behavior make_behavior() {
        return {
            [this](int N) {
                auto& mgr = self_->system().cuda_manager();

                // program loads kernels on all devices: one program_ptr
                // works for any device_number argument below.
                auto program = mgr.create_program_from_cubin(
                    actor_tests::paths::matmul_quiet_cubin, "matrixMul");

                int THREADS = 16;
                int BLOCKS  = (N + THREADS - 1) / THREADS;
                caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1,
                                        THREADS, THREADS, 1);

                mmul_command runner;
                auto result_buf = runner.run(
                    program, dim,
                    static_cast<int>(self_->id()),  // stream affinity key
                    0,                               // shared memory bytes
                    device_id_,                      // ← explicit target GPU
                    caf::cuda::create_in_arg(A_),
                    caf::cuda::create_in_arg(B_),
                    caf::cuda::create_out_arg_with_size<int>(N * N),
                    caf::cuda::create_in_arg(N)
                );

                auto out = caf::cuda::extract_vector<int>(result_buf);
                self_->println(
                    "[GPU {}]  first row: [{}, {}, {}, {}]  "
                    " (C[0][0]={}, expected 80)",
                    device_id_,
                    out[0], out[1], out[2], out[3],
                    out[0]);

                self_->quit();
            }
        };
    }
};

// ===========================================================================
// caf_main
// ===========================================================================
void caf_main(caf::actor_system& sys) {
    auto& mgr = sys.cuda_manager();
    const int num_devices = mgr.get_num_devices();

    caf::scoped_actor self{sys};
    self->println("=== Test 6: Multi-GPU Parallel Matrix Multiplication ===");
    self->println("Detected {} GPU device(s)", num_devices);

    if (num_devices == 1)
        self->println(
            "Note: single-GPU machine — both jobs target device 0.\n"
            "      On a multi-GPU machine each job would run on a "
            "separate device concurrently.");

    // Always launch at least 2 jobs so the multi-device API path is exercised
    // regardless of hardware configuration.
    const int N        = 4;
    const int num_jobs = std::max(num_devices, 2);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_jobs; ++i) {
        int target_device = i % num_devices;
        auto worker = sys.spawn(caf::actor_from_state<GpuWorker>, target_device);
        self->mail(N).send(worker);
    }

    // Block until every spawned actor has called quit().
    self->await_all_other_actors_done();

    double elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0)
                            .count();

    self->println("\n{} job(s) completed in {:.1f} ms", num_jobs, elapsed_ms);
    self->println("=== Test 6 complete ===");
}

CAF_MAIN(caf::cuda::manager)
