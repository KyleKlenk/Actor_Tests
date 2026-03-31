// Test 7: Framework Overhead Stress Test
//
// Measures the per-dispatch latency and throughput of the libcaf_cuda
// command_runner interface by repeatedly launching kernels and recording
// wall-clock time with std::chrono::high_resolution_clock.
//
// All passes run inside a single Benchmarker actor on one CAF thread.
// command_runner::run() is fully synchronous (H2D copy → kernel launch →
// D2H copy → stream sync), so each timed iteration captures the complete
// round-trip cost through the framework.
//
// Two benchmark passes:
//
//   Pass 1 — Trivial kernel (1-element copy, 1 GPU thread)
//     Isolates the framework overhead floor: the time you pay regardless
//     of how much GPU compute your real kernel performs.
//
//   Pass 2 — Matrix multiplication at N = {4, 32, 128, 512}
//     Shows where GPU compute + memory transfer time begin to dominate
//     over the fixed dispatch overhead, revealing the crossover point.
//
// Statistics per pass (N_BENCH = 200 iterations after N_WARMUP = 20 warmup):
//   mean, min, max, p50, p95, p99 latency (ms)  +  dispatch throughput (ops/s)

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "../common/kernel_paths.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>

static constexpr int N_WARMUP = 20;
static constexpr int N_BENCH  = 200;

// ---------------------------------------------------------------------------
// Timing statistics
// ---------------------------------------------------------------------------
struct Stats {
    double mean_ms;
    double min_ms;
    double max_ms;
    double p50_ms;
    double p95_ms;
    double p99_ms;
    double dispatch_per_sec;
};

static Stats compute_stats(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    const int n  = static_cast<int>(times.size());
    double total = std::accumulate(times.begin(), times.end(), 0.0);
    return {
        .mean_ms          = total / n,
        .min_ms           = times.front(),
        .max_ms           = times.back(),
        .p50_ms           = times[n / 2],
        .p95_ms           = times[static_cast<int>(n * 0.95)],
        .p99_ms           = times[static_cast<int>(n * 0.99)],
        .dispatch_per_sec = n / (total / 1000.0)
    };
}

static void print_stats(caf::event_based_actor* self,
                        const std::string& label,
                        const Stats& s,
                        int payload_bytes = 0) {
    self->println("--- {} ---", label);
    if (payload_bytes > 0)
        self->println("  Payload : {} bytes (H2D + D2H)", payload_bytes);
    self->println("  Mean    : {:.3f} ms", s.mean_ms);
    self->println("  Min     : {:.3f} ms", s.min_ms);
    self->println("  Max     : {:.3f} ms", s.max_ms);
    self->println("  p50     : {:.3f} ms", s.p50_ms);
    self->println("  p95     : {:.3f} ms", s.p95_ms);
    self->println("  p99     : {:.3f} ms", s.p99_ms);
    self->println("  Thruput : {:.0f} dispatches/sec\n", s.dispatch_per_sec);
}

// ---------------------------------------------------------------------------
// Benchmarker actor
//
// Triggered by a single integer message (unused value). Runs both benchmark
// passes sequentially, prints results, then quits.
// ---------------------------------------------------------------------------
class Benchmarker {
    caf::event_based_actor* self_;

public:
    explicit Benchmarker(caf::event_based_actor* self) : self_(self) {}

    caf::behavior make_behavior() {
        return {
            [this](int /*trigger*/) {
                auto& mgr  = self_->system().cuda_manager();
                const int aid = static_cast<int>(self_->id());

                self_->println(
                    "=== Test 7: Framework Overhead Stress Test ===\n"
                    "  Warmup iterations : {}\n"
                    "  Benchmark iterations: {}\n",
                    N_WARMUP, N_BENCH);

                // --------------------------------------------------------
                // Pass 1: trivial kernel — 1 thread, 1 read, 1 write
                // Measures the overhead floor of the framework.
                // --------------------------------------------------------
                {
                    using nop_cmd = caf::cuda::command_runner<in<int>, out<int>>;
                    nop_cmd runner;

                    auto prog = mgr.create_program_from_cubin(
                        actor_tests::paths::trivial_cubin, "trivial_kernel");
                    caf::cuda::nd_range dim(1, 1, 1, 1, 1, 1);

                    std::vector<int> inp = {42};
                    const int payload = static_cast<int>(
                        sizeof(int) +   // H2D: 1 int in
                        sizeof(int));   // D2H: 1 int out

                    // Warmup (amortises first-use CUDA context setup)
                    for (int i = 0; i < N_WARMUP; ++i)
                        runner.run(prog, dim, aid,
                                   caf::cuda::create_in_arg(inp),
                                   caf::cuda::create_out_arg_with_size<int>(1));

                    // Timed benchmark
                    std::vector<double> times(N_BENCH);
                    for (int i = 0; i < N_BENCH; ++i) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        runner.run(prog, dim, aid,
                                   caf::cuda::create_in_arg(inp),
                                   caf::cuda::create_out_arg_with_size<int>(1));
                        times[i] = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - t0)
                                       .count();
                    }

                    print_stats(self_,
                                "Pass 1 — trivial kernel (1-element copy, 1 GPU thread)",
                                compute_stats(times),
                                payload);
                }

                // --------------------------------------------------------
                // Pass 2: matrix multiplication at increasing sizes
                //
                // As N grows, H2D/D2H transfer and GPU compute increase.
                // The point where per-dispatch time noticeably exceeds
                // Pass 1 mean is where the compute/memory cost overtakes
                // the fixed framework overhead.
                // --------------------------------------------------------
                {
                    using mmul_cmd = caf::cuda::command_runner<
                        in<int>, in<int>, out<int>, in<int>>;
                    mmul_cmd runner;

                    auto prog = mgr.create_program_from_cubin(
                        actor_tests::paths::matmul_quiet_cubin, "matrixMul");

                    const int sizes[] = {4, 32, 128, 512};

                    for (int N : sizes) {
                        int THREADS = 16;
                        int BLOCKS  = (N + THREADS - 1) / THREADS;
                        caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1,
                                                THREADS, THREADS, 1);

                        // All-ones matrices — correct sum per element = N
                        std::vector<int> A(N * N, 1);
                        std::vector<int> B(N * N, 1);

                        const int payload = static_cast<int>(
                            sizeof(int) * N * N * 2 +  // H2D: A + B
                            sizeof(int) * N * N);       // D2H: C

                        // Warmup
                        for (int i = 0; i < N_WARMUP; ++i)
                            runner.run(prog, dim, aid,
                                       caf::cuda::create_in_arg(A),
                                       caf::cuda::create_in_arg(B),
                                       caf::cuda::create_out_arg_with_size<int>(N * N),
                                       caf::cuda::create_in_arg(N));

                        // Timed benchmark
                        std::vector<double> times(N_BENCH);
                        for (int i = 0; i < N_BENCH; ++i) {
                            auto t0 = std::chrono::high_resolution_clock::now();
                            runner.run(prog, dim, aid,
                                       caf::cuda::create_in_arg(A),
                                       caf::cuda::create_in_arg(B),
                                       caf::cuda::create_out_arg_with_size<int>(N * N),
                                       caf::cuda::create_in_arg(N));
                            times[i] = std::chrono::duration<double, std::milli>(
                                           std::chrono::high_resolution_clock::now() - t0)
                                           .count();
                        }

                        print_stats(self_,
                                    "Pass 2 — MatMul " + std::to_string(N) + "x" +
                                        std::to_string(N),
                                    compute_stats(times),
                                    payload);
                    }
                }

                self_->println("=== Test 7 complete ===");
                self_->quit();
            }
        };
    }
};

// ---------------------------------------------------------------------------
// caf_main
// ---------------------------------------------------------------------------
void caf_main(caf::actor_system& sys) {
    caf::scoped_actor self{sys};
    auto bench = sys.spawn(caf::actor_from_state<Benchmarker>);
    self->mail(0).send(bench);
    self->await_all_other_actors_done();
}

CAF_MAIN(caf::cuda::manager)
