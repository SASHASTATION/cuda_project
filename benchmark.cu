#include "stages.h"
#include "config.h"
#include <cstdio>
#include <vector>

static void write_convergence_csv(const std::vector<StageResult>& results) {
    FILE* f = fopen("convergence.csv", "w");
    if (!f) return;

    fprintf(f, "stage,epoch,loss,accuracy,time_ms,nnz\n");
    for (const auto& r : results) {
        for (const auto& row : r.history) {
            fprintf(f, "%s,%d,%.6f,%.6f,%.4f,%d\n",
                    r.key.c_str(), row.epoch, row.loss, row.acc, row.ms, row.nnz);
        }
    }
    fclose(f);
}

static void print_summary(const std::vector<StageResult>& results) {
    printf("\n");
    printf("╔══════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╗\n");
    printf("║    Optimizer     ║  Loss    ║ Accuracy ║ Total ms ║  Avg ms  ║ Opt Mem  ║\n");
    printf("╠══════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╣\n");

    for (const auto& r : results) {
        const auto& last = r.history.back();
        printf("║ %-16s ║ %8.5f ║  %5.2f%%  ║ %7.1f  ║ %7.2f  ║ %5.1f KB ║\n",
               r.name.c_str(), last.loss, last.acc * 100.0f,
               r.total_ms, r.total_ms / N_EPOCHS, r.opt_mem_bytes / 1024.0f);
    }

    printf("╚══════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╝\n");
}

int main() {
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║    CUDA Optimizer Benchmark — Unified Comparison    ║\n");
    printf("║    %d samples × %d features × %d epochs       ║\n", N_SAMPLES, N_FEATURES, N_EPOCHS);
    printf("╚══════════════════════════════════════════════════════╝\n");

    std::vector<StageResult> results;
    results.push_back(run_adagrad_stage());
    results.push_back(run_adam_stage());
    results.push_back(run_sparse_adagrad_stage());
    results.push_back(run_mixed_adam_stage());

    write_convergence_csv(results);
    print_summary(results);

    printf("\n  Convergence data -> convergence.csv\n\n");
    return 0;
}
