#pragma once
#include <string>
#include <vector>

struct EpochMetrics {
    int epoch = 0;
    float loss = 0.0f;
    float acc = 0.0f;
    float ms = 0.0f;
    int nnz = -1;
};

struct StageResult {
    std::string key;
    std::string name;
    std::vector<EpochMetrics> history;
    float total_ms = 0.0f;
    size_t opt_mem_bytes = 0;
};

StageResult run_adagrad_stage();
StageResult run_adam_stage();
StageResult run_sparse_adagrad_stage();
StageResult run_mixed_adam_stage();
