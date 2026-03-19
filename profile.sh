#!/bin/bash
# ============================================================
#  profile.sh — Профилирование всех этапов
#
#  Использование:
#    ./profile.sh          — полный набор профилей
#    ./profile.sh stage1   — только Adagrad
#    ./profile.sh timeline — Nsight Systems timeline для benchmark
#
#  Выход:
#    profiles/*.ncu-rep  — Nsight Compute (ядра)
#    profiles/*.nsys-rep — Nsight Systems (timeline)
# ============================================================

set -e

ARCH="${ARCH:-70}"
mkdir -p profiles

echo "╔══════════════════════════════════════════════════╗"
echo "║          CUDA Optimizer Profiling Suite          ║"
echo "║          Target architecture: sm_${ARCH}            ║"
echo "╚══════════════════════════════════════════════════╝"

# Собираем всё
make all benchmark ARCH=$ARCH

run_ncu() {
    local target=$1
    local label=$2
    echo ""
    echo "━━━ Profiling $label with Nsight Compute ━━━"

    # Метрики, которые нам важны:
    #   sm__throughput — загрузка SM
    #   dram__throughput — пропускная способность памяти
    #   l1tex__throughput — L1 cache throughput
    #   smsp__sass_average_data_bytes_per_sector_mem_global_op — coalescing efficiency
    ncu --set full \
        --target-processes all \
        -o "profiles/${label}" \
        ./$target 2>&1 | tail -5

    echo "  → profiles/${label}.ncu-rep"
}

run_nsys() {
    local target=$1
    local label=$2
    echo ""
    echo "━━━ Timeline profiling $label with Nsight Systems ━━━"

    nsys profile \
        --stats=true \
        --output="profiles/${label}" \
        ./$target 2>&1 | grep -E "(Time|CUDA|Kernel)" | head -20

    echo "  → profiles/${label}.nsys-rep"
}

case "${1:-all}" in
    stage1)
        run_ncu stage1_adagrad "stage1_adagrad"
        ;;
    stage2)
        run_ncu stage2_adam "stage2_adam"
        ;;
    stage3)
        run_ncu stage3_sparse_adagrad "stage3_sparse"
        ;;
    stage4)
        run_ncu stage4_mixed_adam "stage4_mixed"
        ;;
    timeline)
        run_nsys benchmark "benchmark_timeline"
        ;;
    all)
        run_ncu stage1_adagrad "stage1_adagrad"
        run_ncu stage2_adam "stage2_adam"
        run_ncu stage3_sparse_adagrad "stage3_sparse"
        run_ncu stage4_mixed_adam "stage4_mixed"
        run_nsys benchmark "benchmark_timeline"

        echo ""
        echo "╔══════════════════════════════════════════════════╗"
        echo "║  Все профили сохранены в profiles/               ║"
        echo "║                                                  ║"
        echo "║  Просмотр:                                       ║"
        echo "║    ncu-ui profiles/stage1_adagrad.ncu-rep        ║"
        echo "║    nsys-ui profiles/benchmark_timeline.nsys-rep  ║"
        echo "║                                                  ║"
        echo "║  Ключевые метрики для сравнения:                 ║"
        echo "║    • Memory throughput (DRAM + L1)               ║"
        echo "║    • Occupancy                                   ║"
        echo "║    • Warp stall reasons                          ║"
        echo "║    • Coalescing efficiency (global load/store)   ║"
        echo "╚══════════════════════════════════════════════════╝"
        ;;
    *)
        echo "Использование: $0 [stage1|stage2|stage3|stage4|timeline|all]"
        exit 1
        ;;
esac
