#!/bin/bash
# E3 批量求解：manifest.csv 中每个 run → docker DAMASK_grid → 曲线提取 → 删 hdf5
# 用法：bash e3_run.sh [并发数]   （默认 4；每容器 OMP_NUM_THREADS=并发分到的线程数）
set -u
ROOT=/home/cl/20241015论文
PY=/home/cl/20261016MLIP传播论文/.venv/bin/python
IMG=cr.registry.res.cloud.zhejianglab.com/cldata/damask-sim:paper
NPROC=${1:-4}
RUNS=$ROOT/propagation/e3_runs
CURVES=$ROOT/labels/e3_curves
mkdir -p "$CURVES"

run_one() {
  local wd=$1 ranks=$2
  local dir="$ROOT/$wd"
  local tag=$(basename "$wd")
  if [ -f "$CURVES/${tag}.npz" ]; then echo "[skip] $tag"; return 0; fi
  if [ ! -f "$dir/load.yaml" ]; then echo "[missing] $dir"; return 1; fi
  local t0=$(date +%s)
  docker run --rm -v "$dir":/work \
    --entrypoint /bin/bash "$IMG" -c \
    "source /home/cl/CODE/DAMASK/env/DAMASK.sh && export OMP_NUM_THREADS=1 && mpiexec -n $ranks DAMASK_grid --geom grid.vti --load load.yaml --material material.yaml --workingdirectory /work" \
    > "$dir/solver.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[FAIL rc=$rc] $tag（详见 solver.log）"; return 1; fi
  # 求解产物名 = <geom>_<load>_<material>.hdf5
  local h5=$(ls "$dir"/*_material.hdf5 2>/dev/null | head -1)
  if [ -z "$h5" ]; then echo "[NO-H5] $tag"; return 1; fi
  mv "$h5" "$dir/result.hdf5"
  if ! "$PY" "$ROOT/propagation/e3_postprocess.py" --stage curves --run-dir "$dir" --z "$ZLAYERS"; then
    echo "[POST-FAIL] $tag"; return 1
  fi
  rm -f "$dir/result.hdf5" "$dir"/*.sta
  echo "[done $(( $(date +%s) - t0 ))s] $tag"
}
export -f run_one
export ROOT PY IMG CURVES

# manifest 的 workdir 列（相对 D6）；每 run 的 MPI ranks = min(16/并发数, 8, z 层数)
ZLAYERS=${E3_Z:-4}
export ZLAYERS   # 必须导出：run_one 在 xargs 子 shell 中执行，未导出时 $ZLAYERS 为空串
RANKS=$(( 16 / NPROC > ZLAYERS ? ZLAYERS : 16 / NPROC ))
echo "[config] NPROC=$NPROC RANKS=$RANKS (z=$ZLAYERS)"
tail -n +2 "$RUNS/manifest.csv" | cut -d, -f3 | xargs -P "$NPROC" -I{} bash -c 'run_one {} '"$RANKS"
echo "=== E3 批量结束 $(date) ==="
