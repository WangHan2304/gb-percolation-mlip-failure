#!/bin/bash
# E1 第二轮重试：Cu 310 / Ni 310 / Fe 210（构造族 dmin 上限 1.14/1.11/1.28 Å < 1.3，降至 1.05）
cd /home/cl/20241015论文 || exit 1
V=/home/cl/20261016MLIP传播论文/.venv/bin/python
export OMP_NUM_THREADS=2

nohup "$V" -u audit/gb_energy_audit.py --potentials chgnet mace-mp0-small mace-mp0-medium --metals Cu --planes 3 1 0 --min-dist 1.05 --out results/gb_retry2_cu310.csv > logs/gb_retry2_cu310.log 2>&1 &
echo "Cu310 x3pot -> $!"

nohup "$V" -u audit/gb_energy_audit.py --potentials chgnet mace-mp0-small mace-mp0-medium --metals Ni --planes 3 1 0 --min-dist 1.05 --out results/gb_retry2_ni310.csv > logs/gb_retry2_ni310.log 2>&1 &
echo "Ni310 x3pot -> $!"

nohup "$V" -u audit/gb_energy_audit.py --potentials chgnet mace-mp0-small mace-mp0-medium --metals Fe --planes 2 1 0 --min-dist 1.05 --out results/gb_retry2_fe210.csv > logs/gb_retry2_fe210.log 2>&1 &
echo "Fe210 x3pot -> $!"

echo "--- 5 秒后自检 ---"
sleep 5
ps aux | grep 'gb_energy_audit[.]py' | grep -v grep
