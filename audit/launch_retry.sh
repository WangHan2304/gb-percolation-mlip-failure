#!/bin/bash
# E1 失败晶面重试启动脚本 —— 全部字面量路径，规避 nohup sh -c 变量展开问题
# 待补缺口：
#   chgnet Ni 310/221/113; small+medium Cu 310/221, Ni 310/221/113; Fe 三势 210/310/130
cd /home/cl/20241015论文 || exit 1
mkdir -p logs
V=/home/cl/20261016MLIP传播论文/.venv/bin/python
export OMP_NUM_THREADS=2

nohup "$V" -u audit/gb_energy_audit.py --potentials chgnet --metals Ni --planes 3 1 0 2 2 1 1 1 3 --min-dist 1.3 --out results/gb_retry_chgnet_ni.csv > logs/gb_retry_chgnet_ni.log 2>&1 &
echo "chgnet Ni -> $!"

nohup "$V" -u audit/gb_energy_audit.py --potentials mace-mp0-small --metals Cu --planes 3 1 0 2 2 1 --min-dist 1.3 --out results/gb_retry_small_cu.csv > logs/gb_retry_small_cu.log 2>&1 &
echo "small Cu -> $!"

nohup "$V" -u audit/gb_energy_audit.py --potentials mace-mp0-small --metals Ni --planes 3 1 0 2 2 1 1 1 3 --min-dist 1.3 --out results/gb_retry_small_ni.csv > logs/gb_retry_small_ni.log 2>&1 &
echo "small Ni -> $!"

nohup "$V" -u audit/gb_energy_audit.py --potentials mace-mp0-medium --metals Cu --planes 3 1 0 2 2 1 --min-dist 1.3 --out results/gb_retry_medium_cu.csv > logs/gb_retry_medium_cu.log 2>&1 &
echo "medium Cu -> $!"

nohup "$V" -u audit/gb_energy_audit.py --potentials mace-mp0-medium --metals Ni --planes 3 1 0 2 2 1 1 1 3 --min-dist 1.3 --out results/gb_retry_medium_ni.csv > logs/gb_retry_medium_ni.log 2>&1 &
echo "medium Ni -> $!"

nohup "$V" -u audit/gb_energy_audit.py --potentials chgnet mace-mp0-small mace-mp0-medium --metals Fe --planes 2 1 0 3 1 0 1 3 0 --min-dist 1.3 --out results/gb_retry_fe.csv > logs/gb_retry_fe.log 2>&1 &
echo "Fe x3pot -> $!"

echo "--- 5 秒后自检 ---"
sleep 5
ps aux | grep 'gb_energy_audit[.]py' | grep -v grep
