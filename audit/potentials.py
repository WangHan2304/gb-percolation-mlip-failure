"""统一 uMLIP 计算器工厂：audit 全线共用。

新增势只需在 POTENTIALS 注册一个 factory。统一约定：
- 所有 calculator 接受 ASE Atoms
- 输出 ASE 标准 SinglePointCalculator 接口（get_potential_energy / get_stress）
- 应力约定：ASE eV/Å³，正值=压（ASE 惯例），换算见 utils
"""
from __future__ import annotations

import functools

GPA = 160.21766208  # 1 eV/Å³ = 160.21766208 GPa

# 文献/官方推荐的超胞建议（供 γ 线等扩展实验用）
POTENTIALS = {
    "mace-mp0-small": {
        "label": "MACE-MP-0 (small, PBE)",
        "family": "MACE",
        "training": "MPtrj (PBE)",
    },
    "mace-mp0-medium": {
        "label": "MACE-MP-0 (medium, PBE)",
        "family": "MACE",
        "training": "MPtrj (PBE)",
    },
    "chgnet": {
        "label": "CHGNet 0.3.0",
        "family": "CHGNet",
        "training": "MPtrj (PBE) + magmoms",
    },
}


@functools.lru_cache(maxsize=None)
def get_calculator(name: str, device: str = "cpu"):
    """按名称构造（并缓存）uMLIP 的 ASE calculator。"""
    if name.startswith("mace-mp0-"):
        from mace.calculators import mace_mp

        size = name.removeprefix("mace-mp0-")
        return mace_mp(model=size, default_dtype="float64", device=device)
    if name == "chgnet":
        try:
            from chgnet.model import CHGNetCalculator
        except ImportError:  # 旧版路径
            from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(use_device=device)
    raise KeyError(f"未知势：{name}；可选：{list(POTENTIALS)}")


def available_potentials() -> list[str]:
    """探测当前环境实际可用的势（导入成功即算可用）。"""
    ok = []
    for name in POTENTIALS:
        try:
            get_calculator(name)
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[potentials] {name} 不可用：{type(e).__name__}: {e}")
    return ok
