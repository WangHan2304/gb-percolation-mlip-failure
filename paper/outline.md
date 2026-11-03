# D6 论文大纲与图表规划

生成：2026-10-23 ｜ 基于 E1–E5 全部验收产物 ｜ 目标期刊：npj Computational Materials（主投；Acta Materialia 备选、Computational Materials Science 保底）

---

## 一句话核心主张

晶间失效抗力不只由 GB 能决定，而由 **GB 能谱 × 晶界网络拓扑** 共同决定——用 uMLIP 廉价 GB 能谱 + 组织网络渗流特征训练 ML，可在未见组织上盲测预测沿晶失效敏感度（RQ1/RQ2），且原子级能量误差沿"能谱→网络特征→预测"链路被**吸收而非放大**（RQ3）。

## 标题候选

1. *Grain-boundary network percolation meets machine-learned grain-boundary energies: blind prediction of intergranular failure susceptibility with quantified uncertainty*
2. *Topological or energetic? Disentangling network and grain-boundary energy contributions to intergranular failure via blind machine-learning tests*
3. *Predicting intergranular failure from grain-boundary energy spectra and network percolation: an MLIP–CPFEM–machine-learning pipeline with error-propagation audit*

## 章节结构与内容来源

### 1. Introduction
- GBE/渗流实验经典叙事（316L, Materials 2020）；实验有能谱无预测模型
- 邻居定位（§三 空白核验）：FIP-GNN（晶粒级疲劳局参、非渗流）、uMLIP 327 GB 评测（止于能量精度）→ **三要素（GB 能×网络拓扑→失效标签→ML 盲测）无占位者**
- RQ1（拓扑 vs 能量贡献几何）、RQ2（盲测泛化+适用域）、RQ3（原子级误差传播）
- 贡献清单（4 bullets：能谱库、渗流特征+CPFEM 标签、消融+conformal 盲测、误差传播审计）

### 2. Methods
- **2.1 GB 能谱审计（E1）**：3 金属（Cu/Ni/Fe）× 3 势（chgnet/mace-medium/mace-small），17 个金属–GB 族组合 × 3 势 = 51 格（50 有效 + 1 塌缩剔除）；γ 面 slab 协议；dmin/自愈验收；孪晶层与高角层误差分层
- **2.2 组织库与网络渗流特征（E2）**：2D 周期层级 Voronoi，200 样本；孪晶生成规则；4-邻域边表；γ 引导边能量采样（E1 池、rng 可重放）；winding-DSU 渗流特征（γ_perc、失效簇 AUC）
- **2.3 沿晶失效标签（E3）**：DAMASK 20240629 FFT，64²×z2 降采样 RVE，N=40（ε_max=0.06，21 输出点）；损伤当量模型 v2：σ_c=k·(γ+γ₀)^0.65（γ₀ 文献孪晶锚点吸收负值、κ=0.7 转变区标定、m=2），y-winding 贯通判据；两阶段标签（贯通二分类 + eps_fail）；v1 失败教训一段（σ_c∝γ 直通能谱→首步堆点 57%）
- **2.4 ML 盲测协议（E4/E4b）**：group=组织防泄漏 24/8/8；消融 T/G/TG × RF/线性；split conformal（α=0.1）；适用域 5-NN 距离 τ=cal p95；**2026-10-27 扩容**：模型冻结后新增 12 组织仅进测试集（train/cal 不动、固定 k 不重标定）→ 盲测 20 组织/180 行（回归 100 非删失行），全部盲测指标报**组聚类 bootstrap 95% CI**（B=2000，按组织有放回、组内行整体进出）
- **2.5 误差传播协议（E5）**：误差源=E1 势间池差异（零人工噪声）；扰动比例 p；E2 rng 精确重放；传播增益 G 定义（|Δpred|/pred ÷ |Δγ|/γ）

### 3. Results
- **3.1 能谱的两层误差结构（E1）**：孪晶层符号可翻转（chgnet Cu Σ3=−14.4、small Ni −0.4）vs 高角层 2–3× 离散；Fe Σ3(112) 三势彼此收敛（1.6× 全库最紧）但共同低于 ab initio 窗 340–510——构造层系统偏移，非"命中 DFT"（2026-10-24 锚点核验更正）
- **3.2 网络特征与渗流签名（E2）**：mean_degree≈6 自检；GBE 信号 corr(twin_frac, γ_perc_rel)=+0.17（pooled 1800 行；样本均值聚合 +0.46；分金属 Cu 0.23/Ni 0.25/Fe −0.05）；同组织跨势 γ_perc 分离中位 1.5×（最大 3.4×）
- **3.3 失效标签统计（E3）**：贯通率 43%（155/360）、首步堆点 0%、eps_fail 0.009–0.06 中位 0.036、分金属 Cu 64/Ni 56/Fe 35（删失率 47/53/71%）、贯通时 failed_frac 中位 0.128（部分失效贯通）
- **3.4 消融与盲测（E4/E4b，答 RQ1/RQ2）**：CV AUC T 0.799→TG 0.829、CV R² T 0.247→G 0.403→TG 0.424；**主报 20 组织盲测（180 行）**：AUC TG-RF 0.799 [0.658,0.911]、TG-Lin 0.825 [0.712,0.918]（各区间均离 0.5）；回归 R² 0.26–0.38（G-RF 0.377 [0.086,0.589] 唯一稳超 0，TG-RF 0.299 下沿触 0）、MAE 0.0084–0.0092；原 8 组织点估计 0.851/0.078 分别偏乐观/偏悲观且均落在 20 组 CI 内，forest-vs-lin 差异在 CI 内打平；ext12 单独 AUC 0.770、R² 0.44（纯外推）（此前记档"盲测 R² 0.42"实为 CV 值，以 prediction_audit.csv 对账为准）
- **3.5 UQ 与适用域**：conformal 覆盖 0.86–0.99（名义 0.90；TG-RF 0.86 [0.738,0.959]、TG-Lin 0.99 保守）；OOD 6/180 行（原8/新增12 各 3）
- **3.6 误差传播（E5，答 RQ3）**：p=1 时 γ 特征漂移 15–24%（拓扑恒 0）→ 分类翻转 7.0%（< 标签锚 10.8%）、回归 |Δeps| 0.0030、增益 G=0.39–0.76<1（吸收）；T 通道结构性零传播
- **3.7 标签稳健性（E6，2026-10-27 新增）**：同一 360 条应力史换律重标（m=1/p=0.5/p=1.0，逐金属增益匹配贯通数）→ verdict 一致 97.2–100%、ρ 0.976–0.991、CV/盲测 AUC 结论不变；**γ 解耦对照**（σ_c 常数、匹配 63/57/36）→ 一致仅 64.7%、ρ 崩至 0.119、G-only CV AUC 0.816→0.584 而 T-only 几乎不动——量化"能量通道优势含构造耦合、拓扑为下界"；GBE 代理分金属符号（Fe −0.303 顺经典叙事，Cu +0.658/Ni +0.587 反向）

### 4. Discussion
- RQ1–3 逐条作答；γ 构造内耦合 caveat 的诚实处理（T 净贡献为下界，E5 补充"模型不放大"独立证据）
- Devil's advocate 防御（规划 §七）：标签是同代理模型下的相对敏感度（排序主张）；2D 周期 RVE 消表面效应；合成组织以 EBSD 统计为目标分布
- 局限：2D 降采样 RVE、损伤当量模型非相场交叉验证（E6 换律重标已补，相场/实验仍缺）、盲测 20 组织 CI 仍宽（AUC ±0.1–0.13、R² ±0.25）、conformal 校准仍 ~40 行
- 与 FIP-GNN / uMLIP 能量审计的差异定位；姊妹篇 D5 误差传播叙事互引（RQ3 连接两篇）

### 5. Conclusions
- 三句：共同决定论（含 γ 构造耦合 caveat + 拓扑下界）+ 盲测可学（20 组织 AUC 0.80 [0.66,0.91]、R² 0.26–0.38、覆盖 0.86–0.99、OOD 6/180）+ 误差被吸收（G<1）

## 图表清单

| # | 内容 | 状态 | 来源 |
|---|------|------|------|
| Fig 1 | 管线框架图（E1→E2→E3→E4→E5，四层：原子/组织/标签/ML） | ✅ paper/fig1_pipeline.png | make_figs.py |
| Fig 2 | γ 能谱分族点图（金属×GB 族×势，标注孪晶层病态值与 Fe 锚点） | ✅ paper/fig2_energy_spectra.png | gb_energy_library_merged.csv |
| Fig 3 | 组织与渗流：示例组织 ↔ γ_perc_rel 散点（GBE 信号，r 标注） | ✅ paper/fig3_network_features.png | networks/library + network_features.csv |
| Fig 4 | 标签统计：eps_fail 分布（分金属/势）+ 部分失效散点 | ✅ paper/fig4_labels.png | e3_labels.csv |
| Fig 5 | 消融条形图 + 盲测预测-真值+conformal 区间 | ✅ paper/fig5_ablation_blind.png | e4_ablation.csv + prediction_audit.csv |
| Fig 6 | 适用域：残差 vs d5 距离（τ=3.53 精确复算） | ✅ paper/fig6_domain.png | prediction_audit.csv + e4 重放 |
| Fig 7 | 误差传播链：特征漂移/翻转率+锚/回归漂移 三联 | ✅ paper/fig7_propagation.png | error_propagation.csv |
| Fig 8 | 扩展盲测 CI 森林图（AUC/R²/覆盖率 20 组 bootstrap 误差棒 + orig8 灰叉）+ TG-RF 散点（orig8/ext12 着色） | ✅ paper/e4b_blind_ci.png（figures/ 亦存） | ml/e4b_blindtest_ext.py → e4b_blind_ci.csv |
| Tab 1 | E1 能谱与文献/DFT 锚点对照（10 行，锚点 2026-10-24 核验，出处 [10]–[17] → 现 [14,15,24–28,32]，2026-11-03 扩充 32 条后） | ✅ 已排入 manuscript.md | E1验收报告.md §2 |
| Tab 2 | 消融与盲测指标（CV mean±std + 20 组织盲测值与组聚类 bootstrap 95% CI 全列） | ✅ 数据现成 | results/e4_ablation.csv、e4b_blind_ci.csv（orig8 存 e4_blindtest.csv） |
| Tab 3 | 误差传播增益汇总（p×指标） | ✅ 数据现成 | results/error_propagation.csv |

## 交付步骤（todo g9→g10）

1. 补做 Fig 1–4（新绘脚本 `paper/figs_*.py`，统一风格）→ 本步
2. Manuscript 初稿：`paper/manuscript.md`（结构化长稿，后续可转 LaTeX 投稿格式）
3. 终检：数字与台账/CSV 逐一对账、图号引用一致
