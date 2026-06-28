# Project Context

## 1. Research Background
- **Subject**: Intrahepatic cholangiocarcinoma (iCCA) organoid drug-response study.
- **Imaging Modality**: 1310 nm OCT (Optical Coherence Tomography), 3D label-free live imaging.
- **Drug**: Icaritin (阿可拉定), concentrations: 20 / 40 / 80 μM.
- **Time Points**: Day 3 (pre-drug) → Day 5 (48 h post-drug).

## 2. Core Metric — OAC (Optical Attenuation Coefficient)
- **Symbol**: μs, unit mm⁻¹.
- **Physical Meaning**: Probability of NIR light attenuation per unit length; at 1310 nm absorption is negligible, so OAC ≈ scattering coefficient, reflecting tissue micro-density.
- **Algorithm Used in Thesis**: Single-scattering Depth-Resolved (DR) with dual-B-scan calibration:
  $$\mu_s[i] = \frac{I[i]/A[i]}{2 \cdot \Delta \cdot \sum_{j=i+1}^{N} I[j]/A[j]}$$
  - I[i]: raw OCT signal at depth i
  - A[i]: system calibration coefficient (roll-off & confocal correction)
  - Δ: axial pixel spacing (5.5 μm)
- **Outputs**: OAC Mean (group average), OAC Std (intra-organoid heterogeneity).

## 3. OAC_RATES — Relative Change Rate (%)
Defined in the BOE paper / master thesis:
$$Relative\ change\ rate = \frac{P_{after} - P_{before}}{P_{before}} \times 100\%$$
- Positive = OAC increase (apoptosis, tissue densification).
- Negative = OAC decrease (natural development, lumen expansion).

### Reference Values (calculated from paper Fig.3 / Fig.4)
| Group | Day3 (before) | Day5 (after) | OAC_RATES |
|-------|---------------|--------------|-----------|
| Control | 1.36 mm⁻¹ | 1.15 mm⁻¹ | **-14.3** |
| 20 μM | 1.73 mm⁻¹ | 1.76 mm⁻¹ | **1.9** |
| 40 μM | 1.97 mm⁻¹ | 2.15 mm⁻¹ | **9.3** |
| 80 μM | 1.65 mm⁻¹ | 1.92 mm⁻¹ | **18.5** |

Python mapping:
```python
OAC_RATES = {
    'Control': -14.3,
    '20': 1.9,
    '40': 9.3,
    '80': 18.5,
}
```

## 4. Biological Interpretation
- **Control**: Healthy organoids have loose cells and large lumens → low OAC; natural growth enlarges lumens → OAC drops (-14.3%).
- **Drug treated**: Apoptosis causes nuclear pyknosis, chromatin condensation, organelle accumulation → enhanced Mie scattering at 1310 nm → OAC rises dose-dependently.
- **Key Advantage**: OAC detects sub-cellular apoptotic changes earlier than macroscopic volume shrinkage (late event), reducing false negatives.

## 5. Distinction — Relative Response Ratio
- **OAC_RATES**: Within-group before→after percentage change (single group only).
- **Relative Response Ratio**: Baseline-corrected, eliminates control drift:
  $$\frac{\Delta_{Drug} - \Delta_{Ctrl}}{|\Delta_{Ctrl}|}$$
  Do **not** confuse the two.

## 6. Role in Evaluation System
1. **Single-feature biomarker**: Relative response ratio up to 2.29 (2× volume metric).
2. **Multi-dimensional model**: Combined with volume, sphericity, roughness → K-means++ / PCA.
3. **Composite score**: OAC weight = -0.128 (higher OAC = lower viability).
4. **Validation**: ATP gold-standard correlation improved from 0.906 → 0.938 after fusing OAC; combined therapy test r = 0.887.

## 7. Other OAC Algorithms (for reference)
| Algorithm | Pixel-wise 3D | High-throughput | Cavity/heterogeneity | Speed |
|-----------|---------------|-----------------|----------------------|-------|
| Curve Fitting (CF) | ❌ | ✅ | Poor | Very fast |
| Dual-B-scan DR (thesis) | ✅ | ✅ | Excellent | Fast |
| Layered iterative DR | ✅ | Medium | Good | Medium |
| EHF multi-scattering | ✅ | Slow | Excellent | Slow |
| Monte Carlo | ✅ | N/A | Perfect | Very slow |
| Deep learning | ✅ | Fast (after train) | Excellent | Fast |

## 8. Frequently Used Constants / Parameters
- Axial pixel spacing Δ = 5.5 μm
- OAC_RATES mapping keys: `'Control'`, `'20'`, `'40'`, `'80'`
- PCA / K-means features often include: `volume`, `sphericity`, `roughness`, `OAC_mean`, `OAC_std`

## 9. Coding Conventions in This Repo
- Prefer loading `.mat` or `.npy` OCT pre-processed data.
- Group-level stats: compute per-organoid OAC mean → aggregate by group → calculate relative change rates.
- Plotting: use consistent group colors (Control=gray, 20μM=blue, 40μM=orange, 80μM=red if not otherwise specified).
