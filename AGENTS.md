# Karpathy-Inspired Agent Guidelines & Project Rules

This repository enforces behavioral guidelines derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls, combined with geospatial and carbon MRV domain constraints.

---

## 1. Core Karpathy Principles

### 1.1 Think Before Coding
- **Don't assume. Don't hide confusion. Surface tradeoffs.**
- State your assumptions explicitly before editing. If uncertain, ask rather than guess.
- If multiple interpretations exist, present them — do not pick silently.
- If a simpler approach exists, propose it. Stop and ask for clarification when confused.

### 1.2 Simplicity First
- **Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No premature abstractions or wrapper classes for single-use code.
- No "flexibility" or "configurability" flags that were not requested.
- If 200 lines could be 50, write the 50-line version.

### 1.3 Surgical Changes
- **Touch only what you must. Clean up only your own mess.**
- Do not "improve" adjacent code, comments, or formatting orthogonal to the task.
- Do not refactor things that are not broken.
- Match existing repository style and conventions.
- If your changes create unused imports, variables, or functions, remove them. Never delete pre-existing dead code unless explicitly requested.

### 1.4 Goal-Driven Execution
- **Define success criteria. Loop until verified.**
- Transform requests into testable, verifiable goals:
  - Add feature $\rightarrow$ verify with unit test or reproducible execution.
  - Fix bug $\rightarrow$ write reproducing check, then verify resolution.
- State a brief step-by-step verification plan for multi-step tasks.

---

## 2. Geospatial & Carbon MRV Domain Rules

1. **Projected CRS Integrity**:
   - Never compute polygon or raster areas in `EPSG:4326` or `EPSG:3857`.
   - All spatial calculations and mapping in Chhattisgarh must use `EPSG:32644` (WGS 84 / UTM Zone 44N).

2. **Conceptual & Metric Separation**:
   - Keep satellite-observed rice extent (ha), inferred seasonal inundation duration (days), modelled IPCC Tier-2 methane flux (kg / tonnes $\text{CH}_4$), and counterfactual AWD scenario potentials in strictly separate variables and columns.
   - Keep mass ($\text{CH}_4$ in tonnes) separate from $\text{CO}_2\text{e}$.

3. **Global Warming Potentials (GWPs)**:
   - Use IPCC AR6 100-year GWP values: $\text{CH}_4 = 27.9$, $\text{N}_2\text{O} = 273$.

4. **Zero Data Fabrication**:
   - If district-level input or parameter is unavailable, output `np.nan` / `None`, decrease `data_completeness_score`, and log limitations.
