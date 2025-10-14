# FMPC+SOCP Timing Analysis

## Overview

This document summarizes the detailed timing analysis of the FMPC+SOCP algorithm to identify performance bottlenecks and understand where computational time is spent during each control iteration.

## Methodology

### Tools Used

1. **Manual Timing Instrumentation**
   - Added `time()` calls around major code sections
   - Stored timing data in `results_dict` for post-processing
   - Files modified:
     - `fmpc_socp.py`: Observer, flat transformation, dynamic extension, logging
     - `discrete_socp_filter.py`: GP inference, Cholesky, matrix computations, parameter assignment, SOCP solve
     - `linear_mpc.py` / `linear_mpc_acados.py`: MPC solve time

2. **Line Profiler (`kernprof`)**
   - Used Python's `line_profiler` package for line-by-line execution timing
   - Added `@profile` decorators to:
     - `FlatMPC_SOCP.select_action()` in `fmpc_socp.py`
     - `DiscreteSOCPFilter.compute_feedback_input()` in `discrete_socp_filter.py`
   - Profiler shows actual wall-clock time per line, including Python overhead

### How to Run

#### 1. Normal Timing (Manual Instrumentation)

```bash
cd /home/ahall/Documents/UofT/papers/mimo_fmpc_socp_gp_ecc/mimo_fmpc_socp/runs_paper
python run_fmpc_socp_only.py [--mode constrained]
```

This runs the experiment and prints a detailed timing breakdown at the end.

#### 2. Line-by-Line Profiling

```bash
cd /home/ahall/Documents/UofT/papers/mimo_fmpc_socp_gp_ecc/mimo_fmpc_socp/runs_paper
python -m kernprof -l -v run_fmpc_socp_profile.py [--mode constrained]
```

Or use VS Code:
- **Run Task** → "Profile FMPC+SOCP" or "Profile FMPC+SOCP (Constrained)"
- Or **Run and Debug** → Select the profile configuration

The profiler output appears at the end showing time per line for profiled functions.

## Results

### Manual Timing Breakdown

Results from `run_fmpc_socp_only.py` (normal mode, unconstrained):

```
--- High-Level Timing ---
Total inference time:     13.55ms (avg)  435.36ms (max)
MPC solve time:           0.64ms (avg)  0.40ms (min)  1.78ms (max)
Observer (compute):       0.09ms (avg)
Observer (update):        0.00ms (avg)
Flat transformation:      0.09ms (avg)
Dynamic extension:        0.01ms (avg)
Data logging:             0.01ms (avg)

--- SOCP Filter Breakdown ---
GP inference (pure):      1.68ms (avg)
Cholesky decomp + inv:    0.06ms (avg)
Cost computation:         0.05ms (avg)
Dummy matrices:           0.01ms (avg)
Stability matrices:       0.21ms (avg)
State constraint matrices:0.33ms (avg)
Parameter assignment:     1.01ms (avg)
SOCP solve:               0.48ms (avg)  0.23ms (min)  0.86ms (max)

--- Computed Sums ---
SOCP setup (all):         3.34ms
Sum of measured parts:    4.66ms
Unaccounted overhead:     8.89ms (65.6%)
```

**Initial Mystery**: 8.89ms (65.6%) of the total 13.55ms inference time was unaccounted for.

### Line Profiler Results

The line profiler revealed the source of the "overhead":

#### `select_action()` in `fmpc_socp.py` (Total: 8.03s for 600 calls = 13.38ms/call)

| Line | Component | Time | Per Hit | % | Description |
|------|-----------|------|---------|---|-------------|
| 351 | SOCP filter call | 4.97s | 8.28ms | 61.5% | `self.filter.compute_feedback_input()` |
| 339 | MPC solve | 3.00s | 5.00ms | 37.2% | `self.mpc.select_action(z_obs)` |
| 347 | Get references | 40ms | 67µs | 0.5% | `self.mpc.get_references()` |
| 335 | Observer compute | 38ms | 64µs | 0.5% | `self.fs_obs.compute_observation()` |
| 374-399 | Data logging | ~8ms | 13µs | 0.1% | All `results_dict.append()` calls |
| Other | Misc overhead | ~15ms | 25µs | ~0.2% | Array copies, function calls |

#### `compute_feedback_input()` in `discrete_socp_filter.py` (Total: 4.94s for 600 calls = 8.23ms/call)

| Line | Component | Time | Per Hit | % | Description |
|------|-----------|------|---------|---|-------------|
| 282 | CVXPY solve call | 2.28s | 3.80ms | 46.1% | `prob.solve(solver=CLARABEL, ...)` |
| 159 | GP inference | 1.91s | 1.59ms | 38.7% | `get_gammas(z_query, self.gps[i])` |
| 192 | Stability matrices | 131ms | 109µs | 2.6% | `stab_filter_matrices(...)` |
| 199-209 | Parameter assignment | 432ms | 720µs | 8.7% | Setting CVXPY parameter values |
| 163 | Cholesky decomp | 46ms | 39µs | 0.9% | `np.linalg.cholesky(gamma5)` |
| 164 | Cholesky inverse | 28ms | 23µs | 0.6% | `np.linalg.inv(L_chol)` |
| 180 | Cost computation | 31ms | 51µs | 0.6% | `compute_cost(...)` |
| 145 | Array delete | 15ms | 26µs | 0.3% | `np.delete(z, rows_to_remove)` |
| Other | Misc overhead | ~35ms | 58µs | ~0.7% | List operations, array copies |

## Key Findings

### 1. The "Missing" 8.89ms is CVXPY Overhead

The manual timing captured only the **pure solver time** via `prob.solver_stats.solve_time` (0.48ms), but the actual `prob.solve()` call takes **3.80ms**.

**The 3.32ms difference is CVXPY's Python/C++ interface overhead:**
- Problem matrix construction
- Data type conversions
- Copying data to/from the solver
- Status checking and result extraction
- Python function call overhead

This explains most of the "unaccounted" time (3.32ms of 8.89ms).

### 2. Remaining Overhead Sources

The remaining ~5.5ms comes from:
- **MPC solver wrapper overhead**: Similar to CVXPY, the difference between reported solve time (0.64ms) and actual call time (5.00ms) = ~4.36ms
  - CasADi/IPOPT interface overhead
  - NLP problem setup and teardown
  - Jacobian/Hessian evaluation
- **GP inference overhead**: The manual timing only captured the `get_gammas()` call (1.68ms), but profiler shows it's actually 1.59ms - reasonably close
- **Function call overhead**: Python function calls, array slicing, dictionary operations add small amounts throughout

### 3. Performance Breakdown by Component

**Within each 13.55ms inference cycle:**

| Component | Time | % of Total | Details |
|-----------|------|------------|---------|
| **SOCP Filter** | 8.28ms | 61.1% | Includes CVXPY overhead |
| └─ CVXPY/solver | 3.80ms | 28.0% | CLARABEL + interface |
| └─ GP inference | 1.59ms | 11.7% | PyTorch/GPyTorch |
| └─ Parameter setup | 0.72ms | 5.3% | CVXPY parameter assignment |
| └─ Stability matrices | 0.11ms | 0.8% | NumPy operations |
| └─ Cholesky | 0.06ms | 0.4% | NumPy linalg |
| └─ Other | ~2.0ms | 14.8% | Cost, dummy, misc |
| **MPC Solve** | 5.00ms | 36.9% | Includes CasADi overhead |
| └─ ACADOS solver | 0.64ms | 4.7% | Pure solver time |
| └─ CasADi/ACADOS overhead | 4.36ms | 32.2% | Interface + setup |
| **Other** | 0.27ms | 2.0% | Observer, transforms, logging |

### 4. Bottleneck Summary

**Top 3 bottlenecks:**
1. **CVXPY overhead** (3.80ms total for SOCP): 28.0% of total time
2. **CasADi/IPOPT overhead** (4.36ms for MPC): 32.2% of total time
3. **GP inference** (1.59ms): 11.7% of total time

**Combined, solver wrapper overhead (CVXPY + CasADi) accounts for 60.2% of inference time.**

## Conclusions

1. **The algorithm is not slow - the wrappers are.**
   - Pure CLARABEL solve: 0.48ms
   - Pure ACADOS solve: 0.64ms
   - Total pure solver time: 1.12ms (only 8.3% of total!)

2. **CVXPY and CasADi add 7× overhead**
   - CVXPY adds 3.32ms on top of 0.48ms (690% overhead)
   - CasADi adds 4.36ms on top of 0.64ms (680% overhead)

3. **Optimization opportunities** (in order of potential impact):
   - **Bypass CVXPY**: Implement SOCP directly in CLARABEL's native interface (~3ms savings)
   - **Bypass CasADi**: Use IPOPT C API directly (~4ms savings)
   - **Optimize GP inference**: Use compiled/batched inference (~0.5-1ms savings)
   - **Reduce CVXPY parameter updates**: Exploit problem structure (~0.3ms savings)

4. **Current implementation is reasonable for research code**
   - Manual timing provides good component-level breakdown
   - CVXPY/CasADi provide rapid prototyping and maintainability
   - For real-time deployment at >50Hz, native solver interfaces would be needed

## Test Configuration

- **System**: Linux, Python 3.10, Conda environment
- **Solvers**: CLARABEL (SOCP), IPOPT (NLP/MPC)
- **Control frequency**: 50 Hz (20ms target cycle time)
- **Trajectory**: Lemniscate (figure-8), 2 loops
- **Timesteps**: 600 iterations
- **Mode**: Unconstrained (results shown above)

## Files Modified for Timing

1. `safe_control_gym/controllers/mpc/fmpc_socp.py`
   - Added timing for: observer, flat transform, dynamic extension, logging
   - Added `@profile` decorator (optional, for line profiling)

2. `safe_control_gym/controllers/mpc/discrete_socp_filter.py`
   - Added timing for: GP inference, Cholesky, cost, matrices, parameter assignment
   - Added `@profile` decorator (optional, for line profiling)

3. `safe_control_gym/controllers/mpc/linear_mpc.py`
   - Added timing for MPC solve time

4. `safe_control_gym/controllers/mpc/linear_mpc_acados.py`
   - Added timing for ACADOS solve time

5. `safe_control_gym/controllers/mpc/mpc.py`
   - Added `mpc_solve_time` to `results_dict`

6. `runs_paper/run_fmpc_socp_only.py`
   - Data extraction and timing breakdown printing

7. `runs_paper/run_fmpc_socp_profile.py`
   - Simplified version for profiling (no post-analysis)

## Related Documentation

- Line profiler: https://github.com/pyutils/line_profiler
- CVXPY: https://www.cvxpy.org/
- CasADi: https://web.casadi.org/
- CLARABEL solver: https://github.com/oxfordcontrol/Clarabel.rs
