# ProRL
---

## Installation

Python 3.9 is recommended.

```bash
conda create -n prorl python=3.9 -y
conda activate prorl
pip install -r requirements.txt
```

Before any run, pin BLAS / Numba threads — ProRL uses Python multiprocessing
and assumes single-threaded math workers:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

---

## Quick start: ProRL

Search for a dispatching program on the classic `ft06` instance:

```bash
python -m optimisation.local_search.run_ls \
    --problem_instance_path instances/jsp/ft/ft06 \
    --logdir log/prorl_demo \
    --seed 0 \
    --evaluation_times 10000 --update_times 20 \
    --n_jobs 8 --pop_num 10 \
    --max_tokens 85
```

---

## Quick start: PDR baseline

Evaluate a Priority Dispatching Rule (one of `FIFO`, `SPT`, `MOR`, `MWR`,
`LOR`, `Random`) on the same instance for comparison:

```bash
python -m optimisation.pdr.evaluate \
    --problem_instance_path instances/jsp/ft/ft06 \
    --logdir log/pdr_demo \
    --pdr MWR
```

Output lands at `log/pdr_demo/test/ft06/results.csv`.

---

## Instance suites shipped

All instances live under `instances/jsp/<suite>/<name>`:

| Suite | Jobs × Machines | Notes |
|---|---|---|
| `ft/` | 6×6, 10×10, 20×5 | Fisher & Thompson classics |
| `la/` | 10–30 × 5–15 | Lawrence (40 instances) |
| `orb/` | 10×10 | Applegate & Cook (10 instances) |
| `swv/` | 20–50 × 10–15 | Storer/Wu/Vaccari (20 instances) |
| `yn/` | 20×20 | Yamada & Nakano (4 instances) |
| `abz/` | 10–20 × 10–15 | Adams/Balas/Zawack (5 instances) |
| `ta/` | 15–100 × 15–20 | Taillard's hard suite |
| `ex/`, `cscmax/`, `rcmax/` | various | Demirkol-style large instances |

`instances/jsp/instances.json` lists the best-known-solution (BKS) makespan
for each instance, handy for normalised evaluation.

---

## License

This project is released under the MIT License (see `LICENSE`). The vendored
JSSEnv simulator under `optimisation/jss_env/source/` is also MIT-licensed
and copyright of its original authors.
