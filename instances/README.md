# instances/

Canonical Job Shop Scheduling benchmark instances used by ProRL.
Every runner accepts `--problem_instance_path` pointing at one of these files.

## Layout

```
instances/
└── jsp/
    ├── ft/              # Fisher & Thompson    (ft06, ft10, ft20)
    ├── la/              # Lawrence             (la01 … la40)
    ├── orb/             # Orb / Applegate-Cook (orb01 … orb10)
    ├── swv/             # Storer, Wu, Vaccari  (swv01 … swv20)
    ├── yn/              # Yamada & Nakano      (yn01 … yn04)
    ├── abz/             # Adams, Balas, Zawack (abz5 … abz9)
    ├── ex/              # small teaching / debug instances
    ├── ta/              # Taillard hard suite  (ta01 … ta80)
    ├── cscmax/          # Demirkol-style large (cscmax_*)
    ├── rcmax/           # Demirkol-style large (rcmax_*)
    └── instances.json   # machine-readable catalogue (shape, BKS, …)
```

A full instance path therefore looks like `instances/jsp/ft/ft06` or
`instances/jsp/la/la20`.

## File format

Each instance is a plain-text file:

```
<n_jobs> <n_machines>
<machine> <processing_time>  <machine> <processing_time>  …   (row = job)
```

This is the standard Taillard / OR-Library encoding, which the vendored
JSSEnv simulator under `optimisation/jss_env/source/` expects.

`instances.json` is a machine-readable catalogue of every shipped instance,
listing its shape and best-known-solution makespan — handy for normalised
evaluation.
