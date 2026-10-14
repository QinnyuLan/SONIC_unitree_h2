# H2 MuJoCo Simulator

The simulator owns physics, motor response, rendering and diagnostics. Its DDS bridge receives motor targets and publishes robot state for the C++ runtime. It runs independently of the Python command interfaces.

From the repository root:

```bash
./install.sh sim2sim
bash scripts/run_h2_paper_sim.sh
```

The Python entry point is:

```bash
.venv_sim/bin/python -m sim --help
```

Simulation settings are in `config/sim/h2.yaml`; H2 robot assets are in `assets/robots/h2/`. Resource lookup uses the checkout location, independent of the process working directory. `rendering/` contains viewer/camera code; `diagnostics/` contains simulation logs.

Start the policy runtime in another terminal using `bash scripts/run_sonic_h2_runtime.sh sim`. See [deployment](../../docs/DEPLOYMENT.md#sim2sim) for DDS domains, elastic support and motion commands.
