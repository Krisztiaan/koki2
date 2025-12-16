# Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krisztiaan/koki2/blob/main/colab/koki2_colab.ipynb)

Use `colab/koki2_colab.ipynb` to bootstrap this repo on Colab (CPU/GPU/TPU) and run a small smoke test.

Tip: to avoid committing execution outputs/metadata back into the repo, use **File → Save a copy in Drive** when running in Colab.

The bootstrap notebook also includes an optional small Stage 1 multi-seed ES + held-out evaluation cell; for the latest Stage 1 protocol, use the dedicated stage notebook below.

## Stage notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krisztiaan/koki2/blob/main/colab/stage1_l02_bad_sources.ipynb)

Use `colab/stage1_l02_bad_sources.ipynb` for the **Stage 1** L0.2 harmful sources experiment (multi-seed ES + held-out eval), aligned with `WORK.md`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krisztiaan/koki2/blob/main/colab/stage1_l10_deplete_bad_sources.ipynb)

Use `colab/stage1_l10_deplete_bad_sources.ipynb` for the **Stage 1** L1.0 deplete/respawn + L0.2 harmful sources experiment (multi-seed ES + held-out eval), aligned with `WORK.md`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krisztiaan/koki2/blob/main/colab/stage1_l10_effectsize_grid.ipynb)

Use `colab/stage1_l10_effectsize_grid.ipynb` for the **Stage 1** pre-registered mini-grid (steps × hazard persistence × success bonus) that strengthens the “L1.0 amplifies survival-weighted strategies” evidence, aligned with `WORK.md`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krisztiaan/koki2/blob/main/colab/stage2_l11_plastic_vs_noplast.ipynb)

Use `colab/stage2_l11_plastic_vs_noplast.ipynb` for the **Stage 2** comparison: plastic vs no-plastic on L1.0 deplete/respawn + L1.1 intermittent gradient, aligned with `WORK.md`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krisztiaan/koki2/blob/main/colab/stage2_l11_modulator_grid.ipynb)

Use `colab/stage2_l11_modulator_grid.ipynb` for the **Stage 2** modulator grid (spike vs drive vs event; small `plast_eta` grid) on the stronger-hazard L1.0+L1.1 setup (`--steps 256`, `--bad-source-respawn-delay 0`), aligned with `WORK.md`.
