"""
Partition-seeded multi-start analytical portfolio submission.

This is a repo-native implementation of the strategy described in
`submissions/deep-research-report.md`. The original report recommends
KaHyPar + DREAMPlace + AutoDMP. This runner now supports optional
integration points for KaHyPar / Mt-KaHyPar and a DREAMPlace backend,
while preserving a repo-native fallback when those dependencies are not
installed in the current environment.

1. Build multiple partition-derived hard-macro seeds.
2. Run a small analytical portfolio from those seeds.
3. Legalize hard macros after each major refinement stage.
4. Follow hard-macro updates with cheap soft-macro barycenter updates.
5. Rank the candidate portfolio with the exact challenge proxy when the
   underlying PlacementCost data is available.

Usage:
    uv run evaluate submissions/runner.py
    uv run evaluate submissions/runner.py --all

Optional backends:
    - `kahypar` Python module or `mtkahypar` Python module
    - `PARTCL_KAHYPAR_BINARY=/path/to/KaHyPar`
    - `PARTCL_KAHYPAR_INI=/path/to/km1_kKaHyPar.ini`
    - `PARTCL_DREAMPLACE_MODULE=my_package.my_dreamplace_adapter`

The DREAMPlace adapter module is expected to expose one of:
    - `run_dreamplace(...)`
    - `place(...)`

and return a `[num_hard_macros, 2]` numpy array or tensor. This keeps the
challenge-native benchmark representation decoupled from DREAMPlace's
native database formats.
"""

import importlib
import importlib.util
import inspect
import os
import time
from pathlib import Path
import shutil
import subprocess
import tempfile
from collections import deque

import numpy as np
import torch

from macro_place.benchmark import Benchmark


class PartitionSeededPortfolioPlacer:
    def __init__(self, seed: int = 17):
        self.seed = seed
        self._plc_cache = {}
        self._graph_cache = {}
        self._partition_cache = {}
        self._backend_cache = None
        self._run_state = {}
        self._exact_cache = {}
        self._exact_cache_order = deque()
        self._exact_cache_limit = 512

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._run_state = {
            "dreamplace_backend": "untried",
            "partition_backend": {},
            "exact_cache_hits": 0,
            "exact_cache_misses": 0,
            "exact_time_s": 0.0,
        }
        rng = np.random.default_rng(self.seed + self._stable_name_seed(benchmark.name))
        graph = self._get_graph_cache(benchmark)
        plc = self._load_plc(benchmark.name)
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        profile_mode = os.environ.get("PARTCL_PROFILE_MODE", "").strip().lower()
        only_candidates = os.environ.get("PARTCL_ONLY_CANDIDATES", "").strip()
        allowed_only = {name.strip() for name in only_candidates.split(",") if name.strip()}
        focused_dreamplace_run = bool(allowed_only) and all(
            name.startswith("dreamplace_full_") for name in allowed_only
        )
        setup_complexity = max(num_hard, num_macros)
        fast_focused_min_hard = int(os.environ.get("PARTCL_FAST_FOCUSED_MIN_HARD", "380"))

        fast_focused_init = (
            focused_dreamplace_run
            and num_hard >= fast_focused_min_hard
            and os.environ.get("PARTCL_FAST_FOCUSED_INIT", "1") != "0"
        )
        if fast_focused_init:
            base = benchmark.macro_positions.clone()
            base_hard = self._legalize_hard_numpy(
                base[:num_hard].cpu().numpy().astype(np.float64),
                benchmark,
            )
            base[:num_hard] = torch.tensor(base_hard, dtype=base.dtype)
        else:
            base = self._build_random_start(benchmark)
        hard_base = base[:num_hard].cpu().numpy().astype(np.float64)
        if fast_focused_init:
            outline_placement = benchmark.macro_positions.clone()
            ref_hard = self._legalize_hard_numpy(
                outline_placement[:num_hard].cpu().numpy().astype(np.float64),
                benchmark,
            )
            outline_placement[:num_hard] = torch.tensor(ref_hard, dtype=outline_placement.dtype)
            print(
                f"[partcl:init] {benchmark.name} fast focused DREAMPlace init "
                f"(skipped outline pre-portfolio)",
                flush=True,
            )
        else:
            outline_placement = self._build_outline_baseline(benchmark, graph, rng, plc=plc)
        if plc is not None and profile_mode not in {
            "dreamplace_basin",
            "dreamplace_basin_deep",
            "dreamplace_basin_rebalance",
        }:
            rebalanced_outline = self._hard_outer_rebalance_polish(
                outline_placement.clone(),
                benchmark,
                graph,
                plc,
            )
            outline_costs = self._exact_costs(outline_placement, benchmark, plc)
            rebalanced_costs = self._exact_costs(rebalanced_outline, benchmark, plc)
            if rebalanced_costs["proxy_cost"] + 1.0e-4 < outline_costs["proxy_cost"]:
                print(
                    f"[partcl:outline-base] {benchmark.name} "
                    f"hard-rebalance "
                    f"{outline_costs['proxy_cost']:.4f}->{rebalanced_costs['proxy_cost']:.4f}"
                )
                outline_placement = rebalanced_outline
        outline_hard = outline_placement[:num_hard].cpu().numpy().astype(np.float64)

        complexity = setup_complexity
        size_scale = float(np.clip(np.sqrt(280.0 / max(280.0, float(complexity))), 0.45, 1.0))
        repulsion_period = 1 if complexity <= 320 else 2 if complexity <= 430 else 3
        candidate_specs = []
        candidate_specs.append(
            {
                "name": "outline_legal",
                "direct_placement": True,
                "preserve_soft": True,
                "placement": outline_placement.clone(),
            }
        )
        candidate_specs.append(
            {
                "name": "outline_soft_search",
                "direct_placement": True,
                "iterated_soft_search_refine": True,
                "placement": outline_placement.clone(),
            }
        )
        candidate_specs.append(
            {
                "name": "rebalance_first_search",
                "direct_placement": True,
                "rebalance_first_refine": True,
                "placement": outline_placement.clone(),
            }
        )
        candidate_specs.append(
            {
                "name": "corridor_first_search",
                "direct_placement": True,
                "corridor_first_refine": True,
                "placement": outline_placement.clone(),
            }
        )
        if os.environ.get("PARTCL_ENABLE_DREAMPLACE_BASIN", "1") != "0":
            dreamplace_full_base_cfg = {
                "return_all_macros": 1,
                "macro_place_flag": 0,
                "single_stage": 1,
                "enable_fillers": 0,
                "omit_soft_macros": 0,
                "use_bb": 1,
                "target_density": 0.88,
                "density_weight": 1.5e-5,
                "gamma": 4.0,
                "lr": 0.008,
                "steps": 26 if complexity <= 320 else 24,
                "num_bins_scale": 0.5,
                "stage1_bins_scale": 1.0,
                "stage1_iter_scale": 18,
                "stop_overflow": 0.10,
                "adjust_rudy_area_flag": 0,
                "adjust_pin_area_flag": 0,
                "max_num_area_adjust": 0,
                "random_center_init_flag": 0,
            }
            candidate_specs.append(
                {
                    "name": "dreamplace_full_outline_flat_rebalance",
                    "dreamplace_full_placement": True,
                    "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                    "base_placement": outline_placement.clone(),
                    "rebalance_first_refine": profile_mode != "dreamplace_basin",
                    "cfg": dreamplace_full_base_cfg.copy(),
                }
            )
            for variant_name, variant_cfg in (
                (
                    "dreamplace_full_outline_balanced",
                    {
                        **dreamplace_full_base_cfg,
                        "target_density": 0.84,
                        "density_weight": 2.8e-5,
                        "gamma": 5.2,
                        "lr": 0.0065,
                        "num_bins_scale": 0.75,
                        "stop_overflow": 0.08,
                    },
                ),
                (
                    "dreamplace_full_outline_dense_sharp",
                    {
                        **dreamplace_full_base_cfg,
                        "target_density": 0.90,
                        "density_weight": 3.0e-5,
                        "gamma": 5.6,
                        "lr": 0.006,
                        "num_bins_scale": 0.75,
                        "stop_overflow": 0.08,
                    },
                ),
                (
                    "dreamplace_full_outline_loose",
                    {
                        **dreamplace_full_base_cfg,
                        "target_density": 0.76,
                        "density_weight": 4.0e-5,
                        "gamma": 5.4,
                        "lr": 0.006,
                        "num_bins_scale": 0.75,
                        "stop_overflow": 0.07,
                    },
                ),
            ):
                candidate_specs.append(
                    {
                        "name": variant_name,
                        "dreamplace_full_placement": True,
                        "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                        "base_placement": outline_placement.clone(),
                        "cfg": variant_cfg.copy(),
                    }
                )
            if profile_mode == "dreamplace_basin_deep" or os.environ.get("PARTCL_ENABLE_DREAMPLACE_DEEP", "0") == "1":
                for variant_name, variant_cfg in (
                    (
                        "dreamplace_full_outline_balanced_g6",
                        {
                            **dreamplace_full_base_cfg,
                            "target_density": 0.84,
                            "density_weight": 2.8e-5,
                            "gamma": 6.0,
                            "lr": 0.006,
                            "num_bins_scale": 0.75,
                            "stop_overflow": 0.075,
                        },
                    ),
                    (
                        "dreamplace_full_outline_balanced_bins",
                        {
                            **dreamplace_full_base_cfg,
                            "target_density": 0.84,
                            "density_weight": 2.8e-5,
                            "gamma": 5.2,
                            "lr": 0.006,
                            "num_bins_scale": 1.0,
                            "stop_overflow": 0.075,
                        },
                    ),
                    (
                        "dreamplace_full_outline_balanced_long",
                        {
                            **dreamplace_full_base_cfg,
                            "target_density": 0.84,
                            "density_weight": 2.6e-5,
                            "gamma": 5.4,
                            "lr": 0.0055,
                            "steps": 34 if complexity <= 320 else 30,
                            "num_bins_scale": 0.75,
                            "stage1_iter_scale": 24,
                            "stop_overflow": 0.07,
                        },
                    ),
                    (
                        "dreamplace_full_outline_mid_dense",
                        {
                            **dreamplace_full_base_cfg,
                            "target_density": 0.87,
                            "density_weight": 2.6e-5,
                            "gamma": 5.8,
                            "lr": 0.006,
                            "num_bins_scale": 0.75,
                            "stop_overflow": 0.075,
                        },
                    ),
                    (
                        "dreamplace_full_outline_mid_loose",
                        {
                            **dreamplace_full_base_cfg,
                            "target_density": 0.81,
                            "density_weight": 3.2e-5,
                            "gamma": 5.6,
                            "lr": 0.006,
                            "num_bins_scale": 0.75,
                            "stop_overflow": 0.07,
                        },
                    ),
                    (
                        "dreamplace_full_outline_zero_noise",
                        {
                            **dreamplace_full_base_cfg,
                            "target_density": 0.84,
                            "density_weight": 2.8e-5,
                            "gamma": 5.2,
                            "lr": 0.0065,
                            "num_bins_scale": 0.75,
                            "gp_noise_ratio": 0.0,
                            "stop_overflow": 0.08,
                        },
                    ),
                    (
                        "dreamplace_full_outline_route_diag",
                        {
                            **dreamplace_full_base_cfg,
                            "target_density": 0.82,
                            "density_weight": 3.0e-5,
                            "gamma": 4.8,
                            "lr": 0.0065,
                            "num_bins_scale": 0.75,
                            "stop_overflow": 0.08,
                            "routability_opt_flag": 1,
                            "adjust_rudy_area_flag": 1,
                            "adjust_pin_area_flag": 1,
                            "max_num_area_adjust": 2,
                        },
                    ),
                ):
                    candidate_specs.append(
                        {
                            "name": variant_name,
                            "dreamplace_full_placement": True,
                            "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                            "base_placement": outline_placement.clone(),
                            "cfg": variant_cfg.copy(),
                        }
                    )
                balanced_cfg = {
                    **dreamplace_full_base_cfg,
                    "target_density": 0.84,
                    "density_weight": 2.8e-5,
                    "gamma": 5.2,
                    "lr": 0.0065,
                    "num_bins_scale": 0.75,
                    "stop_overflow": 0.08,
                }
                for polish_name, polish_flags in (
                    ("dreamplace_full_outline_balanced_routepolish", {"dreamplace_route_polish": True}),
                    ("dreamplace_full_outline_balanced_channelpolish", {"dreamplace_channel_polish": True}),
                    ("dreamplace_full_outline_balanced_whitepolish", {"dreamplace_white_polish": True}),
                    (
                        "dreamplace_full_outline_balanced_white2",
                        {"dreamplace_white_polish": True, "dreamplace_white_polish_rounds": 2},
                    ),
                    (
                        "dreamplace_full_outline_balanced_white3",
                        {"dreamplace_white_polish": True, "dreamplace_white_polish_rounds": 3},
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_route",
                        {"dreamplace_white_polish": True, "dreamplace_route_polish": True},
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_channel",
                        {"dreamplace_white_polish": True, "dreamplace_channel_polish": True},
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_exactsoft",
                        {"dreamplace_white_polish": True, "dreamplace_exact_soft_polish": True},
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_fine",
                        {
                            "dreamplace_white_polish": True,
                            "dreamplace_white_polish_cfg": {
                                "limit": 32,
                                "region_rows": 5,
                                "region_cols": 5,
                                "anchor_weight": 0.040,
                                "relax": 0.68,
                                "solver_steps": 20,
                                "max_disp_frac": 0.012,
                                "step_scale": 0.24,
                            },
                        },
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_strong",
                        {
                            "dreamplace_white_polish": True,
                            "dreamplace_white_polish_cfg": {
                                "limit": 36,
                                "region_rows": 5,
                                "region_cols": 5,
                                "anchor_weight": 0.035,
                                "relax": 0.64,
                                "solver_steps": 22,
                                "max_disp_frac": 0.016,
                                "step_scale": 0.34,
                            },
                        },
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_local",
                        {
                            "dreamplace_white_polish": True,
                            "dreamplace_white_polish_cfg": {
                                "limit": 18,
                                "region_rows": 6,
                                "region_cols": 6,
                                "anchor_weight": 0.055,
                                "relax": 0.74,
                                "solver_steps": 14,
                                "max_disp_frac": 0.009,
                                "step_scale": 0.22,
                            },
                        },
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_local_push",
                        {
                            "dreamplace_white_polish": True,
                            "dreamplace_white_polish_cfg": {
                                "limit": 22,
                                "region_rows": 6,
                                "region_cols": 6,
                                "anchor_weight": 0.050,
                                "relax": 0.72,
                                "solver_steps": 16,
                                "max_disp_frac": 0.011,
                                "step_scale": 0.30,
                            },
                        },
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_local_fine",
                        {
                            "dreamplace_white_polish": True,
                            "dreamplace_white_polish_cfg": {
                                "limit": 24,
                                "region_rows": 7,
                                "region_cols": 7,
                                "anchor_weight": 0.052,
                                "relax": 0.72,
                                "solver_steps": 16,
                                "max_disp_frac": 0.010,
                                "step_scale": 0.26,
                            },
                        },
                    ),
                    (
                        "dreamplace_full_outline_balanced_white_local_soft",
                        {
                            "dreamplace_white_polish": True,
                            "dreamplace_white_polish_cfg": {
                                "limit": 16,
                                "region_rows": 6,
                                "region_cols": 6,
                                "anchor_weight": 0.060,
                                "relax": 0.76,
                                "solver_steps": 12,
                                "max_disp_frac": 0.008,
                                "step_scale": 0.20,
                            },
                        },
                    ),
                    (
                        "dreamplace_full_outline_balanced_route_channel",
                        {"dreamplace_route_polish": True, "dreamplace_channel_polish": True},
                    ),
                ):
                    candidate_specs.append(
                        {
                            "name": polish_name,
                            "dreamplace_full_placement": True,
                            "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                            "base_placement": outline_placement.clone(),
                            "cfg": balanced_cfg.copy(),
                            **polish_flags,
                        }
                    )
                dense_sharp_cfg = {
                    **dreamplace_full_base_cfg,
                    "target_density": 0.90,
                    "density_weight": 3.0e-5,
                    "gamma": 5.6,
                    "lr": 0.006,
                    "num_bins_scale": 0.75,
                    "stop_overflow": 0.08,
                }
                loose_cfg = {
                    **dreamplace_full_base_cfg,
                    "target_density": 0.76,
                    "density_weight": 4.0e-5,
                    "gamma": 5.4,
                    "lr": 0.006,
                    "num_bins_scale": 0.75,
                    "stop_overflow": 0.07,
                }
                for variant_name, variant_cfg in (
                    ("dreamplace_full_outline_dense_sharp_white", dense_sharp_cfg),
                    ("dreamplace_full_outline_loose_white", loose_cfg),
                    ("dreamplace_full_outline_flat_white", dreamplace_full_base_cfg),
                ):
                    candidate_specs.append(
                        {
                            "name": variant_name,
                            "dreamplace_full_placement": True,
                            "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                            "base_placement": outline_placement.clone(),
                            "cfg": variant_cfg.copy(),
                            "dreamplace_white_polish": True,
                        }
                    )
                random_balanced_cfg = {
                    **balanced_cfg,
                    "random_center_init_flag": 1,
                    "gp_noise_ratio": 0.02,
                }
                for variant_name, polish in (
                    ("dreamplace_full_balanced_random", False),
                    ("dreamplace_full_balanced_random_white", True),
                ):
                    candidate_specs.append(
                        {
                            "name": variant_name,
                            "dreamplace_full_placement": True,
                            "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                            "base_placement": outline_placement.clone(),
                            "cfg": random_balanced_cfg.copy(),
                            "dreamplace_white_polish": polish,
                        }
                    )
        # Multi-seed DREAMPlace ensemble.
        # Three configs (balanced / dense_sharp / loose) x four random seeds.
        # The current production winner is single-seed dreamplace_full_outline_balanced_white_local_soft;
        # we add 12 sibling variants that differ only in (random_seed, gp_noise_ratio) and apply
        # the same whitespace soft polish. Top-2 by exact proxy will be chosen by the existing
        # shortlist logic (see _exact_score loop below), so the cost of losing seeds is bounded.
        if os.environ.get("PARTCL_ENABLE_DREAMPLACE_ENSEMBLE", "1") != "0":
            ensemble_base_cfg = {
                "return_all_macros": 1,
                "macro_place_flag": 0,
                "single_stage": 1,
                "enable_fillers": 0,
                "omit_soft_macros": 0,
                "use_bb": 1,
                "num_bins_scale": 0.75,
                "stage1_bins_scale": 1.0,
                "stage1_iter_scale": 18,
                "stop_overflow": 0.08,
                "adjust_rudy_area_flag": 0,
                "adjust_pin_area_flag": 0,
                "max_num_area_adjust": 0,
                "random_center_init_flag": 0,
                "steps": 26 if complexity <= 320 else 24,
            }
            ensemble_cfg_variants = (
                (
                    "balanced",
                    {
                        "target_density": 0.84,
                        "density_weight": 2.8e-5,
                        "gamma": 5.2,
                        "lr": 0.0065,
                    },
                ),
                (
                    "dense_sharp",
                    {
                        "target_density": 0.90,
                        "density_weight": 3.2e-5,
                        "gamma": 5.6,
                        "lr": 0.006,
                    },
                ),
                (
                    "loose",
                    {
                        "target_density": 0.78,
                        "density_weight": 3.8e-5,
                        "gamma": 5.4,
                        "lr": 0.006,
                    },
                ),
            )
            # Four well-spread seeds. Hash benchmark.name in too so different benchmarks
            # don't accidentally share an identical full search basin.
            name_salt = sum(ord(ch) for ch in benchmark.name) * 7919
            ensemble_seeds = (
                (1000 + name_salt + 0, 0.005),
                (1000 + name_salt + 7919, 0.010),
                (1000 + name_salt + 15737, 0.018),
                (1000 + name_salt + 24593, 0.012),
            )
            ensemble_polish_cfg = {
                "limit": 16,
                "region_rows": 6,
                "region_cols": 6,
                "anchor_weight": 0.060,
                "relax": 0.76,
                "solver_steps": 12,
                "max_disp_frac": 0.008,
                "step_scale": 0.20,
            }
            for cfg_label, cfg_overrides in ensemble_cfg_variants:
                for seed_idx, (random_seed, gp_noise) in enumerate(ensemble_seeds):
                    variant_cfg = {
                        **ensemble_base_cfg,
                        **cfg_overrides,
                        "random_seed": int(random_seed),
                        "gp_noise_ratio": float(gp_noise),
                    }
                    candidate_specs.append(
                        {
                            "name": f"dreamplace_ensemble_{cfg_label}_s{seed_idx}",
                            "dreamplace_full_placement": True,
                            "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                            "base_placement": outline_placement.clone(),
                            "cfg": variant_cfg,
                            "dreamplace_white_polish": True,
                            "dreamplace_white_polish_cfg": dict(ensemble_polish_cfg),
                        }
                    )

        # RUDY-based congestion refiner.
        # Standalone candidate (cheap, runs from outline_legal) + RUDY-polished
        # sibling of the current production winner. Both gated on/off via env.
        if os.environ.get("PARTCL_ENABLE_RUDY", "1") != "0":
            candidate_specs.append(
                {
                    "name": "outline_rudy_refine",
                    "direct_placement": True,
                    "rudy_refine": True,
                    "rudy_refine_rounds": 2,
                    "placement": outline_placement.clone(),
                }
            )
            if os.environ.get("PARTCL_ENABLE_DREAMPLACE_BASIN", "1") != "0":
                rudy_polish_cfg = {
                    **dreamplace_full_base_cfg,
                    "target_density": 0.84,
                    "density_weight": 2.8e-5,
                    "gamma": 5.2,
                    "lr": 0.0065,
                    "num_bins_scale": 0.75,
                    "stop_overflow": 0.08,
                }
                rudy_white_cfg = {
                    "limit": 16,
                    "region_rows": 6,
                    "region_cols": 6,
                    "anchor_weight": 0.060,
                    "relax": 0.76,
                    "solver_steps": 12,
                    "max_disp_frac": 0.008,
                    "step_scale": 0.20,
                }
                candidate_specs.append(
                    {
                        "name": "dreamplace_full_outline_balanced_white_local_soft_rudy",
                        "dreamplace_full_placement": True,
                        "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                        "base_placement": outline_placement.clone(),
                        "cfg": rudy_polish_cfg.copy(),
                        "dreamplace_white_polish": True,
                        "dreamplace_white_polish_cfg": rudy_white_cfg,
                        "dreamplace_rudy_polish": True,
                        "dreamplace_rudy_polish_rounds": 2,
                    }
                )

        candidate_specs.append(
            {
                "name": "outline_softopt_torch",
                "direct_placement": True,
                "softopt_torch_refine": True,
                "placement": outline_placement.clone(),
            }
        )
        if profile_mode:
            allowed_names = {
                "outline": {"outline_legal", "outline_soft_search", "rebalance_first_search", "corridor_first_search"},
                "outline_torch": {"outline_legal", "outline_soft_search", "rebalance_first_search", "corridor_first_search", "outline_softopt_torch"},
                "dreamplace_basin": {
                    "outline_legal",
                    "dreamplace_full_outline_flat_rebalance",
                    "dreamplace_full_outline_balanced",
                    "dreamplace_full_outline_dense_sharp",
                    "dreamplace_full_outline_loose",
                },
                "dreamplace_basin_rebalance": {
                    "outline_legal",
                    "dreamplace_full_outline_flat_rebalance",
                    "dreamplace_full_outline_balanced",
                    "dreamplace_full_outline_dense_sharp",
                    "dreamplace_full_outline_loose",
                },
                "dreamplace_basin_deep": {
                    "outline_legal",
                    "dreamplace_full_outline_flat_rebalance",
                    "dreamplace_full_outline_balanced",
                    "dreamplace_full_outline_dense_sharp",
                    "dreamplace_full_outline_loose",
                    "dreamplace_full_outline_balanced_g6",
                    "dreamplace_full_outline_balanced_bins",
                    "dreamplace_full_outline_balanced_long",
                    "dreamplace_full_outline_mid_dense",
                    "dreamplace_full_outline_mid_loose",
                    "dreamplace_full_outline_zero_noise",
                    "dreamplace_full_outline_route_diag",
                    "dreamplace_full_outline_balanced_routepolish",
                    "dreamplace_full_outline_balanced_channelpolish",
                    "dreamplace_full_outline_balanced_whitepolish",
                    "dreamplace_full_outline_balanced_white2",
                    "dreamplace_full_outline_balanced_white3",
                    "dreamplace_full_outline_balanced_white_route",
                    "dreamplace_full_outline_balanced_white_channel",
                    "dreamplace_full_outline_balanced_white_exactsoft",
                    "dreamplace_full_outline_balanced_white_fine",
                    "dreamplace_full_outline_balanced_white_strong",
                    "dreamplace_full_outline_balanced_white_local",
                    "dreamplace_full_outline_balanced_white_local_push",
                    "dreamplace_full_outline_balanced_white_local_fine",
                    "dreamplace_full_outline_balanced_white_local_soft",
                    "dreamplace_full_outline_balanced_route_channel",
                    "dreamplace_full_outline_dense_sharp_white",
                    "dreamplace_full_outline_loose_white",
                    "dreamplace_full_outline_flat_white",
                    "dreamplace_full_balanced_random",
                    "dreamplace_full_balanced_random_white",
                },
                "outline_only": {"outline_legal"},
                "ensemble": (
                    {"outline_legal", "dreamplace_full_outline_balanced_white_local_soft"}
                    | {
                        f"dreamplace_ensemble_{cfg_label}_s{seed_idx}"
                        for cfg_label in ("balanced", "dense_sharp", "loose")
                        for seed_idx in range(4)
                    }
                ),
                "rudy": {
                    "outline_legal",
                    "outline_rudy_refine",
                    "dreamplace_full_outline_balanced_white_local_soft",
                    "dreamplace_full_outline_balanced_white_local_soft_rudy",
                },
            }.get(profile_mode)
            if allowed_names is not None:
                candidate_specs = [spec for spec in candidate_specs if spec["name"] in allowed_names]
        if allowed_only:
            candidate_specs = [spec for spec in candidate_specs if spec["name"] in allowed_only]
        skip_remaining_candidate_build = bool(allowed_only) and allowed_only.issubset(
            {spec["name"] for spec in candidate_specs}
        )
        if os.environ.get("PARTCL_ENABLE_OUTLINE_EXTRAS", "0") == "1":
            candidate_specs.append(
                {
                    "name": "outline_micro",
                    "direct_placement": True,
                    "base_name": "outline_legal",
                    "micro_refine": True,
                    "placement": outline_placement.clone(),
                }
            )
            candidate_specs.append(
                {
                    "name": "outline_soft",
                    "direct_placement": True,
                    "soft_refine": True,
                    "placement": outline_placement.clone(),
                }
            )
            candidate_specs.append(
                {
                    "name": "outline_soft_local",
                    "direct_placement": True,
                    "local_soft_refine": True,
                    "placement": outline_placement.clone(),
                }
            )
            candidate_specs.append(
                {
                    "name": "outline_soft_search_plus",
                    "direct_placement": True,
                    "soft_search_plus_refine": True,
                    "placement": outline_placement.clone(),
                }
            )
            candidate_specs.append(
                {
                    "name": "outline_incre",
                    "direct_placement": True,
                    "incre_refine": True,
                    "placement": outline_placement.clone(),
                }
            )
            candidate_specs.append(
                {
                    "name": "outline_replite",
                    "direct_placement": True,
                    "replite_refine": True,
                    "placement": outline_placement.clone(),
                }
            )
        candidate_specs.append(
            {
                "name": "initial_anchor",
                "seed": self._legalize_hard_numpy(hard_base.copy(), benchmark),
                "cfg": {
                    "steps": max(12, int(round(20 * size_scale))),
                    "graph_w": 0.85,
                    "anchor_w": 0.35,
                    "base_w": 0.25,
                    "repulsion_w": 0.08,
                    "legalize_every": 5,
                    "lr": 0.45,
                    "repulsion_period": repulsion_period,
                    "density_weight": 1.0e-4,
                    "gamma": 4.0,
                    "target_density": 0.58 if complexity <= 320 else 0.60,
                },
            }
        )
        candidate_specs.append(
            {
                "name": "outline_anchor",
                "seed": self._legalize_hard_numpy(outline_hard.copy(), benchmark),
                "base_pos": outline_hard.copy(),
                "cfg": {
                    "steps": max(10, int(round(18 * size_scale))),
                    "graph_w": 0.52,
                    "anchor_w": 0.72,
                    "base_w": 0.38,
                    "repulsion_w": 0.18,
                    "legalize_every": 4,
                    "lr": 0.24,
                    "repulsion_period": 1,
                    "density_weight": 1.8e-4 if complexity <= 320 else 1.4e-4,
                    "gamma": 5.0 if complexity <= 320 else 4.6,
                    "target_density": 0.54 if complexity <= 320 else 0.57 if complexity <= 430 else 0.60,
                },
            }
        )

        if complexity <= 320:
            portfolio_defs = [
                (
                    4,
                    "center",
                    {
                        "imbalance": 0.03,
                        "objective": "km1",
                    },
                    {
                        "steps": 26,
                        "graph_w": 1.10,
                        "anchor_w": 0.40,
                        "base_w": 0.12,
                        "repulsion_w": 0.10,
                        "legalize_every": 4,
                        "lr": 0.52,
                    },
                ),
                (
                    4,
                    "boundary",
                    {
                        "imbalance": 0.05,
                        "objective": "cut",
                    },
                    {
                        "steps": 28,
                        "graph_w": 1.20,
                        "anchor_w": 0.36,
                        "base_w": 0.10,
                        "repulsion_w": 0.11,
                        "legalize_every": 4,
                        "lr": 0.55,
                    },
                ),
                (
                    8,
                    "center",
                    {
                        "imbalance": 0.05,
                        "objective": "km1",
                    },
                    {
                        "steps": 30,
                        "graph_w": 1.35,
                        "anchor_w": 0.30,
                        "base_w": 0.08,
                        "repulsion_w": 0.12,
                        "legalize_every": 4,
                        "lr": 0.58,
                    },
                ),
                (
                    8,
                    "boundary",
                    {
                        "imbalance": 0.08,
                        "objective": "km1",
                    },
                    {
                        "steps": 32,
                        "graph_w": 1.45,
                        "anchor_w": 0.28,
                        "base_w": 0.06,
                        "repulsion_w": 0.13,
                        "legalize_every": 4,
                        "lr": 0.60,
                    },
                ),
                (
                    16,
                    "center",
                    {
                        "imbalance": 0.08,
                        "objective": "km1",
                    },
                    {
                        "steps": 30,
                        "graph_w": 1.38,
                        "anchor_w": 0.24,
                        "base_w": 0.05,
                        "repulsion_w": 0.14,
                        "legalize_every": 4,
                        "lr": 0.57,
                    },
                ),
                (
                    16,
                    "boundary",
                    {
                        "imbalance": 0.10,
                        "objective": "cut",
                    },
                    {
                        "steps": 34,
                        "graph_w": 1.50,
                        "anchor_w": 0.22,
                        "base_w": 0.04,
                        "repulsion_w": 0.15,
                        "legalize_every": 4,
                        "lr": 0.59,
                    },
                ),
            ]
        elif complexity <= 430:
            portfolio_defs = [
                (
                    4,
                    "center",
                    {
                        "imbalance": 0.03,
                        "objective": "km1",
                    },
                    {
                        "steps": 24,
                        "graph_w": 1.12,
                        "anchor_w": 0.40,
                        "base_w": 0.10,
                        "repulsion_w": 0.10,
                        "legalize_every": 5,
                        "lr": 0.52,
                    },
                ),
                (
                    4,
                    "boundary",
                    {
                        "imbalance": 0.05,
                        "objective": "cut",
                    },
                    {
                        "steps": 26,
                        "graph_w": 1.22,
                        "anchor_w": 0.34,
                        "base_w": 0.08,
                        "repulsion_w": 0.11,
                        "legalize_every": 5,
                        "lr": 0.56,
                    },
                ),
                (
                    8,
                    "center",
                    {
                        "imbalance": 0.05,
                        "objective": "km1",
                    },
                    {
                        "steps": 28,
                        "graph_w": 1.34,
                        "anchor_w": 0.28,
                        "base_w": 0.06,
                        "repulsion_w": 0.12,
                        "legalize_every": 5,
                        "lr": 0.58,
                    },
                ),
                (
                    8,
                    "boundary",
                    {
                        "imbalance": 0.08,
                        "objective": "km1",
                    },
                    {
                        "steps": 30,
                        "graph_w": 1.42,
                        "anchor_w": 0.24,
                        "base_w": 0.05,
                        "repulsion_w": 0.13,
                        "legalize_every": 5,
                        "lr": 0.60,
                    },
                ),
                (
                    16,
                    "center",
                    {
                        "imbalance": 0.10,
                        "objective": "km1",
                    },
                    {
                        "steps": 28,
                        "graph_w": 1.38,
                        "anchor_w": 0.22,
                        "base_w": 0.04,
                        "repulsion_w": 0.13,
                        "legalize_every": 5,
                        "lr": 0.57,
                    },
                ),
            ]
        else:
            portfolio_defs = [
                (
                    4,
                    "boundary",
                    {
                        "imbalance": 0.05,
                        "objective": "cut",
                    },
                    {
                        "steps": 22,
                        "graph_w": 1.18,
                        "anchor_w": 0.34,
                        "base_w": 0.08,
                        "repulsion_w": 0.10,
                        "legalize_every": 6,
                        "lr": 0.54,
                    },
                ),
                (
                    8,
                    "center",
                    {
                        "imbalance": 0.08,
                        "objective": "km1",
                    },
                    {
                        "steps": 24,
                        "graph_w": 1.30,
                        "anchor_w": 0.28,
                        "base_w": 0.05,
                        "repulsion_w": 0.11,
                        "legalize_every": 6,
                        "lr": 0.57,
                    },
                ),
                (
                    16,
                    "center",
                    {
                        "imbalance": 0.10,
                        "objective": "km1",
                    },
                    {
                        "steps": 24,
                        "graph_w": 1.36,
                        "anchor_w": 0.22,
                        "base_w": 0.04,
                        "repulsion_w": 0.12,
                        "legalize_every": 6,
                        "lr": 0.56,
                    },
                ),
            ]
        if skip_remaining_candidate_build:
            portfolio_defs = []

        for partitions, variant, partition_cfg, cfg in portfolio_defs:
            cfg = cfg.copy()
            cfg["steps"] = max(12, int(round(cfg["steps"] * size_scale)))
            cfg["repulsion_period"] = repulsion_period
            cfg.setdefault("density_weight", 1.2e-4 if complexity <= 320 else 1.0e-4)
            cfg.setdefault("gamma", 4.2 if complexity <= 320 else 4.0)
            cfg.setdefault("target_density", 0.58 if complexity <= 320 else 0.60 if complexity <= 430 else 0.62)
            seed = self._build_partition_seed(
                benchmark,
                graph,
                num_parts=partitions,
                variant=variant,
                partition_cfg=partition_cfg,
                rng=rng,
            )
            candidate_specs.append(
                {
                    "name": (
                        f"partition_{partitions}_{variant}_"
                        f"{partition_cfg['objective']}_e{int(round(100 * partition_cfg['imbalance']))}"
                    ),
                    "seed": seed,
                    "cfg": cfg,
                }
            )

            if partitions == 4 and variant == "boundary" and partition_cfg["objective"] == "cut":
                macrobase_cfg = cfg.copy()
                macrobase_cfg.update(
                    {
                        "allow_flat_fallback": 0,
                        "macro_place_flag": 1,
                        "use_bb": 1,
                        "omit_soft_macros": 1,
                        "single_stage": 1,
                        "enable_fillers": 0,
                        "adjust_rudy_area_flag": 0,
                        "adjust_pin_area_flag": 0,
                        "max_num_area_adjust": 0,
                        "steps": 14 if complexity <= 320 else 12,
                        "target_density": 0.58 if complexity <= 320 else 0.60,
                        "density_weight": 1.5e-4 if complexity <= 320 else 1.0e-4,
                        "gamma": 4.4 if complexity <= 320 else 4.0,
                        "lr": 0.15 if complexity <= 320 else 0.12,
                        "stop_overflow": 0.50 if complexity <= 320 else 0.55,
                        "macro_halo_scale": 0.04,
                        "gp_noise_ratio": 0.010,
                        "random_center_init_flag": 0,
                    }
                )
                macro_variants = [
                    ("", macrobase_cfg),
                    (
                        "_spread",
                        {
                            **macrobase_cfg,
                            "target_density": max(
                                0.54,
                                float(macrobase_cfg["target_density"]) - (0.03 if complexity <= 320 else 0.02),
                            ),
                            "density_weight": float(macrobase_cfg["density_weight"]) * 1.35,
                            "gamma": float(macrobase_cfg["gamma"]) + 0.35,
                            "stop_overflow": max(0.44, float(macrobase_cfg["stop_overflow"]) - 0.04),
                            "macro_halo_scale": float(macrobase_cfg["macro_halo_scale"]) + 0.015,
                        },
                    ),
                    (
                        "_gentle",
                        {
                            **macrobase_cfg,
                            "lr": float(macrobase_cfg["lr"]) * 0.82,
                            "gamma": max(3.6, float(macrobase_cfg["gamma"]) - 0.25),
                            "steps": int(macrobase_cfg["steps"]) + 2,
                            "stop_overflow": min(0.60, float(macrobase_cfg["stop_overflow"]) + 0.03),
                            "gp_noise_ratio": 0.0,
                        },
                    ),
                ]
                for variant_suffix, variant_cfg in macro_variants:
                    candidate_specs.append(
                        {
                            "name": (
                                f"macrobase_{partitions}_{variant}_"
                                f"{partition_cfg['objective']}_e{int(round(100 * partition_cfg['imbalance']))}"
                                f"{variant_suffix}"
                            ),
                            "seed": seed.copy(),
                            "cfg": variant_cfg.copy(),
                        }
                    )
                    candidate_specs.append(
                        {
                            "name": (
                                f"macrobase_{partitions}_{variant}_"
                                f"{partition_cfg['objective']}_e{int(round(100 * partition_cfg['imbalance']))}"
                                f"{variant_suffix}_softsearch"
                            ),
                            "seed": seed.copy(),
                            "cfg": variant_cfg.copy(),
                            "exact_soft_search": True,
                        }
                    )

            if os.environ.get("PARTCL_ENABLE_OUTLINE_GUIDED", "0") == "1" and (
                (complexity <= 430 and partitions in (4, 8))
                or (
                    complexity > 430
                    and (
                        (partitions == 4 and variant == "boundary")
                        or (partitions in (8, 16) and variant == "center")
                    )
                )
            ):
                guided_seed = self._build_outline_guided_seed(
                    benchmark=benchmark,
                    outline_hard=outline_hard,
                    guide_seed=seed,
                    guide_mix=(
                        0.12 if complexity <= 320 and variant == "boundary"
                        else 0.09 if complexity <= 430
                        else 0.06 if variant == "boundary"
                        else 0.05
                    ),
                    repulsion_weight=0.22 if complexity <= 320 else 0.18 if complexity <= 430 else 0.16,
                )
                guided_cfg = cfg.copy()
                guided_cfg.update(
                    {
                        "graph_w": float(cfg["graph_w"]) * 0.72,
                        "anchor_w": max(float(cfg["anchor_w"]), 0.58),
                        "base_w": max(float(cfg["base_w"]), 0.28),
                        "repulsion_w": max(float(cfg["repulsion_w"]), 0.16),
                        "lr": float(cfg["lr"]) * 0.72,
                        "legalize_every": min(int(cfg["legalize_every"]), 4),
                        "density_weight": max(float(cfg["density_weight"]), 1.6e-4 if complexity <= 320 else 1.2e-4),
                        "gamma": max(float(cfg["gamma"]), 4.8 if complexity <= 320 else 4.4),
                        "target_density": min(
                            float(cfg["target_density"]),
                            0.54 if complexity <= 320 else 0.57 if complexity <= 430 else 0.60,
                        ),
                    }
                )
                candidate_specs.append(
                    {
                        "name": (
                            f"outline_guided_{partitions}_{variant}_"
                            f"{partition_cfg['objective']}_e{int(round(100 * partition_cfg['imbalance']))}"
                        ),
                        "seed": guided_seed,
                        "base_pos": outline_hard.copy(),
                        "cfg": guided_cfg,
                    }
                )


            variant_specs = []
            if (
                complexity <= 320
                and partitions in (8, 16)
                and variant == "center"
                and partition_cfg["objective"] == "km1"
            ):
                spread_cfg = cfg.copy()
                spread_cfg.update(
                    {
                        "target_density": max(0.50, float(cfg["target_density"]) - 0.05),
                        "density_weight": float(cfg["density_weight"]) * 2.5,
                        "gamma": float(cfg["gamma"]) + 1.5,
                        "gp_noise_ratio": 0.02,
                        "stop_overflow": 0.06,
                        "num_bins_scale": 0.85,
                    }
                )
                variant_specs.append(("spread", spread_cfg))

            if (
                os.environ.get("PARTCL_ENABLE_EXPERIMENTAL_DREAMPLACE", "0") == "1"
                and (
                complexity <= 320
                and partitions in (4, 8)
                and (
                    (variant == "boundary" and partition_cfg["objective"] == "cut")
                    or (variant == "center" and partition_cfg["objective"] == "km1")
                )
                )
            ):
                route_cfg = cfg.copy()
                route_cfg.update(
                    {
                        "target_density": max(0.49, float(cfg["target_density"]) - 0.06),
                        "density_weight": float(cfg["density_weight"]) * 3.0,
                        "gamma": float(cfg["gamma"]) + 2.0,
                        "gp_noise_ratio": 0.025,
                        "stop_overflow": 0.05,
                        "num_bins_scale": 1.10,
                        "lr": float(cfg["lr"]) * 0.92,
                    }
                )
                variant_specs.append(("route", route_cfg))

                dp_halo_cfg = cfg.copy()
                dp_halo_cfg.update(
                    {
                        "target_density": max(0.52, float(cfg["target_density"]) - 0.04),
                        "density_weight": float(cfg["density_weight"]) * 2.2,
                        "gamma": float(cfg["gamma"]) + 1.2,
                        "gp_noise_ratio": 0.018,
                        "stop_overflow": 0.07,
                        "stage1_bins_scale": 0.75,
                        "stage2_bins_scale": 1.15,
                        "stage1_lr_scale": 1.0,
                        "stage2_lr_scale": 0.60,
                        "two_stage_density_scaler": 120.0,
                        "macro_halo_scale": 0.08,
                        "enable_fillers": 1,
                        "random_center_init_flag": 1,
                    }
                )
                variant_specs.append(("dp_halo", dp_halo_cfg))

            if (
                os.environ.get("PARTCL_ENABLE_EXPERIMENTAL_DREAMPLACE", "0") == "1"
                and (
                complexity > 320
                and partitions in (4, 8)
                and (
                    (variant == "boundary" and partition_cfg["objective"] == "cut")
                    or (variant == "center" and partition_cfg["objective"] == "km1")
                )
                )
            ):
                route_cfg = cfg.copy()
                route_cfg.update(
                    {
                        "target_density": max(0.54, float(cfg["target_density"]) - 0.04),
                        "density_weight": float(cfg["density_weight"]) * 1.8,
                        "gamma": float(cfg["gamma"]) + 1.2,
                        "gp_noise_ratio": 0.015,
                        "stop_overflow": 0.07,
                        "num_bins_scale": 1.08,
                        "lr": float(cfg["lr"]) * 0.95,
                    }
                )
                variant_specs.append(("route", route_cfg))

                dp_halo_cfg = cfg.copy()
                dp_halo_cfg.update(
                    {
                        "target_density": max(0.55, float(cfg["target_density"]) - 0.03),
                        "density_weight": float(cfg["density_weight"]) * 1.7,
                        "gamma": float(cfg["gamma"]) + 0.9,
                        "gp_noise_ratio": 0.014,
                        "stop_overflow": 0.08,
                        "stage1_bins_scale": 0.8,
                        "stage2_bins_scale": 1.12,
                        "stage2_lr_scale": 0.65,
                        "two_stage_density_scaler": 160.0,
                        "macro_halo_scale": 0.06,
                        "enable_fillers": 1,
                        "random_center_init_flag": 1,
                    }
                )
                variant_specs.append(("dp_halo", dp_halo_cfg))

            if (
                complexity <= 430
                and partitions == 8
                and variant == "center"
                and partition_cfg["objective"] == "km1"
            ):
                softspread_cfg = cfg.copy()
                softspread_cfg.update(
                    {
                        "target_density": max(0.52, float(cfg["target_density"]) - 0.04),
                        "density_weight": float(cfg["density_weight"]) * 1.8,
                        "gamma": float(cfg["gamma"]) + 1.0,
                        "gp_noise_ratio": 0.015,
                        "stop_overflow": 0.07,
                    }
                )
                variant_specs.append(("softspread", softspread_cfg))

            if (
                complexity > 430
                and partitions == 8
                and variant == "center"
                and partition_cfg["objective"] == "km1"
            ):
                softspread_cfg = cfg.copy()
                softspread_cfg.update(
                    {
                        "target_density": max(0.55, float(cfg["target_density"]) - 0.03),
                        "density_weight": float(cfg["density_weight"]) * 1.5,
                        "gamma": float(cfg["gamma"]) + 0.8,
                        "gp_noise_ratio": 0.012,
                        "stop_overflow": 0.08,
                    }
                )
                variant_specs.append(("softspread", softspread_cfg))

            for suffix, variant_cfg in variant_specs:
                candidate_specs.append(
                    {
                        "name": (
                            f"partition_{partitions}_{variant}_"
                            f"{partition_cfg['objective']}_e{int(round(100 * partition_cfg['imbalance']))}_{suffix}"
                        ),
                        "seed": seed.copy(),
                        "cfg": variant_cfg,
                    }
                )
        if profile_mode:
            allowed_names = {
                "outline": {"outline_legal", "outline_soft_search", "rebalance_first_search", "corridor_first_search"},
                "outline_torch": {"outline_legal", "outline_soft_search", "rebalance_first_search", "corridor_first_search", "outline_softopt_torch"},
                "dreamplace_basin": {
                    "outline_legal",
                    "dreamplace_full_outline_flat_rebalance",
                    "dreamplace_full_outline_balanced",
                    "dreamplace_full_outline_dense_sharp",
                    "dreamplace_full_outline_loose",
                },
                "dreamplace_basin_rebalance": {
                    "outline_legal",
                    "dreamplace_full_outline_flat_rebalance",
                    "dreamplace_full_outline_balanced",
                    "dreamplace_full_outline_dense_sharp",
                    "dreamplace_full_outline_loose",
                },
                "dreamplace_basin_deep": {
                    "outline_legal",
                    "dreamplace_full_outline_flat_rebalance",
                    "dreamplace_full_outline_balanced",
                    "dreamplace_full_outline_dense_sharp",
                    "dreamplace_full_outline_loose",
                    "dreamplace_full_outline_balanced_g6",
                    "dreamplace_full_outline_balanced_bins",
                    "dreamplace_full_outline_balanced_long",
                    "dreamplace_full_outline_mid_dense",
                    "dreamplace_full_outline_mid_loose",
                    "dreamplace_full_outline_zero_noise",
                    "dreamplace_full_outline_route_diag",
                    "dreamplace_full_outline_balanced_routepolish",
                    "dreamplace_full_outline_balanced_channelpolish",
                    "dreamplace_full_outline_balanced_whitepolish",
                    "dreamplace_full_outline_balanced_white2",
                    "dreamplace_full_outline_balanced_white3",
                    "dreamplace_full_outline_balanced_white_route",
                    "dreamplace_full_outline_balanced_white_channel",
                    "dreamplace_full_outline_balanced_white_exactsoft",
                    "dreamplace_full_outline_balanced_white_fine",
                    "dreamplace_full_outline_balanced_white_strong",
                    "dreamplace_full_outline_balanced_white_local",
                    "dreamplace_full_outline_balanced_white_local_push",
                    "dreamplace_full_outline_balanced_white_local_fine",
                    "dreamplace_full_outline_balanced_white_local_soft",
                    "dreamplace_full_outline_balanced_route_channel",
                    "dreamplace_full_outline_dense_sharp_white",
                    "dreamplace_full_outline_loose_white",
                    "dreamplace_full_outline_flat_white",
                    "dreamplace_full_balanced_random",
                    "dreamplace_full_balanced_random_white",
                },
                "outline_only": {"outline_legal"},
                "ensemble": (
                    {"outline_legal", "dreamplace_full_outline_balanced_white_local_soft"}
                    | {
                        f"dreamplace_ensemble_{cfg_label}_s{seed_idx}"
                        for cfg_label in ("balanced", "dense_sharp", "loose")
                        for seed_idx in range(4)
                    }
                ),
                "rudy": {
                    "outline_legal",
                    "outline_rudy_refine",
                    "dreamplace_full_outline_balanced_white_local_soft",
                    "dreamplace_full_outline_balanced_white_local_soft_rudy",
                },
            }.get(profile_mode)
            if allowed_names is not None:
                candidate_specs = [spec for spec in candidate_specs if spec["name"] in allowed_names]
        if allowed_only:
            candidate_specs = [spec for spec in candidate_specs if spec["name"] in allowed_only]
        if not candidate_specs:
            if allowed_only:
                requested = ",".join(sorted(allowed_only))
                raise RuntimeError(f"No candidates selected by PARTCL_ONLY_CANDIDATES={requested}")
            raise RuntimeError("No candidate placements were generated")

        candidate_results = []
        self._log_backend_summary(benchmark, phase="start")

        for spec in candidate_specs:
            print(f"[partcl:candidate] {benchmark.name} start {spec['name']}", flush=True)
            candidate_start = time.perf_counter()
            if spec.get("dreamplace_full_placement"):
                refine_backend = "dreamplace_full"
                dp_macro_pos = self._run_dreamplace_full_backend(
                    seed=spec["seed"],
                    benchmark=benchmark,
                    plc=plc,
                    cfg=spec["cfg"],
                )
                if dp_macro_pos is None:
                    refine_backend = "dreamplace_full_failed"
                    placement = spec.get("base_placement", outline_placement).clone()
                else:
                    placement = spec.get("base_placement", base).clone()
                    placement[: benchmark.num_macros] = torch.tensor(
                        dp_macro_pos,
                        dtype=placement.dtype,
                    )
                    hard_pos = self._legalize_hard_numpy(
                        placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64),
                        benchmark,
                    )
                    placement[: benchmark.num_hard_macros] = torch.tensor(
                        hard_pos,
                        dtype=placement.dtype,
                    )
                if spec.get("rebalance_first_refine") and plc is not None:
                    placement = self._rebalance_first_refine(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                if spec.get("dreamplace_route_polish") and plc is not None:
                    placement = self._outline_congestion_soft_refine_strong(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                if spec.get("dreamplace_channel_polish") and plc is not None:
                    placement = self._outline_channel_soft_refine(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                if spec.get("dreamplace_white_polish") and plc is not None:
                    white_cfg = dict(spec.get("dreamplace_white_polish_cfg", {}))
                    for _ in range(int(spec.get("dreamplace_white_polish_rounds", 1))):
                        placement = self._outline_whitespace_soft_refine(
                            placement=placement,
                            benchmark=benchmark,
                            graph=graph,
                            plc=plc,
                            **white_cfg,
                        )
                if spec.get("dreamplace_exact_soft_polish") and plc is not None:
                    placement = self._exact_soft_hotspot_polish(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                if spec.get("dreamplace_rudy_polish") and plc is not None:
                    rudy_rounds = int(spec.get("dreamplace_rudy_polish_rounds", 1))
                    for _ in range(max(1, rudy_rounds)):
                        new_placement = self._outline_rudy_refine(
                            placement=placement,
                            benchmark=benchmark,
                            graph=graph,
                            plc=plc,
                        )
                        if new_placement is placement:
                            break
                        placement = new_placement
            elif spec.get("direct_placement"):
                refine_backend = "baseline"
                if spec["name"] == "outline_legal":
                    placement = spec.get("placement", outline_placement).clone()
                elif spec.get("micro_refine"):
                    placement = self._outline_micro_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("soft_refine"):
                    placement = self._outline_soft_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                    )
                elif spec.get("route_refine"):
                    placement = self._outline_route_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("local_soft_refine"):
                    placement = self._outline_soft_local_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("iterated_soft_search_refine"):
                    placement = self._iterated_soft_search_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("rebalance_first_refine"):
                    placement = self._rebalance_first_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("corridor_first_refine"):
                    placement = self._corridor_first_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("rudy_refine"):
                    seed_placement = spec.get("placement", outline_placement).clone()
                    rudy_rounds = int(spec.get("rudy_refine_rounds", 2))
                    placement = seed_placement
                    for _ in range(max(1, rudy_rounds)):
                        new_placement = self._outline_rudy_refine(
                            placement=placement,
                            benchmark=benchmark,
                            graph=graph,
                            plc=plc,
                        )
                        if new_placement is placement:
                            break
                        placement = new_placement
                elif spec.get("soft_search_refine"):
                    placement = self._macrobase_soft_search(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("softopt_torch_refine"):
                    placement = self._outline_softopt_torch_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("soft_search_plus_refine"):
                    placement = self._outline_soft_search_plus(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("incre_refine"):
                    placement = self._outline_incre_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("replite_refine"):
                    placement = self._outline_replite_refine(
                        placement=spec.get("placement", outline_placement).clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                else:
                    placement = base.clone()
                    hard_pos = self._legalize_hard_numpy(
                        placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64),
                        benchmark,
                    )
                    placement[: benchmark.num_hard_macros] = torch.tensor(
                        hard_pos,
                        dtype=placement.dtype,
                    )
                    if not spec.get("preserve_soft", False):
                        placement = self._follow_soft_macros(
                            placement=placement,
                            benchmark=benchmark,
                            graph=graph,
                            iterations=1 if complexity > 450 else 2 if complexity > 320 else 3,
                        )
            else:
                hard_pos, refine_backend = self._run_dreamplace_or_fallback(
                    seed=spec["seed"],
                    base_pos=spec.get("base_pos", hard_base),
                    benchmark=benchmark,
                    graph=graph,
                    plc=plc,
                    cfg=spec["cfg"],
                )

                placement = base.clone()
                placement[: benchmark.num_hard_macros] = torch.tensor(
                    hard_pos,
                    dtype=placement.dtype,
                )
                if spec.get("exact_soft_search") and plc is not None:
                    placement = self._macrobase_soft_search(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("local_soft_post") and plc is not None:
                    placement = self._outline_soft_local_refine(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                else:
                    placement = self._follow_soft_macros(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        iterations=1 if complexity > 450 else 2 if complexity > 320 else 3,
                    )

                hard_pos = self._legalize_hard_numpy(
                    placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64),
                    benchmark,
                )
                placement[: benchmark.num_hard_macros] = torch.tensor(
                    hard_pos,
                    dtype=placement.dtype,
                )
                if spec.get("exact_soft_search") and plc is not None:
                    placement = self._macrobase_soft_search(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                elif spec.get("local_soft_post") and plc is not None:
                    placement = self._outline_soft_local_refine(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                    )
                else:
                    placement = self._follow_soft_macros(
                        placement=placement,
                        benchmark=benchmark,
                        graph=graph,
                        iterations=1 if complexity > 320 else 2,
                    )
            surrogate = self._surrogate_score(placement, benchmark, graph)
            candidate_results.append(
                {
                    "name": spec["name"],
                    "placement": placement,
                    "surrogate": surrogate,
                    "refine_backend": refine_backend,
                    "force_exact": bool(spec.get("direct_placement")),
                    "runtime_s": time.perf_counter() - candidate_start,
                }
            )
            print(
                f"[partcl:candidate] {benchmark.name} done {spec['name']} "
                f"backend={refine_backend} surrogate={surrogate:.4f} "
                f"runtime={candidate_results[-1]['runtime_s']:.2f}s"
            )

        if plc is None:
            best = min(candidate_results, key=lambda item: item["surrogate"])
            self._log_backend_summary(benchmark, phase="final")
            self._log_exact_cache_summary(benchmark)
            print(
                f"[partcl:select] {benchmark.name} winner={best['name']} "
                f"surrogate={best['surrogate']:.4f}"
            )
            return best["placement"]

        if complexity <= 320 or len(candidate_results) <= 8:
            shortlist = sorted(candidate_results, key=lambda item: item["surrogate"])
        else:
            top_k = 3 if complexity > 450 else 4 if complexity > 320 else min(len(candidate_results), 7)
            shortlist = sorted(candidate_results, key=lambda item: item["surrogate"])[:top_k]
        dc_candidates = sorted(
            candidate_results,
            key=lambda item: self._surrogate_dc_score(item["placement"], benchmark, graph),
        )[:2 if complexity <= 320 else 1]
        cong_candidates = sorted(
            candidate_results,
            key=lambda item: self._surrogate_congestion_score(item["placement"], benchmark, graph),
        )[:1]
        if dc_candidates or cong_candidates:
            by_name = {item["name"]: item for item in shortlist}
            for item in dc_candidates + cong_candidates:
                by_name[item["name"]] = item
            shortlist = list(by_name.values())
        forced = [item for item in candidate_results if item.get("force_exact")]
        if forced:
            by_name = {item["name"]: item for item in shortlist}
            for item in forced:
                by_name[item["name"]] = item
            shortlist = list(by_name.values())

        best_score = float("inf")
        best_placement = shortlist[0]["placement"]
        best_name = shortlist[0]["name"]
        for item in shortlist:
            score = self._exact_score(item["placement"], benchmark, plc)
            print(
                f"[partcl:select] {benchmark.name} exact {item['name']} "
                f"surrogate={item['surrogate']:.4f} proxy={score:.4f}"
            )
            if score < best_score:
                best_score = score
                best_placement = item["placement"]
                best_name = item["name"]

        refine_rounds = 2 if complexity <= 320 else 1
        refine_budget = 3 if complexity <= 320 else 2 if complexity <= 430 else 1
        skip_post_refine = bool(allowed_only) or os.environ.get("PARTCL_SKIP_POST_REFINE", "0") == "1"
        refine_pool = [] if skip_post_refine else shortlist[:refine_budget]
        for item in refine_pool:
            refined = self._post_refine_candidate(
                placement=item["placement"],
                benchmark=benchmark,
                graph=graph,
                plc=plc,
                rounds=refine_rounds,
            )
            refined_score = self._exact_score(refined, benchmark, plc)
            print(
                f"[partcl:refine] {benchmark.name} {item['name']} "
                f"proxy={refined_score:.4f}"
            )
            if refined_score < best_score:
                best_score = refined_score
                best_placement = refined
                best_name = f"{item['name']}_refined"

        self._log_backend_summary(benchmark, phase="final")
        self._log_exact_cache_summary(benchmark)
        print(
            f"[partcl:select] {benchmark.name} winner={best_name} proxy={best_score:.4f}"
        )
        return best_placement

    def _log_backend_summary(self, benchmark: Benchmark, phase: str) -> None:
        partition_summary = ", ".join(
            f"k={k}:{v}" for k, v in sorted(self._run_state.get("partition_backend", {}).items())
        )
        if not partition_summary:
            partition_summary = "none"
        print(
            "[partcl] "
            f"{benchmark.name} {phase}: dreamplace={self._run_state.get('dreamplace_backend', 'unknown')} "
            f"partitioner={partition_summary}"
        )

    def _stable_name_seed(self, name: str) -> int:
        return sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))

    def _surrogate_dc_score(self, placement: torch.Tensor, benchmark: Benchmark, graph: dict) -> float:
        del graph
        return float(self._coarse_proxy_score(placement, benchmark))

    def _surrogate_congestion_score(self, placement: torch.Tensor, benchmark: Benchmark, graph: dict) -> float:
        del graph
        nrow = min(18, max(8, benchmark.grid_rows // 2))
        ncol = min(18, max(8, benchmark.grid_cols // 2))
        cell_w = benchmark.canvas_width / max(ncol, 1)
        cell_h = benchmark.canvas_height / max(nrow, 1)
        pos = placement.cpu().numpy().astype(np.float64)
        congestion = np.zeros((nrow, ncol), dtype=np.float64)
        port_start = benchmark.num_macros
        for net_idx, nodes in enumerate(benchmark.net_nodes):
            if len(nodes) < 2:
                continue
            pts = []
            for node in nodes.tolist():
                if node < benchmark.num_macros:
                    pts.append(pos[node])
                else:
                    port_idx = node - port_start
                    if 0 <= port_idx < benchmark.port_positions.shape[0]:
                        pts.append(benchmark.port_positions[port_idx].cpu().numpy().astype(np.float64))
            if len(pts) < 2:
                continue
            pts_arr = np.asarray(pts, dtype=np.float64)
            xmin, ymin = np.min(pts_arr, axis=0)
            xmax, ymax = np.max(pts_arr, axis=0)
            c0 = min(ncol - 1, max(0, int(xmin / max(cell_w, 1.0e-9))))
            c1 = min(ncol - 1, max(0, int(xmax / max(cell_w, 1.0e-9))))
            r0 = min(nrow - 1, max(0, int(ymin / max(cell_h, 1.0e-9))))
            r1 = min(nrow - 1, max(0, int(ymax / max(cell_h, 1.0e-9))))
            span = max((r1 - r0 + 1) * (c1 - c0 + 1), 1)
            weight = float(benchmark.net_weights[net_idx].item()) if net_idx < len(benchmark.net_weights) else 1.0
            congestion[r0 : r1 + 1, c0 : c1 + 1] += weight / span
        return float(np.mean(np.sort(congestion.reshape(-1))[-max(1, congestion.size // 20) :]))

    def _get_graph_cache(self, benchmark: Benchmark) -> dict:
        key = (
            benchmark.name,
            benchmark.num_macros,
            benchmark.num_nets,
            benchmark.num_hard_macros,
        )
        if key not in self._graph_cache:
            self._graph_cache[key] = self._build_graph_cache(benchmark)
        return self._graph_cache[key]

    def _build_graph_cache(self, benchmark: Benchmark) -> dict:
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        hard_base = benchmark.macro_positions[:num_hard].cpu().numpy().astype(np.float64)
        areas = (
            benchmark.macro_sizes[:num_hard, 0] * benchmark.macro_sizes[:num_hard, 1]
        ).cpu().numpy().astype(np.float64)

        edge_dict = {}
        hard_degree = np.zeros(num_hard, dtype=np.float64)
        soft_neighbors = [[] for _ in range(max(0, num_macros - num_hard))]
        soft_neighbor_cap = max(8, int(os.environ.get("PARTCL_SOFT_NEIGHBOR_CAP", "96")))

        for net_idx, nodes in enumerate(benchmark.net_nodes):
            node_list = nodes.tolist()
            if len(node_list) < 2:
                continue

            weight = float(benchmark.net_weights[net_idx].item()) if net_idx < len(benchmark.net_weights) else 1.0
            hard_nodes = [n for n in node_list if n < num_hard]

            if 2 <= len(hard_nodes) <= 24:
                pair_w = weight / max(1, len(hard_nodes) - 1)
                for i in range(len(hard_nodes)):
                    a = hard_nodes[i]
                    for j in range(i + 1, len(hard_nodes)):
                        b = hard_nodes[j]
                        key = (a, b) if a < b else (b, a)
                        edge_dict[key] = edge_dict.get(key, 0.0) + pair_w

            if len(node_list) > soft_neighbor_cap:
                neighbor_nodes = [n for n in node_list if n < num_hard]
                if len(neighbor_nodes) < soft_neighbor_cap:
                    for n in node_list:
                        if n >= num_hard and n < num_macros:
                            neighbor_nodes.append(n)
                            if len(neighbor_nodes) >= soft_neighbor_cap:
                                break
                neighbor_nodes = neighbor_nodes[:soft_neighbor_cap]
            else:
                neighbor_nodes = node_list
            node_weight = weight / max(1, len(neighbor_nodes) - 1)
            for node in node_list:
                if num_hard <= node < num_macros:
                    soft_idx = node - num_hard
                    for other in neighbor_nodes:
                        if other == node:
                            continue
                        soft_neighbors[soft_idx].append((other, node_weight))

        if edge_dict:
            hard_i = np.array([pair[0] for pair in edge_dict], dtype=np.int64)
            hard_j = np.array([pair[1] for pair in edge_dict], dtype=np.int64)
            hard_w = np.array([edge_dict[pair] for pair in edge_dict], dtype=np.float64)
            np.add.at(hard_degree, hard_i, hard_w)
            np.add.at(hard_degree, hard_j, hard_w)
        else:
            hard_i = np.zeros(0, dtype=np.int64)
            hard_j = np.zeros(0, dtype=np.int64)
            hard_w = np.zeros(0, dtype=np.float64)

        smooth_pos = hard_base.copy()
        if len(hard_w) > 0:
            for _ in range(3):
                nbr_sum = np.zeros_like(smooth_pos)
                nbr_w = np.zeros(num_hard, dtype=np.float64)
                np.add.at(nbr_sum, hard_i, smooth_pos[hard_j] * hard_w[:, None])
                np.add.at(nbr_sum, hard_j, smooth_pos[hard_i] * hard_w[:, None])
                np.add.at(nbr_w, hard_i, hard_w)
                np.add.at(nbr_w, hard_j, hard_w)
                mask = nbr_w > 0
                smooth_pos[mask] = 0.55 * smooth_pos[mask] + 0.45 * (
                    nbr_sum[mask] / nbr_w[mask, None]
                )

        return {
            "hard_i": hard_i,
            "hard_j": hard_j,
            "hard_w": hard_w,
            "hard_degree": hard_degree,
            "areas": areas,
            "smooth_pos": smooth_pos,
            "soft_neighbors": soft_neighbors,
            "num_ports": num_ports,
            "hard_hyperedges": self._collect_hard_hyperedges(benchmark, max_degree=100),
        }

    def _collect_hard_hyperedges(self, benchmark: Benchmark, max_degree: int) -> list[dict]:
        num_hard = benchmark.num_hard_macros
        hyperedges = []
        for net_idx, nodes in enumerate(benchmark.net_nodes):
            hard_nodes = sorted({int(node) for node in nodes.tolist() if int(node) < num_hard})
            if len(hard_nodes) < 2 or len(hard_nodes) > max_degree:
                continue
            weight = (
                float(benchmark.net_weights[net_idx].item())
                if net_idx < len(benchmark.net_weights)
                else 1.0
            )
            hyperedges.append({"nodes": hard_nodes, "weight": max(weight, 1.0e-6)})
        return hyperedges

    def _get_backend_status(self) -> dict:
        if self._backend_cache is not None:
            return self._backend_cache

        def _find_module(name: str) -> bool:
            try:
                return importlib.util.find_spec(name) is not None
            except ModuleNotFoundError:
                return False

        self._backend_cache = {
            "kahypar_module": _find_module("kahypar"),
            "mtkahypar_module": _find_module("mtkahypar"),
            "kahypar_binary": os.environ.get("PARTCL_KAHYPAR_BINARY") or shutil.which("kahypar"),
            "kahypar_ini": os.environ.get("PARTCL_KAHYPAR_INI"),
            "dreamplace_module": os.environ.get("PARTCL_DREAMPLACE_MODULE"),
        }
        return self._backend_cache

    def _build_partition_seed(
        self,
        benchmark: Benchmark,
        graph: dict,
        num_parts: int,
        variant: str,
        partition_cfg: dict | None,
        rng,
    ) -> np.ndarray:
        num_hard = benchmark.num_hard_macros
        base = benchmark.macro_positions[:num_hard].cpu().numpy().astype(np.float64)
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        areas = graph["areas"]
        degrees = graph["hard_degree"]
        smooth = graph["smooth_pos"]
        partition_ids = self._partition_hard_macros(
            benchmark=benchmark,
            graph=graph,
            num_parts=num_parts,
            imbalance=float((partition_cfg or {}).get("imbalance", 0.05)),
            objective=str((partition_cfg or {}).get("objective", "km1")),
        )

        leaves = []

        def recurse(indices, rect):
            if len(indices) <= 1:
                leaves.append((indices, rect))
                return

            parts_here = sorted({int(partition_ids[idx]) for idx in indices})
            if len(parts_here) <= 1:
                leaves.append((indices, rect))
                return

            left, bottom, right, top = rect
            axis = 0 if (right - left) >= (top - bottom) else 1
            part_keys = {
                part: float(
                    np.mean(
                        0.65 * smooth[[idx for idx in indices if partition_ids[idx] == part], axis]
                        + 0.35 * base[[idx for idx in indices if partition_ids[idx] == part], axis]
                    )
                )
                for part in parts_here
            }
            ordered_parts = sorted(parts_here, key=lambda part: part_keys[part])
            split_at = max(1, len(ordered_parts) // 2)
            left_parts = set(ordered_parts[:split_at])
            right_parts = set(ordered_parts[split_at:])
            left_indices = [idx for idx in indices if partition_ids[idx] in left_parts]
            right_indices = [idx for idx in indices if partition_ids[idx] in right_parts]
            if not left_indices or not right_indices:
                leaves.append((indices, rect))
                return

            total_area = sum(areas[idx] for idx in indices)
            area_frac = sum(areas[idx] for idx in left_indices) / max(total_area, 1e-9)
            area_frac = float(np.clip(area_frac, 0.3, 0.7))

            if axis == 0:
                split_coord = left + (right - left) * area_frac
                rect_a = (left, bottom, split_coord, top)
                rect_b = (split_coord, bottom, right, top)
            else:
                split_coord = bottom + (top - bottom) * area_frac
                rect_a = (left, bottom, right, split_coord)
                rect_b = (left, split_coord, right, top)

            recurse(left_indices, rect_a)
            recurse(right_indices, rect_b)

        all_indices = list(range(num_hard))
        recurse(all_indices, (0.0, 0.0, benchmark.canvas_width, benchmark.canvas_height))

        seed = base.copy()
        for indices, rect in leaves:
            if not indices:
                continue

            points = self._region_grid_points(
                rect=rect,
                count=len(indices),
                widths=widths[indices],
                heights=heights[indices],
            )
            if variant == "boundary":
                left, bottom, right, top = rect
                center = np.array([(left + right) * 0.5, (bottom + top) * 0.5])
                point_order = np.argsort(
                    [
                        min(
                            pt[0] - left,
                            right - pt[0],
                            pt[1] - bottom,
                            top - pt[1],
                        )
                        for pt in points
                    ]
                )
                macro_order = sorted(
                    indices,
                    key=lambda idx: (-(degrees[idx] + 0.25 * areas[idx]), np.linalg.norm(base[idx] - center)),
                )
            else:
                left, bottom, right, top = rect
                center = np.array([(left + right) * 0.5, (bottom + top) * 0.5])
                point_order = np.argsort([np.sum((pt - center) ** 2) for pt in points])
                macro_order = sorted(
                    indices,
                    key=lambda idx: (-(areas[idx] + 0.5 * degrees[idx]), np.linalg.norm(base[idx] - center)),
                )

            for macro_idx, point_idx in zip(macro_order, point_order):
                jitter = 0.02 * np.array(
                    [benchmark.canvas_width, benchmark.canvas_height],
                    dtype=np.float64,
                )
                point = points[point_idx].copy()
                point += rng.normal(0.0, 1.0, 2) * jitter / max(1, len(indices))
                seed[macro_idx, 0] = np.clip(point[0], widths[macro_idx] * 0.5, benchmark.canvas_width - widths[macro_idx] * 0.5)
                seed[macro_idx, 1] = np.clip(point[1], heights[macro_idx] * 0.5, benchmark.canvas_height - heights[macro_idx] * 0.5)

        return self._legalize_hard_numpy(seed, benchmark)

    def _build_outline_guided_seed(
        self,
        benchmark: Benchmark,
        outline_hard: np.ndarray,
        guide_seed: np.ndarray,
        guide_mix: float,
        repulsion_weight: float,
    ) -> np.ndarray:
        num_hard = benchmark.num_hard_macros
        pos = outline_hard.copy()
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        half_w = widths * 0.5
        half_h = heights * 0.5
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        center = np.array(
            [0.5 * float(benchmark.canvas_width), 0.5 * float(benchmark.canvas_height)],
            dtype=np.float64,
        )
        outline_delta = outline_hard - center[None, :]
        outline_norm = np.linalg.norm(outline_delta, axis=1, keepdims=True)
        outline_dir = np.divide(
            outline_delta,
            np.maximum(outline_norm, 1.0e-9),
            out=np.zeros_like(outline_delta),
        )

        pos[movable] += guide_mix * (guide_seed[movable] - outline_hard[movable])
        pos[movable] += 0.012 * canvas_scale * outline_dir[movable]

        for _ in range(4):
            grad = np.zeros_like(pos)
            grad += 0.92 * (pos - outline_hard)
            grad += 0.28 * (pos - guide_seed)
            grad += self._repulsion_gradient(
                pos=pos,
                widths=widths,
                heights=heights,
                movable=movable,
                weight=repulsion_weight,
            )
            grad[~movable] = 0.0
            pos[movable] -= 0.18 * grad[movable]
            pos[:, 0] = np.clip(pos[:, 0], half_w, benchmark.canvas_width - half_w)
            pos[:, 1] = np.clip(pos[:, 1], half_h, benchmark.canvas_height - half_h)
            pos = self._legalize_hard_numpy(pos, benchmark)

        return pos

    def _partition_hard_macros(
        self,
        benchmark: Benchmark,
        graph: dict,
        num_parts: int,
        imbalance: float,
        objective: str,
    ) -> np.ndarray:
        cache_key = (
            benchmark.name,
            benchmark.num_hard_macros,
            num_parts,
            round(float(imbalance), 4),
            objective.lower(),
        )
        cached = self._partition_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        parts = self._run_kahypar_partition(
            benchmark=benchmark,
            graph=graph,
            num_parts=num_parts,
            imbalance=imbalance,
            objective=objective,
        )
        if parts is not None and len(parts) == benchmark.num_hard_macros:
            result = parts.astype(np.int64, copy=False)
        else:
            result = self._recursive_partition_labels(benchmark, graph, num_parts)
        self._partition_cache[cache_key] = result.copy()
        return result

    def _recursive_partition_labels(self, benchmark: Benchmark, graph: dict, num_parts: int) -> np.ndarray:
        num_hard = benchmark.num_hard_macros
        labels = np.zeros(num_hard, dtype=np.int64)
        smooth = graph["smooth_pos"]
        base = benchmark.macro_positions[:num_hard].cpu().numpy().astype(np.float64)
        areas = graph["areas"]

        next_label = 0

        def recurse(indices, parts_left):
            nonlocal next_label
            if parts_left <= 1 or len(indices) <= 1:
                labels[indices] = next_label
                next_label += 1
                return

            left, bottom, right, top = self._bounding_rect(base[indices])
            axis = 0 if (right - left) >= (top - bottom) else 1
            keys = 0.65 * smooth[indices, axis] + 0.35 * base[indices, axis]
            order = np.argsort(keys)
            ordered = [indices[idx] for idx in order]

            total_area = sum(areas[idx] for idx in ordered)
            accum = 0.0
            split_at = max(1, len(ordered) // 2)
            best_gap = float("inf")
            for cut in range(1, len(ordered)):
                accum += areas[ordered[cut - 1]]
                gap = abs(accum - 0.5 * total_area)
                if gap < best_gap:
                    best_gap = gap
                    split_at = cut

            left_indices = ordered[:split_at]
            right_indices = ordered[split_at:]
            if not left_indices or not right_indices:
                labels[indices] = next_label
                next_label += 1
                return

            parts_a = parts_left // 2
            parts_b = parts_left - parts_a
            recurse(left_indices, parts_a)
            recurse(right_indices, parts_b)

        recurse(list(range(num_hard)), num_parts)
        return labels

    def _bounding_rect(self, pos: np.ndarray) -> tuple[float, float, float, float]:
        if len(pos) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(np.min(pos[:, 0])),
            float(np.min(pos[:, 1])),
            float(np.max(pos[:, 0])),
            float(np.max(pos[:, 1])),
        )

    def _run_kahypar_partition(
        self,
        benchmark: Benchmark,
        graph: dict,
        num_parts: int,
        imbalance: float,
        objective: str,
    ) -> np.ndarray | None:
        backends = self._get_backend_status()
        if not graph["hard_hyperedges"] or num_parts <= 1:
            return None

        if backends["mtkahypar_module"]:
            parts = self._run_mtkahypar_partition(
                benchmark=benchmark,
                hyperedges=graph["hard_hyperedges"],
                num_parts=num_parts,
                imbalance=imbalance,
                objective=objective,
            )
            if parts is not None:
                self._run_state.setdefault("partition_backend", {})[num_parts] = "mtkahypar"
                return parts

        if backends["kahypar_module"]:
            parts = self._run_kahypar_python_partition(
                benchmark=benchmark,
                hyperedges=graph["hard_hyperedges"],
                num_parts=num_parts,
                imbalance=imbalance,
                objective=objective,
            )
            if parts is not None:
                self._run_state.setdefault("partition_backend", {})[num_parts] = "kahypar_py"
                return parts

        if backends["kahypar_binary"] and backends["kahypar_ini"]:
            parts = self._run_kahypar_cli_partition(
                benchmark=benchmark,
                hyperedges=graph["hard_hyperedges"],
                num_parts=num_parts,
                imbalance=imbalance,
                objective=objective,
                binary=backends["kahypar_binary"],
                ini_path=backends["kahypar_ini"],
            )
            if parts is not None:
                self._run_state.setdefault("partition_backend", {})[num_parts] = "kahypar_cli"
                return parts

        self._run_state.setdefault("partition_backend", {})[num_parts] = "fallback"
        return None

    def _run_kahypar_python_partition(
        self,
        benchmark: Benchmark,
        hyperedges: list[dict],
        num_parts: int,
        imbalance: float,
        objective: str,
    ) -> np.ndarray | None:
        try:
            kahypar = importlib.import_module("kahypar")
            num_vertices = benchmark.num_hard_macros
            edge_indices = [0]
            flat_edges = []
            edge_weights = []
            for edge in hyperedges:
                flat_edges.extend([node for node in edge["nodes"]])
                edge_indices.append(len(flat_edges))
                edge_weights.append(max(1, int(round(edge["weight"]))))

            vertex_weights = np.maximum(
                1,
                np.rint(
                    (
                        benchmark.macro_sizes[:num_vertices, 0]
                        * benchmark.macro_sizes[:num_vertices, 1]
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    * 1000.0
                ).astype(np.int64),
            )
            hypergraph = kahypar.Hypergraph(
                num_vertices,
                len(hyperedges),
                edge_indices,
                flat_edges,
                num_parts,
                edge_weights,
                vertex_weights.tolist(),
            )
            context = kahypar.Context()
            context.setK(num_parts)
            context.setEpsilon(float(imbalance))
            context.suppressOutput(True)
            if hasattr(context, "setObjective"):
                context.setObjective(objective.upper())
            kahypar.partition(hypergraph, context)
            return np.asarray([hypergraph.blockID(i) for i in range(num_vertices)], dtype=np.int64)
        except Exception:
            return None

    def _run_mtkahypar_partition(
        self,
        benchmark: Benchmark,
        hyperedges: list[dict],
        num_parts: int,
        imbalance: float,
        objective: str,
    ) -> np.ndarray | None:
        try:
            mtkahypar = importlib.import_module("mtkahypar")
            initializer = None
            if hasattr(mtkahypar, "initialize"):
                try:
                    initializer = mtkahypar.initialize(1, False)
                except TypeError:
                    initializer = mtkahypar.initialize(1)
            num_vertices = benchmark.num_hard_macros
            hyperedge_nodes = []
            edge_weights = []
            for edge in hyperedges:
                hyperedge_nodes.append([int(node) for node in edge["nodes"]])
                edge_weights.append(max(1, int(round(edge["weight"]))))

            vertex_weights = np.maximum(
                1,
                np.rint(
                    (
                        benchmark.macro_sizes[:num_vertices, 0]
                        * benchmark.macro_sizes[:num_vertices, 1]
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    * 1000.0
                ).astype(np.int64),
            )
            preset = getattr(getattr(mtkahypar, "PresetType", object), "DETERMINISTIC", None)
            if initializer is not None and hasattr(initializer, "context_from_preset"):
                context = initializer.context_from_preset(
                    preset if preset is not None else getattr(mtkahypar.PresetType, "DEFAULT")
                )
            else:
                return None
            if hasattr(context, "logging"):
                context.logging = False
            objective_enum = getattr(getattr(mtkahypar, "Objective", object), objective.upper(), None)
            if hasattr(context, "set_partitioning_parameters"):
                context.set_partitioning_parameters(
                    num_parts,
                    imbalance,
                    objective_enum if objective_enum is not None else getattr(mtkahypar.Objective, "KM1"),
                )
            elif hasattr(context, "setPartitioningParameters"):
                context.setPartitioningParameters(
                    num_parts,
                    imbalance,
                    objective_enum if objective_enum is not None else getattr(mtkahypar.Objective, "KM1"),
                )
            if hasattr(mtkahypar, "set_seed"):
                mtkahypar.set_seed(self.seed + self._stable_name_seed(benchmark.name) + num_parts)
            if initializer is not None and hasattr(initializer, "create_hypergraph"):
                hypergraph = initializer.create_hypergraph(
                    context,
                    num_vertices,
                    len(hyperedges),
                    hyperedge_nodes,
                    vertex_weights.tolist(),
                    edge_weights,
                )
            else:
                return None
            partitioned = hypergraph.partition(context)
            if hasattr(partitioned, "block_id"):
                return np.asarray([partitioned.block_id(i) for i in range(num_vertices)], dtype=np.int64)
            if hasattr(partitioned, "blockID"):
                return np.asarray([partitioned.blockID(i) for i in range(num_vertices)], dtype=np.int64)
        except Exception:
            return None
        return None

    def _run_kahypar_cli_partition(
        self,
        benchmark: Benchmark,
        hyperedges: list[dict],
        num_parts: int,
        imbalance: float,
        objective: str,
        binary: str,
        ini_path: str,
    ) -> np.ndarray | None:
        try:
            with tempfile.TemporaryDirectory(prefix="partcl-kahypar-") as tmpdir:
                hgr_path = Path(tmpdir) / "graph.hgr"
                fix_path = Path(tmpdir) / "graph.fix"
                out_path = Path(tmpdir) / "graph.part"
                self._write_hgr_file(hgr_path, benchmark, hyperedges)
                self._write_fix_file(fix_path, benchmark)
                cmd = [
                    binary,
                    "-h",
                    str(hgr_path),
                    "-k",
                    str(num_parts),
                    "-e",
                    str(imbalance),
                    "-o",
                    objective,
                    "-m",
                    "direct",
                    "-p",
                    ini_path,
                    "--writePartitionFile=true",
                    "--partitionFile",
                    str(out_path),
                    "--fixed",
                    str(fix_path),
                ]
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0 or not out_path.exists():
                    return None
                parts = np.loadtxt(out_path, dtype=np.int64)
                if parts.ndim == 0:
                    parts = np.asarray([int(parts)])
                if len(parts) != benchmark.num_hard_macros:
                    return None
                return parts
        except Exception:
            return None

    def _write_hgr_file(self, path: Path, benchmark: Benchmark, hyperedges: list[dict]) -> None:
        vertex_weights = np.maximum(
            1,
            np.rint(
                (
                    benchmark.macro_sizes[: benchmark.num_hard_macros, 0]
                    * benchmark.macro_sizes[: benchmark.num_hard_macros, 1]
                )
                .cpu()
                .numpy()
                .astype(np.float64)
                * 1000.0
            ).astype(np.int64),
        )
        lines = [f"{len(hyperedges)} {benchmark.num_hard_macros} 11"]
        for edge in hyperedges:
            nodes = " ".join(str(node + 1) for node in edge["nodes"])
            lines.append(f"{max(1, int(round(edge['weight'])))} {nodes}")
        lines.extend(str(int(weight)) for weight in vertex_weights.tolist())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_fix_file(self, path: Path, benchmark: Benchmark) -> None:
        fixed = benchmark.macro_fixed[: benchmark.num_hard_macros].cpu().numpy()
        # -1 marks a free vertex in hMetis-style fix files.
        lines = ["-1" for _ in range(benchmark.num_hard_macros)]
        for idx, is_fixed in enumerate(fixed):
            if is_fixed:
                lines[idx] = "0"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_dreamplace_or_fallback(
        self,
        seed: np.ndarray,
        base_pos: np.ndarray,
        benchmark: Benchmark,
        graph: dict,
        plc,
        cfg: dict,
    ) -> tuple[np.ndarray, str]:
        hard_pos = self._run_dreamplace_backend(
            seed=seed,
            benchmark=benchmark,
            plc=plc,
            cfg=cfg,
        )
        if hard_pos is not None:
            self._run_state["dreamplace_backend"] = "adapter"
            return self._legalize_hard_numpy(hard_pos, benchmark), "dreamplace"
        if self._run_state.get("dreamplace_backend") == "untried":
            self._run_state["dreamplace_backend"] = "fallback"
        return (
            self._analytical_refine(
                seed=seed,
                base_pos=base_pos,
                benchmark=benchmark,
                graph=graph,
                cfg=cfg,
            ),
            "fallback",
        )

    def _run_dreamplace_backend(
        self,
        seed: np.ndarray,
        benchmark: Benchmark,
        plc,
        cfg: dict,
    ) -> np.ndarray | None:
        module = self._load_dreamplace_adapter_module()
        if module is None:
            self._run_state["dreamplace_backend"] = "missing_adapter"
            return None

        for fn_name in ("run_dreamplace", "place"):
            fn = getattr(module, fn_name, None)
            if callable(fn):
                result = self._invoke_dreamplace_callable(
                    fn=fn,
                    benchmark=benchmark,
                    plc=plc,
                    seed=seed,
                    cfg=cfg,
                )
                if result is not None:
                    return result
        self._run_state["dreamplace_backend"] = "adapter_failed"
        return None

    def _run_dreamplace_full_backend(
        self,
        seed: np.ndarray,
        benchmark: Benchmark,
        plc,
        cfg: dict,
    ) -> np.ndarray | None:
        full_cfg = cfg.copy()
        full_cfg["return_all_macros"] = 1
        module = self._load_dreamplace_adapter_module()
        if module is None:
            self._run_state["dreamplace_backend"] = "missing_adapter"
            return None

        for fn_name in ("run_dreamplace", "place"):
            fn = getattr(module, fn_name, None)
            if not callable(fn):
                continue
            result = self._invoke_dreamplace_callable(
                fn=fn,
                benchmark=benchmark,
                plc=plc,
                seed=seed,
                cfg=full_cfg,
            )
            if result is None:
                continue
            result = np.asarray(result, dtype=np.float64)
            if result.shape == (benchmark.num_macros, 2):
                self._run_state["dreamplace_backend"] = "adapter_full"
                return result
        self._run_state["dreamplace_backend"] = "adapter_full_failed"
        return None

    def _load_dreamplace_adapter_module(self):
        module_name = self._get_backend_status()["dreamplace_module"]
        if module_name:
            try:
                return importlib.import_module(module_name)
            except Exception:
                return None

        adapter_path = Path(__file__).with_name("dreamplace_adapter.py")
        if not adapter_path.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location("partcl_dreamplace_adapter", adapter_path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            return None

    def _invoke_dreamplace_callable(self, fn, benchmark: Benchmark, plc, seed: np.ndarray, cfg: dict) -> np.ndarray | None:
        kwargs = {
            "benchmark": benchmark,
            "plc": plc,
            "seed": seed.copy(),
            "config": cfg.copy(),
            "hard_macro_indices": benchmark.hard_macro_indices,
            "soft_macro_indices": benchmark.soft_macro_indices,
        }
        try:
            signature = inspect.signature(fn)
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters or any(
                    param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
                )
            }
            result = fn(**accepted)
        except TypeError:
            try:
                result = fn(benchmark, seed.copy(), cfg.copy())
            except Exception:
                return None
        except Exception:
            return None

        if result is None:
            return None
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        result = np.asarray(result, dtype=np.float64)
        expected_shape = (
            (benchmark.num_macros, 2)
            if bool(int(cfg.get("return_all_macros", 0)))
            else (benchmark.num_hard_macros, 2)
        )
        if result.shape != expected_shape:
            return None
        return result

    def _region_grid_points(self, rect, count: int, widths, heights) -> np.ndarray:
        left, bottom, right, top = rect
        region_w = max(right - left, 1e-6)
        region_h = max(top - bottom, 1e-6)
        aspect = region_w / region_h
        cols = max(1, int(np.ceil(np.sqrt(count * aspect))))
        rows = max(1, int(np.ceil(count / cols)))

        margin_x = min(region_w * 0.2, max(float(np.max(widths)) * 0.6, region_w * 0.05))
        margin_y = min(region_h * 0.2, max(float(np.max(heights)) * 0.6, region_h * 0.05))

        if right - left <= 2 * margin_x:
            xs = np.full(cols, (left + right) * 0.5)
        else:
            xs = np.linspace(left + margin_x, right - margin_x, cols)
        if top - bottom <= 2 * margin_y:
            ys = np.full(rows, (bottom + top) * 0.5)
        else:
            ys = np.linspace(bottom + margin_y, top - margin_y, rows)

        points = []
        for row_idx, y in enumerate(ys):
            x_iter = xs if row_idx % 2 == 0 else xs[::-1]
            for x in x_iter:
                points.append([x, y])
        return np.asarray(points[:count], dtype=np.float64)

    def _analytical_refine(
        self,
        seed: np.ndarray,
        base_pos: np.ndarray,
        benchmark: Benchmark,
        graph: dict,
        cfg: dict,
    ) -> np.ndarray:
        num_hard = benchmark.num_hard_macros
        pos = seed.copy()
        prev_pos = None
        prev_grad = None

        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        half_w = widths * 0.5
        half_h = heights * 0.5
        degree = graph["hard_degree"]
        hard_i = graph["hard_i"]
        hard_j = graph["hard_j"]
        hard_w = graph["hard_w"]
        repulsion_period = max(1, int(cfg.get("repulsion_period", 1)))
        repulsion_grad = np.zeros((num_hard, 2), dtype=np.float64)

        step_size = cfg["lr"]
        for step in range(cfg["steps"]):
            grad = np.zeros((num_hard, 2), dtype=np.float64)

            if len(hard_w) > 0:
                nbr_sum = np.zeros_like(pos)
                nbr_w = np.zeros(num_hard, dtype=np.float64)
                np.add.at(nbr_sum, hard_i, pos[hard_j] * hard_w[:, None])
                np.add.at(nbr_sum, hard_j, pos[hard_i] * hard_w[:, None])
                np.add.at(nbr_w, hard_i, hard_w)
                np.add.at(nbr_w, hard_j, hard_w)
                mask = nbr_w > 0
                bary = pos.copy()
                bary[mask] = nbr_sum[mask] / nbr_w[mask, None]
                grad += cfg["graph_w"] * (pos - bary)

            grad += cfg["anchor_w"] * (pos - seed)
            grad += cfg["base_w"] * (pos - base_pos)
            if step % repulsion_period == 0:
                repulsion_grad = self._repulsion_gradient(
                    pos=pos,
                    widths=widths,
                    heights=heights,
                    movable=movable,
                    weight=cfg["repulsion_w"],
                )
            grad += repulsion_grad

            grad[~movable] = 0.0
            if prev_pos is not None and prev_grad is not None:
                s = (pos[movable] - prev_pos[movable]).reshape(-1)
                y = (grad[movable] - prev_grad[movable]).reshape(-1)
                denom = float(np.dot(s, y))
                if denom > 1.0e-9:
                    step_size = float(np.clip(np.dot(s, s) / denom, 0.06, 0.85))

            prev_pos = pos.copy()
            prev_grad = grad.copy()

            scale = 1.0 + degree[:, None]
            pos[movable] -= step_size * grad[movable] / scale[movable]
            pos[:, 0] = np.clip(pos[:, 0], half_w, benchmark.canvas_width - half_w)
            pos[:, 1] = np.clip(pos[:, 1], half_h, benchmark.canvas_height - half_h)

            if (step + 1) % cfg["legalize_every"] == 0:
                pos = self._legalize_hard_numpy(pos, benchmark)

        return self._legalize_hard_numpy(pos, benchmark)

    def _repulsion_gradient(
        self,
        pos: np.ndarray,
        widths: np.ndarray,
        heights: np.ndarray,
        movable: np.ndarray,
        weight: float,
    ) -> np.ndarray:
        num_hard = pos.shape[0]
        dx = pos[:, 0:1] - pos[None, :, 0]
        dy = pos[:, 1:2] - pos[None, :, 1]
        adx = np.abs(dx)
        ady = np.abs(dy)

        sep_x = (widths[:, None] + widths[None, :]) * 0.5 + 0.02
        sep_y = (heights[:, None] + heights[None, :]) * 0.5 + 0.02
        overlap_x = sep_x - adx
        overlap_y = sep_y - ady

        mask = (overlap_x > 0.0) & (overlap_y > 0.0)
        np.fill_diagonal(mask, False)

        sign_x = np.sign(dx)
        sign_y = np.sign(dy)
        zero_x = sign_x == 0.0
        zero_y = sign_y == 0.0
        idx_grid = np.arange(num_hard)
        sign_x[zero_x] = np.where(idx_grid[:, None] >= idx_grid[None, :], 1.0, -1.0)[zero_x]
        sign_y[zero_y] = np.where(idx_grid[:, None] >= idx_grid[None, :], 1.0, -1.0)[zero_y]

        prefer_x = overlap_x <= overlap_y
        force_x = weight * overlap_x * sign_x * mask * prefer_x
        force_y = weight * overlap_y * sign_y * mask * (~prefer_x)

        grad = np.zeros((num_hard, 2), dtype=np.float64)
        grad[:, 0] += np.sum(force_x, axis=1)
        grad[:, 1] += np.sum(force_y, axis=1)
        grad[~movable] = 0.0
        return grad

    def _follow_soft_macros(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        iterations: int,
    ) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        if num_soft == 0:
            return placement
        result = placement.clone()
        for _ in range(max(1, iterations)):
            result = self._quadratic_soft_macro_follow(
                placement=result,
                benchmark=benchmark,
                graph=graph,
                anchor_weight=0.20,
                relax=0.72,
                solver_steps=8 + 3 * max(1, iterations),
            )
        return result

    def _quadratic_soft_macro_follow(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        anchor_weight: float,
        relax: float,
        solver_steps: int,
        anchor_pos: np.ndarray | None = None,
        movable_soft: np.ndarray | None = None,
    ) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        if num_soft == 0:
            return placement

        system = self._get_soft_follow_system(graph, benchmark)
        result = placement.clone()
        pos = result.cpu().numpy().astype(np.float64)
        soft = pos[num_hard:].copy()
        hard = pos[:num_hard]
        if anchor_pos is None:
            base_soft = benchmark.macro_positions[num_hard:].cpu().numpy().astype(np.float64)
        else:
            base_soft = np.asarray(anchor_pos, dtype=np.float64).copy()
        port_pos = benchmark.port_positions.cpu().numpy().astype(np.float64)
        widths = benchmark.macro_sizes[num_hard:, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[num_hard:, 1].cpu().numpy().astype(np.float64)
        fixed_soft = benchmark.macro_fixed[num_hard:].cpu().numpy()
        if movable_soft is not None:
            fixed_soft = fixed_soft | (~np.asarray(movable_soft, dtype=bool))

        rhs = anchor_weight * base_soft
        diag = anchor_weight + system["degree_sum"]
        for soft_idx, terms in enumerate(system["external_terms"]):
            accum = rhs[soft_idx]
            for term_kind, term_idx, weight in terms:
                if term_kind == 0:
                    accum += weight * hard[term_idx]
                else:
                    accum += weight * port_pos[term_idx]
            rhs[soft_idx] = accum

        for _ in range(max(1, solver_steps)):
            prev = soft.copy()
            for soft_idx in range(num_soft):
                if fixed_soft[soft_idx]:
                    soft[soft_idx] = base_soft[soft_idx]
                    continue

                accum = rhs[soft_idx].copy()
                for other_soft, weight in system["soft_terms"][soft_idx]:
                    accum += weight * prev[other_soft]
                target = accum / max(diag[soft_idx], 1.0e-9)
                soft[soft_idx] = (1.0 - relax) * prev[soft_idx] + relax * target

            soft[:, 0] = np.clip(soft[:, 0], widths * 0.5, benchmark.canvas_width - widths * 0.5)
            soft[:, 1] = np.clip(soft[:, 1], heights * 0.5, benchmark.canvas_height - heights * 0.5)
            soft[fixed_soft] = base_soft[fixed_soft]

        result[num_hard:] = torch.tensor(soft, dtype=result.dtype)
        return result

    def _get_soft_follow_system(self, graph: dict, benchmark: Benchmark) -> dict:
        cached = graph.get("soft_follow_system")
        if cached is not None:
            return cached

        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        num_macros = benchmark.num_macros
        soft_terms = [[] for _ in range(num_soft)]
        external_terms = [[] for _ in range(num_soft)]
        degree_sum = np.zeros(num_soft, dtype=np.float64)

        for soft_idx, neighbors in enumerate(graph["soft_neighbors"]):
            soft_weight_map = {}
            external_weight_map = {}
            for other, weight in neighbors:
                w = float(weight)
                if w <= 0.0:
                    continue
                if num_hard <= other < num_macros:
                    other_soft = int(other - num_hard)
                    soft_weight_map[other_soft] = soft_weight_map.get(other_soft, 0.0) + w
                elif other < num_hard:
                    external_weight_map[(0, int(other))] = external_weight_map.get((0, int(other)), 0.0) + w
                else:
                    port_idx = int(other - num_macros)
                    external_weight_map[(1, port_idx)] = external_weight_map.get((1, port_idx), 0.0) + w

            soft_terms[soft_idx] = [(other_soft, w) for other_soft, w in soft_weight_map.items()]
            external_terms[soft_idx] = [(kind, idx, w) for (kind, idx), w in external_weight_map.items()]
            degree_sum[soft_idx] = sum(soft_weight_map.values()) + sum(external_weight_map.values())

        graph["soft_follow_system"] = {
            "soft_terms": soft_terms,
            "external_terms": external_terms,
            "degree_sum": degree_sum,
        }
        return graph["soft_follow_system"]

    def _strong_soft_macro_follow(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
    ) -> torch.Tensor:
        result = self._follow_soft_macros(
            placement=placement,
            benchmark=benchmark,
            graph=graph,
            iterations=4 if benchmark.num_macros <= 320 else 3,
        )
        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        if num_soft == 0:
            return result

        hard = result[:num_hard]
        soft = result[num_hard:].clone()
        widths = benchmark.macro_sizes[:, 0]
        heights = benchmark.macro_sizes[:, 1]
        base = benchmark.macro_positions
        for _ in range(2):
            snapshot = soft.clone()
            for soft_idx in range(num_soft):
                bench_idx = num_hard + soft_idx
                if benchmark.macro_fixed[bench_idx]:
                    soft[soft_idx] = base[bench_idx]
                    continue

                repel = torch.zeros(2, dtype=result.dtype)
                for hard_idx in range(num_hard):
                    delta = snapshot[soft_idx] - hard[hard_idx]
                    sep_x = 0.5 * (widths[bench_idx] + widths[hard_idx]) + 0.05
                    sep_y = 0.5 * (heights[bench_idx] + heights[hard_idx]) + 0.05
                    if abs(float(delta[0])) < float(sep_x) and abs(float(delta[1])) < float(sep_y):
                        if abs(float(delta[0])) >= abs(float(delta[1])):
                            repel[0] += 0.12 * torch.sign(delta[0] if delta[0] != 0 else torch.tensor(1.0))
                        else:
                            repel[1] += 0.12 * torch.sign(delta[1] if delta[1] != 0 else torch.tensor(1.0))

                target = 0.88 * snapshot[soft_idx] + 0.12 * base[bench_idx] + repel
                w = widths[bench_idx].item()
                h = heights[bench_idx].item()
                target[0] = torch.clamp(target[0], min=w * 0.5, max=benchmark.canvas_width - w * 0.5)
                target[1] = torch.clamp(target[1], min=h * 0.5, max=benchmark.canvas_height - h * 0.5)
                soft[soft_idx] = target
        result[num_hard:] = soft
        return result

    def _post_refine_candidate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
        rounds: int,
    ) -> torch.Tensor:
        if plc is None or rounds <= 0:
            return placement

        best = self._strong_soft_macro_follow(placement, benchmark, graph)
        best_costs = self._exact_costs(best, benchmark, plc)
        for step in range(rounds):
            candidate = self._hotspot_refine_once(
                placement=best,
                benchmark=benchmark,
                graph=graph,
                plc=plc,
                move_cap=0.018 / max(1, step + 1),
            )
            candidate = self._strong_soft_macro_follow(candidate, benchmark, graph)
            costs = self._exact_costs(candidate, benchmark, plc)
            print(
                f"[partcl:refine] {benchmark.name} iter={step + 1} "
                f"wl={costs['wirelength_cost']:.4f} den={costs['density_cost']:.4f} "
                f"cong={costs['congestion_cost']:.4f} proxy={costs['proxy_cost']:.4f}"
            )
            if costs["proxy_cost"] + 1.0e-4 < best_costs["proxy_cost"]:
                best = candidate
                best_costs = costs
            else:
                break
        return best

    def _hotspot_refine_once(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
        move_cap: float,
    ) -> torch.Tensor:
        from macro_place.objective import _set_placement

        result = placement.clone()
        _set_placement(plc, result, benchmark)

        nrow = benchmark.grid_rows
        ncol = benchmark.grid_cols
        density = np.asarray(plc.grid_cells, dtype=np.float64).reshape(nrow, ncol)
        h_cong = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        congestion = np.maximum(h_cong, v_cong)

        dens_thr = float(np.percentile(density, 90.0))
        cong_thr = float(np.percentile(congestion, 95.0))
        hot_density = density >= dens_thr
        hot_congestion = congestion >= cong_thr
        hot_mask = hot_density | hot_congestion
        if not np.any(hot_mask):
            return result

        hotspot_scores = np.zeros((nrow, ncol), dtype=np.float64)
        hotspot_scores += np.maximum(0.0, density - dens_thr)
        hotspot_scores += 1.9 * np.maximum(0.0, congestion - cong_thr)
        hotspot_scores *= hot_mask.astype(np.float64)

        num_hard = benchmark.num_hard_macros
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        base = benchmark.macro_positions[:num_hard].cpu().numpy().astype(np.float64)
        pos = result[:num_hard].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        grid_w = benchmark.canvas_width / max(ncol, 1)
        grid_h = benchmark.canvas_height / max(nrow, 1)
        max_move = move_cap * max(benchmark.canvas_width, benchmark.canvas_height)

        hard_i = graph["hard_i"]
        hard_j = graph["hard_j"]
        hard_w = graph["hard_w"]
        bary = pos.copy()
        if len(hard_w) > 0:
            nbr_sum = np.zeros_like(pos)
            nbr_w = np.zeros(num_hard, dtype=np.float64)
            np.add.at(nbr_sum, hard_i, pos[hard_j] * hard_w[:, None])
            np.add.at(nbr_sum, hard_j, pos[hard_i] * hard_w[:, None])
            np.add.at(nbr_w, hard_i, hard_w)
            np.add.at(nbr_w, hard_j, hard_w)
            mask = nbr_w > 0
            bary[mask] = nbr_sum[mask] / nbr_w[mask, None]

        pressures = np.zeros(num_hard, dtype=np.float64)
        delta = np.zeros((num_hard, 2), dtype=np.float64)
        macro_boxes = []
        for idx in range(num_hard):
            xmin = max(0, int(np.floor((pos[idx, 0] - widths[idx] * 0.5) / grid_w)))
            xmax = min(ncol - 1, int(np.floor((pos[idx, 0] + widths[idx] * 0.5) / grid_w)))
            ymin = max(0, int(np.floor((pos[idx, 1] - heights[idx] * 0.5) / grid_h)))
            ymax = min(nrow - 1, int(np.floor((pos[idx, 1] + heights[idx] * 0.5) / grid_h)))
            macro_boxes.append((xmin, xmax, ymin, ymax))

        flat_hot = np.argsort(hotspot_scores.reshape(-1))[::-1]
        hotspot_limit = 18 if num_hard <= 320 else 14 if num_hard <= 430 else 10
        active_bins = []
        active_macro_indices = set()
        for flat_idx in flat_hot:
            score = hotspot_scores.reshape(-1)[flat_idx]
            if score <= 0.0:
                break
            row = int(flat_idx // ncol)
            col = int(flat_idx % ncol)
            active_bins.append((row, col, score))
            for idx, (xmin, xmax, ymin, ymax) in enumerate(macro_boxes):
                if not movable[idx]:
                    continue
                if xmin <= col <= xmax and ymin <= row <= ymax:
                    active_macro_indices.add(idx)
            if len(active_bins) >= hotspot_limit and len(active_macro_indices) >= hotspot_limit:
                break

        if not active_macro_indices:
            return result

        for idx in sorted(active_macro_indices):
            xmin, xmax, ymin, ymax = macro_boxes[idx]

            local_vec = np.zeros(2, dtype=np.float64)
            pressure = 0.0
            for row, col, score in active_bins:
                if not (xmin <= col <= xmax and ymin <= row <= ymax):
                    continue
                cx = (col + 0.5) * grid_w
                cy = (row + 0.5) * grid_h
                away = pos[idx] - np.array([cx, cy], dtype=np.float64)
                norm = np.linalg.norm(away)
                if norm < 1.0e-9:
                    away = pos[idx] - bary[idx]
                    norm = np.linalg.norm(away)
                if norm < 1.0e-9:
                    away = pos[idx] - base[idx]
                    norm = np.linalg.norm(away)
                if norm < 1.0e-9:
                    continue
                local_vec += score * away / norm
                pressure += score

            if pressure <= 0.0:
                continue
            wl_pull = 0.24 * (bary[idx] - pos[idx]) + 0.06 * (base[idx] - pos[idx])
            delta[idx] = local_vec + wl_pull
            pressures[idx] = pressure

        if not np.any(pressures > 0.0):
            return result

        limit = 8 if num_hard <= 320 else 6 if num_hard <= 430 else 5
        target_indices = np.argsort(-pressures)[:limit]
        new_pos = pos.copy()
        for idx in target_indices:
            vec = delta[idx]
            norm = np.linalg.norm(vec)
            if norm < 1.0e-9:
                continue
            scale = min(
                max_move,
                0.35 * max_move
                + pressures[idx] / max(np.max(pressures), 1.0e-9) * 0.45 * max_move,
            )
            new_pos[idx] += (vec / norm) * scale

        new_pos = self._legalize_hard_numpy(new_pos, benchmark)
        result[:num_hard] = torch.tensor(new_pos, dtype=result.dtype)
        return result

    def _exact_costs(self, placement: torch.Tensor, benchmark: Benchmark, plc) -> dict:
        from macro_place.objective import compute_proxy_cost

        key = (benchmark.name, self._placement_cache_key(placement))
        cached = self._exact_cache.get(key)
        if cached is not None:
            self._run_state["exact_cache_hits"] = self._run_state.get("exact_cache_hits", 0) + 1
            self._restore_exact_cache_state(plc, cached)
            return dict(cached["costs"])

        start = time.perf_counter()
        costs = compute_proxy_cost(placement, benchmark, plc)
        self._run_state["exact_cache_misses"] = self._run_state.get("exact_cache_misses", 0) + 1
        self._run_state["exact_time_s"] = self._run_state.get("exact_time_s", 0.0) + (time.perf_counter() - start)
        self._store_exact_cache_entry(key, plc, costs)
        return dict(costs)

    def _placement_cache_key(self, placement: torch.Tensor) -> bytes:
        quant = float(os.environ.get("PARTCL_EXACT_CACHE_QUANT", "1e-4"))
        arr = placement.detach().cpu().contiguous().numpy().astype(np.float64, copy=False)
        scaled = np.rint(arr / max(quant, 1.0e-9)).astype(np.int64, copy=False)
        return scaled.tobytes()

    def _store_exact_cache_entry(self, key, plc, costs: dict) -> None:
        entry = {
            "costs": dict(costs),
            "grid_cells": np.asarray(plc.grid_cells, dtype=np.float32).copy(),
            "h_cong": np.asarray(plc.H_routing_cong, dtype=np.float32).copy(),
            "v_cong": np.asarray(plc.V_routing_cong, dtype=np.float32).copy(),
        }
        self._exact_cache[key] = entry
        self._exact_cache_order.append(key)
        while len(self._exact_cache_order) > self._exact_cache_limit:
            old_key = self._exact_cache_order.popleft()
            self._exact_cache.pop(old_key, None)

    def _restore_exact_cache_state(self, plc, entry: dict) -> None:
        plc.grid_cells = entry["grid_cells"].copy()
        plc.H_routing_cong = entry["h_cong"].copy()
        plc.V_routing_cong = entry["v_cong"].copy()

    def _log_exact_cache_summary(self, benchmark: Benchmark) -> None:
        hits = int(self._run_state.get("exact_cache_hits", 0))
        misses = int(self._run_state.get("exact_cache_misses", 0))
        total = hits + misses
        hit_rate = (hits / total) if total > 0 else 0.0
        exact_time_s = float(self._run_state.get("exact_time_s", 0.0))
        print(
            f"[partcl:exact] {benchmark.name} "
            f"hits={hits} misses={misses} hit_rate={hit_rate:.1%} "
            f"compute_time={exact_time_s:.2f}s"
        )

    def _exact_score(self, placement: torch.Tensor, benchmark: Benchmark, plc) -> float:
        costs = self._exact_costs(placement, benchmark, plc)
        return float(costs["proxy_cost"] + 1000.0 * costs["overlap_count"])

    def _surrogate_score(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        base_pos: np.ndarray | None = None,
        base_weight: float = 0.02,
    ) -> float:
        pos = placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64)
        if base_pos is None:
            base = benchmark.macro_positions[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64)
        else:
            base = np.asarray(base_pos, dtype=np.float64)
        hard_i = graph["hard_i"]
        hard_j = graph["hard_j"]
        hard_w = graph["hard_w"]

        score = 0.0
        if len(hard_w) > 0:
            diff = np.abs(pos[hard_i] - pos[hard_j])
            score += float(np.sum((diff[:, 0] + diff[:, 1]) * hard_w))
        score += float(base_weight) * float(np.sum(np.square(pos - base)))

        widths = benchmark.macro_sizes[: benchmark.num_hard_macros, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[: benchmark.num_hard_macros, 1].cpu().numpy().astype(np.float64)
        dx = np.abs(pos[:, 0:1] - pos[None, :, 0])
        dy = np.abs(pos[:, 1:2] - pos[None, :, 1])
        ox = np.maximum(0.0, (widths[:, None] + widths[None, :]) * 0.5 - dx)
        oy = np.maximum(0.0, (heights[:, None] + heights[None, :]) * 0.5 - dy)
        overlap = ox * oy
        score += 2000.0 * float(np.triu(overlap, 1).sum())
        return score

    def _coarse_proxy_score(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        nrow = min(18, max(8, benchmark.grid_rows // 2))
        ncol = min(18, max(8, benchmark.grid_cols // 2))
        cell_w = benchmark.canvas_width / max(ncol, 1)
        cell_h = benchmark.canvas_height / max(nrow, 1)
        cell_area = max(cell_w * cell_h, 1.0e-9)

        pos = placement.cpu().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
        num_macros = benchmark.num_macros
        density = np.zeros((nrow, ncol), dtype=np.float64)
        for idx in range(num_macros):
            x, y = pos[idx]
            c = min(ncol - 1, max(0, int(x / max(cell_w, 1.0e-9))))
            r = min(nrow - 1, max(0, int(y / max(cell_h, 1.0e-9))))
            density[r, c] += (sizes[idx, 0] * sizes[idx, 1]) / cell_area

        congestion = np.zeros((nrow, ncol), dtype=np.float64)
        port_start = benchmark.num_macros
        for net_idx, nodes in enumerate(benchmark.net_nodes):
            if len(nodes) < 2:
                continue
            pts = []
            for node in nodes.tolist():
                if node < benchmark.num_macros:
                    pts.append(pos[node])
                else:
                    port_idx = node - port_start
                    if 0 <= port_idx < benchmark.port_positions.shape[0]:
                        pts.append(benchmark.port_positions[port_idx].cpu().numpy().astype(np.float64))
            if len(pts) < 2:
                continue
            pts_arr = np.asarray(pts, dtype=np.float64)
            xmin, ymin = np.min(pts_arr, axis=0)
            xmax, ymax = np.max(pts_arr, axis=0)
            c0 = min(ncol - 1, max(0, int(xmin / max(cell_w, 1.0e-9))))
            c1 = min(ncol - 1, max(0, int(xmax / max(cell_w, 1.0e-9))))
            r0 = min(nrow - 1, max(0, int(ymin / max(cell_h, 1.0e-9))))
            r1 = min(nrow - 1, max(0, int(ymax / max(cell_h, 1.0e-9))))
            span = max((r1 - r0 + 1) * (c1 - c0 + 1), 1)
            weight = (
                float(benchmark.net_weights[net_idx].item())
                if net_idx < len(benchmark.net_weights)
                else 1.0
            )
            congestion[r0 : r1 + 1, c0 : c1 + 1] += weight / span

        dens_tail = float(np.mean(np.sort(density.reshape(-1))[-max(1, density.size // 10) :]))
        cong_tail = float(np.mean(np.sort(congestion.reshape(-1))[-max(1, congestion.size // 20) :]))
        center = np.array([0.5 * benchmark.canvas_width, 0.5 * benchmark.canvas_height], dtype=np.float64)
        hard = pos[: benchmark.num_hard_macros]
        hard_spread = float(np.mean(np.linalg.norm(hard - center[None, :], axis=1)))
        return 55.0 * dens_tail + 32.0 * cong_tail - 0.35 * hard_spread

    def _legalize_hard_numpy(self, pos: np.ndarray, benchmark: Benchmark) -> np.ndarray:
        num_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        pos = pos.copy()
        pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) * 0.5
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) * 0.5
        placed = np.zeros(num_hard, dtype=bool)
        legal = pos.copy()

        order = sorted(
            range(num_hard),
            key=lambda idx: -(sizes[idx, 0] * sizes[idx, 1]),
        )

        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue

            if placed.any():
                dx = np.abs(legal[idx, 0] - legal[:, 0])
                dy = np.abs(legal[idx, 1] - legal[:, 1])
                conflicts = (dx < sep_x[idx] + 0.02) & (dy < sep_y[idx] + 0.02) & placed
                conflicts[idx] = False
                if not conflicts.any():
                    placed[idx] = True
                    continue

            step = max(sizes[idx, 0], sizes[idx, 1]) * 0.22 + 0.01
            best = legal[idx].copy()
            best_dist = float("inf")
            for ring in range(1, 72):
                found = False
                for dx_ring in range(-ring, ring + 1):
                    for dy_ring in range(-ring, ring + 1):
                        if abs(dx_ring) != ring and abs(dy_ring) != ring:
                            continue

                        cand_x = np.clip(pos[idx, 0] + dx_ring * step, half_w[idx], canvas_w - half_w[idx])
                        cand_y = np.clip(pos[idx, 1] + dy_ring * step, half_h[idx], canvas_h - half_h[idx])
                        if placed.any():
                            dx = np.abs(cand_x - legal[:, 0])
                            dy = np.abs(cand_y - legal[:, 1])
                            conflicts = (dx < sep_x[idx] + 0.02) & (dy < sep_y[idx] + 0.02) & placed
                            conflicts[idx] = False
                            if conflicts.any():
                                continue

                        dist = (cand_x - pos[idx, 0]) ** 2 + (cand_y - pos[idx, 1]) ** 2
                        if dist < best_dist:
                            best = np.array([cand_x, cand_y], dtype=np.float64)
                            best_dist = dist
                            found = True
                if found:
                    break

            legal[idx] = best
            placed[idx] = True

        if self._count_hard_overlaps_numpy(legal, benchmark) == 0:
            return legal
        return self._repair_hard_overlaps_numpy(legal, benchmark)

    def _count_hard_overlaps_numpy(self, pos: np.ndarray, benchmark: Benchmark) -> int:
        num_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        count = 0
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                overlap_x = (half_w[i] + half_w[j]) - abs(float(pos[j, 0] - pos[i, 0]))
                overlap_y = (half_h[i] + half_h[j]) - abs(float(pos[j, 1] - pos[i, 1]))
                if overlap_x > 1.0e-7 and overlap_y > 1.0e-7:
                    count += 1
        return count

    def _repair_hard_overlaps_numpy(self, pos: np.ndarray, benchmark: Benchmark) -> np.ndarray:
        num_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        legal = pos.copy()
        clearance = 0.04

        for _ in range(80):
            moved = False
            for i in range(num_hard):
                for j in range(i + 1, num_hard):
                    dx = legal[j, 0] - legal[i, 0]
                    dy = legal[j, 1] - legal[i, 1]
                    overlap_x = (half_w[i] + half_w[j] + clearance) - abs(dx)
                    overlap_y = (half_h[i] + half_h[j] + clearance) - abs(dy)
                    if overlap_x <= 0.0 or overlap_y <= 0.0:
                        continue
                    if not movable[i] and not movable[j]:
                        continue

                    sign_x = 1.0 if dx >= 0.0 else -1.0
                    sign_y = 1.0 if dy >= 0.0 else -1.0
                    use_x = overlap_x <= overlap_y
                    axis = 0 if use_x else 1
                    sign = sign_x if use_x else sign_y
                    amount = (overlap_x if use_x else overlap_y) + clearance

                    if movable[i] and movable[j]:
                        legal[i, axis] -= 0.5 * sign * amount
                        legal[j, axis] += 0.5 * sign * amount
                    elif movable[i]:
                        legal[i, axis] -= sign * amount
                    else:
                        legal[j, axis] += sign * amount
                    moved = True

            legal[:, 0] = np.clip(legal[:, 0], half_w, canvas_w - half_w)
            legal[:, 1] = np.clip(legal[:, 1], half_h, canvas_h - half_h)
            if not moved:
                break

        return legal

    def _load_plc(self, name: str):
        if name in self._plc_cache:
            return self._plc_cache[name]

        from macro_place.loader import load_benchmark

        plc = None
        ibm_root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        if ibm_root.exists():
            netlist = (ibm_root / "netlist.pb.txt").as_posix()
            plc_file = (ibm_root / "initial.plc").as_posix()
            _, plc = load_benchmark(netlist, plc_file, name=name)
        else:
            ng45 = {
                "ariane133": "ariane133",
                "ariane133_ng45": "ariane133",
                "ariane133_ng45_random": "ariane133",
                "ariane136": "ariane136",
                "ariane136_ng45": "ariane136",
                "mempool_tile": "mempool_tile",
                "mempool_tile_ng45": "mempool_tile",
                "nvdla": "nvdla",
                "nvdla_ng45": "nvdla",
            }
            design = ng45.get(name)
            if design is not None:
                base = (
                    Path("external/MacroPlacement/Flows/NanGate45")
                    / design
                    / "netlist"
                    / "output_CT_Grouping"
                )
                netlist = base / "netlist.pb.txt"
                plc_file = base / "initial.plc"
                if netlist.exists():
                    _, plc = load_benchmark(netlist.as_posix(), plc_file.as_posix(), name=design)

        self._plc_cache[name] = plc
        return plc

    def _build_outline_baseline(self, benchmark: Benchmark, graph: dict, rng, plc=None) -> torch.Tensor:
        candidate_placements: list[tuple[str, torch.Tensor]] = []
        random_start = self._build_random_start(benchmark)
        reference_outline = benchmark.macro_positions.clone()
        ref_hard = self._legalize_hard_numpy(
            reference_outline[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64),
            benchmark,
        )
        reference_outline[: benchmark.num_hard_macros] = torch.tensor(ref_hard, dtype=reference_outline.dtype)
        candidate_placements.append(("reference_outline", reference_outline.clone()))
        candidate_placements.append(("random_start", random_start.clone()))

        num_hard = benchmark.num_hard_macros
        small_hard_case = num_hard <= 320
        hard_random = random_start[:num_hard].cpu().numpy().astype(np.float64)
        base_cfg = {
            "steps": 34 if small_hard_case else 28,
            "graph_w": 0.92,
            "anchor_w": 0.14,
            "base_w": 0.0,
            "repulsion_w": 0.20,
            "legalize_every": 6,
            "lr": 0.32,
            "repulsion_period": 1 if num_hard <= 430 else 2,
            "density_weight": 0.0,
            "gamma": 4.0,
            "target_density": 0.58,
        }
        refined_random = self._analytical_refine(
            seed=hard_random.copy(),
            base_pos=hard_random.copy(),
            benchmark=benchmark,
            graph=graph,
            cfg=base_cfg,
        )
        trial = random_start.clone()
        trial[:num_hard] = torch.tensor(refined_random, dtype=trial.dtype)
        trial = self._follow_soft_macros(
            placement=trial,
            benchmark=benchmark,
            graph=graph,
            iterations=2 if small_hard_case else 1,
        )
        candidate_placements.append(("random_refined", trial))
        for factor, bias in ((1.28, 0.04), (1.40, 0.06), (1.52, 0.10)):
            spread_hard = self._build_spread_hard_seed(refined_random, benchmark, factor=factor, channel_bias=bias)
            spread_trial = random_start.clone()
            spread_trial[:num_hard] = torch.tensor(spread_hard, dtype=spread_trial.dtype)
            spread_trial = self._quadratic_soft_macro_follow(
                placement=spread_trial,
                benchmark=benchmark,
                graph=graph,
                anchor_weight=0.04 if small_hard_case else 0.08,
                relax=0.76 if small_hard_case else 0.74,
                solver_steps=18 if small_hard_case else 14,
            )
            candidate_placements.append((f"random_spread_f{factor:.2f}_b{bias:.2f}", spread_trial))

        partition_variants = [
            (4, "center", {"imbalance": 0.04, "objective": "km1"}),
            (4, "boundary", {"imbalance": 0.05, "objective": "cut"}),
        ]
        if small_hard_case:
            partition_variants.append((8, "center", {"imbalance": 0.05, "objective": "km1"}))

        for parts, variant, part_cfg in partition_variants:
            seed = self._build_partition_seed(
                benchmark=benchmark,
                graph=graph,
                num_parts=parts,
                variant=variant,
                partition_cfg=part_cfg,
                rng=rng,
            )
            cfg = dict(base_cfg)
            cfg.update(
                {
                    "steps": 34 if parts <= 4 else 38,
                    "graph_w": 1.08 if variant == "center" else 0.96,
                    "anchor_w": 0.22,
                    "repulsion_w": 0.16 if variant == "boundary" else 0.14,
                    "lr": 0.30,
                }
            )
            refined = self._analytical_refine(
                seed=seed.copy(),
                base_pos=seed.copy(),
                benchmark=benchmark,
                graph=graph,
                cfg=cfg,
            )
            candidate = random_start.clone()
            candidate[:num_hard] = torch.tensor(refined, dtype=candidate.dtype)
            candidate = self._follow_soft_macros(
                placement=candidate,
                benchmark=benchmark,
                graph=graph,
                iterations=2 if small_hard_case else 1,
            )
            base_name = f"partition_{parts}_{variant}_{part_cfg['objective']}"
            candidate_placements.append((base_name, candidate))
            spread_schedule = ((1.24, 0.03), (1.36, 0.05), (1.48, 0.08))
            if parts == 8 and variant == "center":
                spread_schedule = ((1.30, 0.04), (1.36, 0.05), (1.42, 0.06))
            for factor, bias in spread_schedule:
                spread_hard = self._build_spread_hard_seed(refined, benchmark, factor=factor, channel_bias=bias)
                spread_candidate = random_start.clone()
                spread_candidate[:num_hard] = torch.tensor(spread_hard, dtype=spread_candidate.dtype)
                spread_candidate = self._quadratic_soft_macro_follow(
                    placement=spread_candidate,
                    benchmark=benchmark,
                    graph=graph,
                    anchor_weight=0.04 if small_hard_case else 0.08,
                    relax=0.76 if small_hard_case else 0.74,
                    solver_steps=18 if small_hard_case else 14,
                )
                candidate_placements.append((f"{base_name}_spread_f{factor:.2f}_b{bias:.2f}", spread_candidate))

        best = None
        best_name = None
        best_score = None
        debug_baseline = os.environ.get("PARTCL_DEBUG_BASELINE", "0") == "1"
        for name, candidate in candidate_placements:
            if plc is not None:
                score = self._exact_score(candidate, benchmark, plc)
            else:
                score = self._surrogate_score(
                    candidate,
                    benchmark,
                    graph,
                    base_pos=candidate[:num_hard].cpu().numpy().astype(np.float64),
                    base_weight=0.0,
                )
                score += self._coarse_proxy_score(candidate, benchmark)
            if debug_baseline:
                print(f"[partcl:outline-base] {benchmark.name} candidate={name} score={score:.4f}")
            if best_score is None or score < best_score:
                best = candidate
                best_name = name
                best_score = score

        if best is not None:
            if debug_baseline and best_name is not None:
                print(f"[partcl:outline-base] {benchmark.name} winner={best_name} score={best_score:.4f}")
            return best

        placement = random_start
        hard_pos = self._legalize_hard_numpy(
            placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64),
            benchmark,
        )
        placement[: benchmark.num_hard_macros] = torch.tensor(
            hard_pos,
            dtype=placement.dtype,
        )
        return placement

    def _build_random_start(self, benchmark: Benchmark) -> torch.Tensor:
        outline_path = Path(__file__).with_name("submission_outline.py")
        if outline_path.exists():
            try:
                spec = importlib.util.spec_from_file_location("partcl_submission_outline_seed", outline_path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    placer_cls = getattr(module, "SubmissionOutlinePlacer", None)
                    if placer_cls is not None:
                        placer = placer_cls(seed=self.seed)
                        return placer._initialize_placement(benchmark)
            except Exception:
                pass

        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask()
        if not torch.any(movable):
            return placement

        env_seed = os.environ.get("PARTCL_RANDOM_INIT_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = self.seed
        else:
            seed = self.seed + self._stable_name_seed(benchmark.name)

        gen = torch.Generator(device=placement.device if placement.device.type != "cpu" else "cpu")
        gen.manual_seed(seed)
        widths = benchmark.macro_sizes[:, 0]
        heights = benchmark.macro_sizes[:, 1]
        x_min = widths * 0.5
        x_max = benchmark.canvas_width - widths * 0.5
        y_min = heights * 0.5
        y_max = benchmark.canvas_height - heights * 0.5
        rand_x = x_min + torch.rand(benchmark.num_macros, generator=gen) * torch.clamp(x_max - x_min, min=0.0)
        rand_y = y_min + torch.rand(benchmark.num_macros, generator=gen) * torch.clamp(y_max - y_min, min=0.0)
        placement[movable, 0] = rand_x[movable]
        placement[movable, 1] = rand_y[movable]
        placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return placement

    def _build_spread_hard_seed(
        self,
        hard_pos: np.ndarray,
        benchmark: Benchmark,
        factor: float,
        channel_bias: float = 0.0,
    ) -> np.ndarray:
        num_hard = benchmark.num_hard_macros
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        center = np.array([0.5 * benchmark.canvas_width, 0.5 * benchmark.canvas_height], dtype=np.float64)
        pos = hard_pos.copy()
        delta = pos - center[None, :]
        delta *= factor
        pos[movable] = center[None, :] + delta[movable]
        if channel_bias > 0.0:
            right_mask = pos[:, 0] >= center[0]
            top_mask = pos[:, 1] >= center[1]
            pos[movable & right_mask, 0] += channel_bias * benchmark.canvas_width
            pos[movable & ~right_mask, 0] -= channel_bias * benchmark.canvas_width
            pos[movable & top_mask, 1] += 0.55 * channel_bias * benchmark.canvas_height
            pos[movable & ~top_mask, 1] -= 0.55 * channel_bias * benchmark.canvas_height
        pos[:, 0] = np.clip(pos[:, 0], widths * 0.5, benchmark.canvas_width - widths * 0.5)
        pos[:, 1] = np.clip(pos[:, 1], heights * 0.5, benchmark.canvas_height - heights * 0.5)
        return self._legalize_hard_numpy(pos, benchmark)

    def _outline_micro_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None:
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        num_hard = benchmark.num_hard_macros
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        hard_i = graph["hard_i"]
        hard_j = graph["hard_j"]
        hard_w = graph["hard_w"]

        for _ in range(3):
            hot = self._get_hotspot_targets(
                placement=current,
                benchmark=benchmark,
                plc=plc,
                widths=widths,
                heights=heights,
                movable=movable,
                limit=6,
            )
            if not hot:
                break

            pos = current[:num_hard].cpu().numpy().astype(np.float64)
            bary = pos.copy()
            if len(hard_w) > 0:
                nbr_sum = np.zeros_like(pos)
                nbr_w = np.zeros(num_hard, dtype=np.float64)
                np.add.at(nbr_sum, hard_i, pos[hard_j] * hard_w[:, None])
                np.add.at(nbr_sum, hard_j, pos[hard_i] * hard_w[:, None])
                np.add.at(nbr_w, hard_i, hard_w)
                np.add.at(nbr_w, hard_j, hard_w)
                mask = nbr_w > 0
                bary[mask] = nbr_sum[mask] / nbr_w[mask, None]

            best_candidate = None
            best_costs = current_costs
            canvas_scale = max(benchmark.canvas_width, benchmark.canvas_height)
            for idx, hotspot_vec in hot:
                away = hotspot_vec + 0.18 * (pos[idx] - bary[idx])
                norm = np.linalg.norm(away)
                if norm < 1.0e-9:
                    continue
                away = away / norm
                for step_frac in (0.003, 0.006, 0.010):
                    trial = current.clone()
                    trial_pos = trial[:num_hard].cpu().numpy().astype(np.float64)
                    trial_pos[idx] += away * (step_frac * canvas_scale)
                    trial_pos = self._legalize_hard_numpy(trial_pos, benchmark)
                    trial[:num_hard] = torch.tensor(trial_pos, dtype=trial.dtype)
                    costs = self._exact_costs(trial, benchmark, plc)
                    if costs["proxy_cost"] + 1.0e-4 < best_costs["proxy_cost"]:
                        best_candidate = trial
                        best_costs = costs

            if best_candidate is None:
                break
            current = best_candidate
            current_costs = best_costs
            print(
                f"[partcl:outline] {benchmark.name} "
                f"wl={current_costs['wirelength_cost']:.4f} "
                f"den={current_costs['density_cost']:.4f} "
                f"cong={current_costs['congestion_cost']:.4f} "
                f"proxy={current_costs['proxy_cost']:.4f}"
            )

        return current

    def _rebalance_first_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None:
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        print(
            f"[partcl:rebalance-first] {benchmark.name} start "
            f"wl={current_costs['wirelength_cost']:.4f} "
            f"den={current_costs['density_cost']:.4f} "
            f"cong={current_costs['congestion_cost']:.4f} "
            f"proxy={current_costs['proxy_cost']:.4f}"
        )

        def dc_sum(costs: dict) -> float:
            return float(costs["density_cost"] + costs["congestion_cost"])

        def maybe_adopt(
            trial: torch.Tensor,
            trial_costs: dict,
            stage: str,
        ) -> None:
            nonlocal current, current_costs
            print(
                f"[partcl:rebalance-first] {benchmark.name} {stage} "
                f"wl={trial_costs['wirelength_cost']:.4f} "
                f"den={trial_costs['density_cost']:.4f} "
                f"cong={trial_costs['congestion_cost']:.4f} "
                f"proxy={trial_costs['proxy_cost']:.4f}"
            )
            proxy_gain = current_costs["proxy_cost"] - trial_costs["proxy_cost"]
            dc_gain = dc_sum(current_costs) - dc_sum(trial_costs)
            if proxy_gain >= 1.0e-4:
                current = trial
                current_costs = trial_costs
                return
            if dc_gain >= 0.020 and trial_costs["proxy_cost"] <= current_costs["proxy_cost"] + 0.025:
                current = trial
                current_costs = trial_costs

        rounds = max(1, int(os.environ.get("PARTCL_REBALANCE_FIRST_ROUNDS", "2")))
        hard_passes = max(1, int(os.environ.get("PARTCL_REBALANCE_FIRST_HARD_PASSES", "2")))
        enable_replite = os.environ.get("PARTCL_REBALANCE_FIRST_REPLITE", "1") != "0"
        route_heavy_mode = False
        whitespace_mode = False
        local_tail_mode = False
        best_corridor_proxy_gain = 0.0
        best_corridor_dc_gain = 0.0

        for round_idx in range(rounds):
            for hard_idx in range(hard_passes):
                trial = self._hard_outer_rebalance_polish(current.clone(), benchmark, graph, plc)
                trial_costs = self._exact_costs(trial, benchmark, plc)
                old_proxy = current_costs["proxy_cost"]
                maybe_adopt(trial, trial_costs, f"round={round_idx + 1} stage=hard{hard_idx + 1}")
                if current_costs["proxy_cost"] >= old_proxy - 1.0e-4:
                    break

            corridor_trials = [
                (
                    "channel",
                    self._outline_channel_soft_refine(current.clone(), benchmark, graph, plc),
                ),
                (
                    "whitespace",
                    self._outline_whitespace_soft_refine(current.clone(), benchmark, graph, plc),
                ),
                (
                    "channel_then_route",
                    self._outline_congestion_soft_refine(
                        self._outline_channel_soft_refine(current.clone(), benchmark, graph, plc),
                        benchmark,
                        graph,
                        plc,
                    ),
                ),
                (
                    "whitespace_then_channel",
                    self._outline_corridor_combo_refine(
                        current.clone(),
                        benchmark,
                        graph,
                        plc,
                        mode="whitespace_then_channel",
                    ),
                ),
                (
                    "channel_then_whitespace",
                    self._outline_corridor_combo_refine(
                        current.clone(),
                        benchmark,
                        graph,
                        plc,
                        mode="channel_then_whitespace",
                    ),
                ),
                (
                    "corridor_then_rebalance",
                    self._outline_corridor_combo_refine(
                        current.clone(),
                        benchmark,
                        graph,
                        plc,
                        mode="corridor_then_rebalance",
                    ),
                ),
            ]
            for label, trial in corridor_trials:
                trial_costs = self._exact_costs(trial, benchmark, plc)
                corridor_proxy_gain = current_costs["proxy_cost"] - trial_costs["proxy_cost"]
                corridor_dc_gain = dc_sum(current_costs) - dc_sum(trial_costs)
                if corridor_proxy_gain > best_corridor_proxy_gain:
                    best_corridor_proxy_gain = corridor_proxy_gain
                if corridor_dc_gain > best_corridor_dc_gain:
                    best_corridor_dc_gain = corridor_dc_gain
                maybe_adopt(trial, trial_costs, f"round={round_idx + 1} stage={label}")

            routed = self._outline_route_refine(current.clone(), benchmark, graph, plc)
            routed_costs = self._exact_costs(routed, benchmark, plc)
            pre_route_costs = current_costs
            maybe_adopt(routed, routed_costs, f"round={round_idx + 1} stage=route")
            route_proxy_gain = pre_route_costs["proxy_cost"] - current_costs["proxy_cost"]
            route_cong_gain = pre_route_costs["congestion_cost"] - current_costs["congestion_cost"]
            if route_proxy_gain >= 0.010 or route_cong_gain >= 0.030:
                route_heavy_mode = True
                route_followups = [
                    (
                        "route_then_channel",
                        self._outline_channel_soft_refine(current.clone(), benchmark, graph, plc),
                    ),
                    (
                        "route_then_whitespace",
                        self._outline_whitespace_soft_refine(current.clone(), benchmark, graph, plc),
                    ),
                    (
                        "route_then_channel_then_route",
                        self._outline_congestion_soft_refine(
                            self._outline_channel_soft_refine(current.clone(), benchmark, graph, plc),
                            benchmark,
                            graph,
                            plc,
                        ),
                    ),
                ]
                for label, trial in route_followups:
                    trial_costs = self._exact_costs(trial, benchmark, plc)
                    maybe_adopt(trial, trial_costs, f"round={round_idx + 1} stage={label}")

            if enable_replite:
                replite = self._outline_replite_refine(current.clone(), benchmark, graph, plc)
                replite_costs = self._exact_costs(replite, benchmark, plc)
                maybe_adopt(replite, replite_costs, f"round={round_idx + 1} stage=replite")

        if not route_heavy_mode:
            if best_corridor_proxy_gain >= 0.0008 or best_corridor_dc_gain >= 0.015:
                whitespace_mode = True
            else:
                local_tail_mode = True

        if whitespace_mode:
            ws_seed = self._outline_corridor_combo_refine(
                current.clone(),
                benchmark,
                graph,
                plc,
                mode="corridor_then_rebalance",
            )
            ws_seed_costs = self._exact_costs(ws_seed, benchmark, plc)
            maybe_adopt(ws_seed, ws_seed_costs, "stage=whitespace_mode_seed")

        if route_heavy_mode:
            route_seed = self._outline_congestion_soft_refine(current.clone(), benchmark, graph, plc)
            route_seed_costs = self._exact_costs(route_seed, benchmark, plc)
            maybe_adopt(route_seed, route_seed_costs, "stage=route_heavy_seed")
            route_channel = self._outline_congestion_soft_refine(
                self._outline_channel_soft_refine(current.clone(), benchmark, graph, plc),
                benchmark,
                graph,
                plc,
            )
            route_channel_costs = self._exact_costs(route_channel, benchmark, plc)
            maybe_adopt(route_channel, route_channel_costs, "stage=route_heavy_channel")
        elif local_tail_mode:
            local_seed = self._outline_soft_local_refine(current.clone(), benchmark, graph, plc)
            local_seed_costs = self._exact_costs(local_seed, benchmark, plc)
            maybe_adopt(local_seed, local_seed_costs, "stage=local_tail_seed")

        mode_name = "route-heavy" if route_heavy_mode else "whitespace-heavy" if whitespace_mode else "local-tail"
        print(f"[partcl:rebalance-first] {benchmark.name} mode={mode_name}")

        best = current.clone()
        best_costs = current_costs
        polished = self._iterated_soft_search_refine(current.clone(), benchmark, graph, plc)
        polished_costs = self._exact_costs(polished, benchmark, plc)
        print(
            f"[partcl:rebalance-first] {benchmark.name} final "
            f"wl={polished_costs['wirelength_cost']:.4f} "
            f"den={polished_costs['density_cost']:.4f} "
            f"cong={polished_costs['congestion_cost']:.4f} "
            f"proxy={polished_costs['proxy_cost']:.4f}"
        )
        if polished_costs["proxy_cost"] + 1.0e-4 < best_costs["proxy_cost"]:
            best = polished
            best_costs = polished_costs

        small_outer_case = benchmark.num_hard_macros <= 320 or benchmark.num_macros <= 320
        default_outer_cycles = "6" if small_outer_case else "2"
        outer_cycles = max(1, int(os.environ.get("PARTCL_REBALANCE_OUTER_CYCLES", default_outer_cycles)))
        default_outer_min_gain = "0.0025" if small_outer_case else "0.0040"
        outer_min_gain = float(os.environ.get("PARTCL_REBALANCE_OUTER_MIN_GAIN", default_outer_min_gain))
        for outer_idx in range(1, outer_cycles):
            trial = best.clone()
            trial_costs = best_costs
            cycle_start_costs = best_costs

            hard_trial = self._hard_outer_rebalance_polish(trial.clone(), benchmark, graph, plc)
            hard_trial_costs = self._exact_costs(hard_trial, benchmark, plc)
            print(
                f"[partcl:rebalance-first] {benchmark.name} outer={outer_idx + 1} stage=hard "
                f"wl={hard_trial_costs['wirelength_cost']:.4f} "
                f"den={hard_trial_costs['density_cost']:.4f} "
                f"cong={hard_trial_costs['congestion_cost']:.4f} "
                f"proxy={hard_trial_costs['proxy_cost']:.4f}"
            )
            if hard_trial_costs["proxy_cost"] + 1.0e-4 < trial_costs["proxy_cost"]:
                trial = hard_trial
                trial_costs = hard_trial_costs

            if route_heavy_mode:
                route_trial = self._outline_congestion_soft_refine(trial.clone(), benchmark, graph, plc)
                route_trial = self._outline_congestion_soft_refine(
                    self._outline_channel_soft_refine(route_trial, benchmark, graph, plc),
                    benchmark,
                    graph,
                    plc,
                )
                route_trial_costs = self._exact_costs(route_trial, benchmark, plc)
                print(
                    f"[partcl:rebalance-first] {benchmark.name} outer={outer_idx + 1} stage=route-heavy "
                    f"wl={route_trial_costs['wirelength_cost']:.4f} "
                    f"den={route_trial_costs['density_cost']:.4f} "
                    f"cong={route_trial_costs['congestion_cost']:.4f} "
                    f"proxy={route_trial_costs['proxy_cost']:.4f}"
                )
                if route_trial_costs["proxy_cost"] + 1.0e-4 < trial_costs["proxy_cost"]:
                    trial = route_trial
                    trial_costs = route_trial_costs
            elif whitespace_mode:
                ws_trial = self._outline_corridor_combo_refine(
                    trial.clone(),
                    benchmark,
                    graph,
                    plc,
                    mode="corridor_then_rebalance",
                )
                ws_trial_costs = self._exact_costs(ws_trial, benchmark, plc)
                print(
                    f"[partcl:rebalance-first] {benchmark.name} outer={outer_idx + 1} stage=whitespace-heavy "
                    f"wl={ws_trial_costs['wirelength_cost']:.4f} "
                    f"den={ws_trial_costs['density_cost']:.4f} "
                    f"cong={ws_trial_costs['congestion_cost']:.4f} "
                    f"proxy={ws_trial_costs['proxy_cost']:.4f}"
                )
                if ws_trial_costs["proxy_cost"] + 1.0e-4 < trial_costs["proxy_cost"]:
                    trial = ws_trial
                    trial_costs = ws_trial_costs
            else:
                local_trial = self._outline_soft_local_refine(trial.clone(), benchmark, graph, plc)
                local_trial_costs = self._exact_costs(local_trial, benchmark, plc)
                print(
                    f"[partcl:rebalance-first] {benchmark.name} outer={outer_idx + 1} stage=local-tail "
                    f"wl={local_trial_costs['wirelength_cost']:.4f} "
                    f"den={local_trial_costs['density_cost']:.4f} "
                    f"cong={local_trial_costs['congestion_cost']:.4f} "
                    f"proxy={local_trial_costs['proxy_cost']:.4f}"
                )
                if local_trial_costs["proxy_cost"] + 1.0e-4 < trial_costs["proxy_cost"]:
                    trial = local_trial
                    trial_costs = local_trial_costs

            repolished = self._iterated_soft_search_refine(trial.clone(), benchmark, graph, plc)
            repolished_costs = self._exact_costs(repolished, benchmark, plc)
            print(
                f"[partcl:rebalance-first] {benchmark.name} outer={outer_idx + 1} repolish "
                f"wl={repolished_costs['wirelength_cost']:.4f} "
                f"den={repolished_costs['density_cost']:.4f} "
                f"cong={repolished_costs['congestion_cost']:.4f} "
                f"proxy={repolished_costs['proxy_cost']:.4f}"
            )
            if repolished_costs["proxy_cost"] + 1.0e-4 < best_costs["proxy_cost"]:
                best = repolished
                best_costs = repolished_costs
            else:
                print(
                    f"[partcl:rebalance-first] {benchmark.name} outer={outer_idx + 1} stop=no_improvement"
                )
                break

            cycle_gain = cycle_start_costs["proxy_cost"] - best_costs["proxy_cost"]
            print(
                f"[partcl:rebalance-first] {benchmark.name} outer={outer_idx + 1} gain={cycle_gain:.4f}"
            )
            if cycle_gain < outer_min_gain:
                print(
                    f"[partcl:rebalance-first] {benchmark.name} outer={outer_idx + 1} "
                    f"stop=gain_below_threshold threshold={outer_min_gain:.4f}"
                )
                break

        return best

    def _corridor_first_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None or benchmark.num_soft_macros == 0:
            return placement

        def dc_sum(costs: dict) -> float:
            return float(costs["density_cost"] + costs["congestion_cost"])

        def adopt_candidate(
            current: torch.Tensor,
            current_costs: dict,
            trial: torch.Tensor,
            label: str,
            stage: str,
        ) -> tuple[torch.Tensor, dict]:
            trial_costs = self._exact_costs(trial, benchmark, plc)
            print(
                f"[partcl:corridor-first] {benchmark.name} stage={stage} choice={label} "
                f"wl={trial_costs['wirelength_cost']:.4f} "
                f"den={trial_costs['density_cost']:.4f} "
                f"cong={trial_costs['congestion_cost']:.4f} "
                f"proxy={trial_costs['proxy_cost']:.4f}"
            )
            proxy_gain = current_costs["proxy_cost"] - trial_costs["proxy_cost"]
            dc_gain = dc_sum(current_costs) - dc_sum(trial_costs)
            if proxy_gain >= 1.0e-4:
                return trial, trial_costs
            if dc_gain >= 0.020 and trial_costs["proxy_cost"] <= current_costs["proxy_cost"] + 0.030:
                return trial, trial_costs
            return current, current_costs

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        print(
            f"[partcl:corridor-first] {benchmark.name} start "
            f"wl={current_costs['wirelength_cost']:.4f} "
            f"den={current_costs['density_cost']:.4f} "
            f"cong={current_costs['congestion_cost']:.4f} "
            f"proxy={current_costs['proxy_cost']:.4f}"
        )

        stage_candidates = [
            ("channel", self._outline_channel_soft_refine(current.clone(), benchmark, graph, plc)),
            ("whitespace", self._outline_whitespace_soft_refine(current.clone(), benchmark, graph, plc)),
            (
                "channel_then_route",
                self._outline_congestion_soft_refine(
                    self._outline_channel_soft_refine(current.clone(), benchmark, graph, plc),
                    benchmark,
                    graph,
                    plc,
                ),
            ),
            (
                "whitespace_then_route",
                self._outline_congestion_soft_refine(
                    self._outline_whitespace_soft_refine(current.clone(), benchmark, graph, plc),
                    benchmark,
                    graph,
                    plc,
                ),
            ),
        ]

        ranked_stage = []
        for label, trial in stage_candidates:
            costs = self._exact_costs(trial, benchmark, plc)
            ranked_stage.append((costs["proxy_cost"], dc_sum(costs), label, trial, costs))
            print(
                f"[partcl:corridor-first] {benchmark.name} seed={label} "
                f"wl={costs['wirelength_cost']:.4f} "
                f"den={costs['density_cost']:.4f} "
                f"cong={costs['congestion_cost']:.4f} "
                f"proxy={costs['proxy_cost']:.4f}"
            )

        ranked_stage.sort(key=lambda item: (item[0], item[1]))
        if ranked_stage:
            best_proxy = ranked_stage[0]
            current, current_costs = adopt_candidate(current, current_costs, best_proxy[3], best_proxy[2], "seed_best_proxy")
            best_dc = min(ranked_stage, key=lambda item: item[1])
            current, current_costs = adopt_candidate(current, current_costs, best_dc[3], best_dc[2], "seed_best_dc")

        current = self._soft_region_rebalance_polish(current.clone(), benchmark, graph, plc)
        current_costs = self._exact_costs(current, benchmark, plc)
        print(
            f"[partcl:corridor-first] {benchmark.name} stage=rebalance "
            f"wl={current_costs['wirelength_cost']:.4f} "
            f"den={current_costs['density_cost']:.4f} "
            f"cong={current_costs['congestion_cost']:.4f} "
            f"proxy={current_costs['proxy_cost']:.4f}"
        )

        polished = self._iterated_soft_search_refine(current.clone(), benchmark, graph, plc)
        polished_costs = self._exact_costs(polished, benchmark, plc)
        print(
            f"[partcl:corridor-first] {benchmark.name} final "
            f"wl={polished_costs['wirelength_cost']:.4f} "
            f"den={polished_costs['density_cost']:.4f} "
            f"cong={polished_costs['congestion_cost']:.4f} "
            f"proxy={polished_costs['proxy_cost']:.4f}"
        )
        if polished_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
            return polished
        return current

    def _outline_corridor_combo_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
        mode: str,
    ) -> torch.Tensor:
        current = placement.clone()
        if plc is None or benchmark.num_soft_macros == 0:
            return current

        if mode == "whitespace_then_channel":
            current = self._outline_whitespace_soft_refine(current, benchmark, graph, plc)
            current = self._outline_channel_soft_refine(current, benchmark, graph, plc)
        elif mode == "channel_then_whitespace":
            current = self._outline_channel_soft_refine(current, benchmark, graph, plc)
            current = self._outline_whitespace_soft_refine(current, benchmark, graph, plc)
        elif mode == "corridor_then_rebalance":
            current = self._outline_channel_soft_refine(current, benchmark, graph, plc)
            current = self._outline_whitespace_soft_refine(current, benchmark, graph, plc)
            current = self._soft_region_rebalance_polish(current, benchmark, graph, plc)
        else:
            return current
        return current

    def _outline_soft_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
    ) -> torch.Tensor:
        return self._quadratic_soft_macro_follow(
            placement=placement,
            benchmark=benchmark,
            graph=graph,
            anchor_weight=0.35,
            relax=0.72,
            solver_steps=11,
        )

    def _outline_soft_local_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft == 0:
            return self._outline_soft_refine(placement, benchmark, graph)

        current = placement.clone()
        soft_targets = self._get_soft_hotspot_targets(current, benchmark, plc, limit=28)
        if not soft_targets:
            return current

        movable_soft = np.zeros(num_soft, dtype=bool)
        movable_soft[soft_targets] = True
        anchor_pos = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        return self._quadratic_soft_macro_follow(
            placement=current,
            benchmark=benchmark,
            graph=graph,
            anchor_weight=0.08,
            relax=0.70,
            solver_steps=14 if benchmark.num_macros <= 320 else 12,
            anchor_pos=anchor_pos,
            movable_soft=movable_soft,
        )

    def _outline_congestion_soft_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft == 0:
            return placement

        current = placement.clone()
        targets = self._get_soft_route_targets(current, benchmark, plc, limit=24 if benchmark.num_macros <= 320 else 18)
        if not targets:
            if os.environ.get("PARTCL_DEBUG_WHITE", "0") == "1":
                print(f"[partcl:white] {benchmark.name} no targets")
            return current

        movable_soft = np.zeros(num_soft, dtype=bool)
        target_indices = [idx for idx, _ in targets]
        movable_soft[target_indices] = True
        anchor_pos = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        current = self._quadratic_soft_macro_follow(
            placement=current,
            benchmark=benchmark,
            graph=graph,
            anchor_weight=0.04,
            relax=0.68,
            solver_steps=18 if benchmark.num_macros <= 320 else 14,
            anchor_pos=anchor_pos,
            movable_soft=movable_soft,
        )

        soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        base_soft = anchor_pos
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = (0.010 if benchmark.num_macros <= 320 else 0.008) * canvas_scale
        target_map = {idx: vec for idx, vec in targets}
        for soft_idx, route_vec in target_map.items():
            norm = np.linalg.norm(route_vec)
            if norm < 1.0e-9:
                continue
            direction = route_vec / norm
            proposed = soft[soft_idx] + direction * (0.20 * max_disp)
            disp = proposed - base_soft[soft_idx]
            disp_norm = np.linalg.norm(disp)
            if disp_norm > max_disp:
                proposed = base_soft[soft_idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
            soft[soft_idx] = proposed

        soft[:, 0] = np.clip(soft[:, 0], widths * 0.5, benchmark.canvas_width - widths * 0.5)
        soft[:, 1] = np.clip(soft[:, 1], heights * 0.5, benchmark.canvas_height - heights * 0.5)
        current[benchmark.num_hard_macros :] = torch.tensor(soft, dtype=current.dtype)
        return current

    def _outline_congestion_soft_refine_strong(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft == 0:
            return placement

        current = placement.clone()
        targets = self._get_soft_route_targets(current, benchmark, plc, limit=36 if benchmark.num_macros <= 320 else 26)
        if not targets:
            return current

        movable_soft = np.zeros(num_soft, dtype=bool)
        target_indices = [idx for idx, _ in targets]
        movable_soft[target_indices] = True
        anchor_pos = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        current = self._quadratic_soft_macro_follow(
            placement=current,
            benchmark=benchmark,
            graph=graph,
            anchor_weight=0.025,
            relax=0.66,
            solver_steps=22 if benchmark.num_macros <= 320 else 17,
            anchor_pos=anchor_pos,
            movable_soft=movable_soft,
        )

        soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        base_soft = anchor_pos
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = (0.014 if benchmark.num_macros <= 320 else 0.010) * canvas_scale
        target_map = {idx: vec for idx, vec in targets}
        for soft_idx, route_vec in target_map.items():
            norm = np.linalg.norm(route_vec)
            if norm < 1.0e-9:
                continue
            direction = route_vec / norm
            proposed = soft[soft_idx] + direction * (0.32 * max_disp)
            disp = proposed - base_soft[soft_idx]
            disp_norm = np.linalg.norm(disp)
            if disp_norm > max_disp:
                proposed = base_soft[soft_idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
            soft[soft_idx] = proposed

        soft[:, 0] = np.clip(soft[:, 0], widths * 0.5, benchmark.canvas_width - widths * 0.5)
        soft[:, 1] = np.clip(soft[:, 1], heights * 0.5, benchmark.canvas_height - heights * 0.5)
        current[benchmark.num_hard_macros :] = torch.tensor(soft, dtype=current.dtype)
        return current

    def _outline_channel_soft_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft == 0:
            return placement

        current = placement.clone()
        targets = self._get_soft_channel_targets(
            current,
            benchmark,
            plc,
            limit=28 if benchmark.num_macros <= 320 else 20,
        )
        if not targets:
            return current

        movable_soft = np.zeros(num_soft, dtype=bool)
        target_indices = [idx for idx, _ in targets]
        movable_soft[target_indices] = True
        anchor_pos = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        current = self._quadratic_soft_macro_follow(
            placement=current,
            benchmark=benchmark,
            graph=graph,
            anchor_weight=0.035,
            relax=0.67,
            solver_steps=18 if benchmark.num_macros <= 320 else 15,
            anchor_pos=anchor_pos,
            movable_soft=movable_soft,
        )

        soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        base_soft = anchor_pos
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = (0.013 if benchmark.num_macros <= 320 else 0.010) * canvas_scale
        target_map = {idx: vec for idx, vec in targets}
        for soft_idx, route_vec in target_map.items():
            norm = np.linalg.norm(route_vec)
            if norm < 1.0e-9:
                continue
            direction = route_vec / norm
            proposed = soft[soft_idx] + direction * (0.28 * max_disp)
            disp = proposed - base_soft[soft_idx]
            disp_norm = np.linalg.norm(disp)
            if disp_norm > max_disp:
                proposed = base_soft[soft_idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
            soft[soft_idx] = proposed

        soft[:, 0] = np.clip(soft[:, 0], widths * 0.5, benchmark.canvas_width - widths * 0.5)
        soft[:, 1] = np.clip(soft[:, 1], heights * 0.5, benchmark.canvas_height - heights * 0.5)
        current[benchmark.num_hard_macros :] = torch.tensor(soft, dtype=current.dtype)
        return current

    def _outline_whitespace_soft_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
        *,
        limit: int | None = None,
        region_rows: int | None = None,
        region_cols: int | None = None,
        anchor_weight: float = 0.045,
        relax: float = 0.70,
        solver_steps: int | None = None,
        max_disp_frac: float | None = None,
        step_scale: float = 0.26,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft == 0:
            return placement

        current = placement.clone()
        targets = self._get_soft_region_targets(
            current,
            benchmark,
            plc,
            limit=limit if limit is not None else 24 if benchmark.num_macros <= 320 else 18,
            region_rows=region_rows if region_rows is not None else 4 if benchmark.grid_rows >= 24 else 3,
            region_cols=region_cols if region_cols is not None else 4 if benchmark.grid_cols >= 24 else 3,
        )
        if not targets:
            return current

        movable_soft = np.zeros(num_soft, dtype=bool)
        target_indices = [idx for idx, _ in targets]
        movable_soft[target_indices] = True
        anchor_pos = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        current = self._quadratic_soft_macro_follow(
            placement=current,
            benchmark=benchmark,
            graph=graph,
            anchor_weight=anchor_weight,
            relax=relax,
            solver_steps=solver_steps if solver_steps is not None else 18 if benchmark.num_macros <= 320 else 14,
            anchor_pos=anchor_pos,
            movable_soft=movable_soft,
        )

        soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        base_soft = anchor_pos
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = (max_disp_frac if max_disp_frac is not None else 0.013 if benchmark.num_macros <= 320 else 0.010) * canvas_scale
        target_map = {idx: vec for idx, vec in targets}
        moved = 0
        for soft_idx, region_vec in target_map.items():
            norm = np.linalg.norm(region_vec)
            if norm < 1.0e-9:
                continue
            direction = region_vec / norm
            proposed = soft[soft_idx] + direction * (step_scale * max_disp)
            disp = proposed - base_soft[soft_idx]
            disp_norm = np.linalg.norm(disp)
            if disp_norm > max_disp:
                proposed = base_soft[soft_idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
            if np.linalg.norm(proposed - soft[soft_idx]) > 1.0e-9:
                moved += 1
            soft[soft_idx] = proposed

        soft[:, 0] = np.clip(soft[:, 0], widths * 0.5, benchmark.canvas_width - widths * 0.5)
        soft[:, 1] = np.clip(soft[:, 1], heights * 0.5, benchmark.canvas_height - heights * 0.5)
        current[benchmark.num_hard_macros :] = torch.tensor(soft, dtype=current.dtype)
        if os.environ.get("PARTCL_DEBUG_WHITE", "0") == "1":
            print(f"[partcl:white] {benchmark.name} targets={len(targets)} moved={moved}")
        return current

    def _outline_softopt_torch_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if num_soft == 0:
            return placement

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current = placement.clone()
        current = self._outline_soft_local_refine(current, benchmark, graph, plc)
        base_pos = current.to(device)
        soft = base_pos[benchmark.num_hard_macros :].clone().detach().requires_grad_(True)
        hard = base_pos[: benchmark.num_hard_macros].detach()
        sizes = benchmark.macro_sizes.to(device)
        half = sizes * 0.5
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        rows = benchmark.grid_rows
        cols = benchmark.grid_cols
        cell_w = canvas_w / max(cols, 1)
        cell_h = canvas_h / max(rows, 1)
        xs = (torch.arange(cols, device=device, dtype=soft.dtype) + 0.5) * cell_w
        ys = (torch.arange(rows, device=device, dtype=soft.dtype) + 0.5) * cell_h
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        soft_sizes = sizes[benchmark.num_hard_macros :]
        soft_fixed = benchmark.macro_fixed[benchmark.num_hard_macros :].to(device)
        anchor_soft = base_pos[benchmark.num_hard_macros :].detach()

        opt = torch.optim.Adam([soft], lr=0.030 if benchmark.num_macros <= 320 else 0.022)
        best = base_pos.clone().detach()
        best_score = float("inf")
        phases = [
            {
                "steps": 90 if benchmark.num_macros <= 320 else 70,
                "wl_w": 0.18,
                "den_w": 1.35,
                "cong_w": 1.05,
                "disp_w": 0.07,
                "tail_temp": 0.045,
                "lr": 0.028 if benchmark.num_macros <= 320 else 0.020,
            },
            {
                "steps": 70 if benchmark.num_macros <= 320 else 55,
                "wl_w": 0.24,
                "den_w": 0.95,
                "cong_w": 1.85,
                "disp_w": 0.10,
                "tail_temp": 0.032,
                "lr": 0.020 if benchmark.num_macros <= 320 else 0.016,
            },
        ]
        step_idx = 0
        for phase in phases:
            for group in opt.param_groups:
                group["lr"] = phase["lr"]
            for _ in range(phase["steps"]):
                step_idx += 1
                opt.zero_grad()
                soft_work = torch.where(soft_fixed[:, None], anchor_soft, soft)
                soft_x = soft_work[:, 0].clamp(
                    half[benchmark.num_hard_macros :, 0],
                    canvas_w - half[benchmark.num_hard_macros :, 0],
                )
                soft_y = soft_work[:, 1].clamp(
                    half[benchmark.num_hard_macros :, 1],
                    canvas_h - half[benchmark.num_hard_macros :, 1],
                )
                soft_work = torch.stack([soft_x, soft_y], dim=1)
                full = torch.cat([hard, soft_work], dim=0)

                wl = self._softopt_wirelength_loss(full, benchmark, device)
                density = self._softopt_density_loss(
                    soft_pos=soft_work,
                    soft_sizes=soft_sizes,
                    grid_x=grid_x,
                    grid_y=grid_y,
                    cell_w=cell_w,
                    cell_h=cell_h,
                    tail_temp=phase["tail_temp"],
                )
                congestion = self._softopt_congestion_loss(
                    full,
                    benchmark,
                    grid_x,
                    grid_y,
                    cell_w,
                    cell_h,
                    device,
                    tail_temp=phase["tail_temp"],
                )
                disp = ((soft_work - anchor_soft) ** 2).mean()
                loss = (
                    phase["wl_w"] * wl
                    + phase["den_w"] * density
                    + phase["cong_w"] * congestion
                    + phase["disp_w"] * disp
                )
                loss.backward()
                opt.step()

                if step_idx % 20 == 0 or step_idx == 1 or step_idx == sum(p["steps"] for p in phases):
                    trial = placement.clone()
                    trial[benchmark.num_hard_macros :] = soft_work.detach().cpu()
                    if plc is not None:
                        costs = self._exact_costs(trial, benchmark, plc)
                        proxy = float(costs["proxy_cost"])
                        if proxy < best_score:
                            best = trial.to(device)
                            best_score = proxy

        result = best.detach().cpu()
        if plc is not None:
            costs = self._exact_costs(result, benchmark, plc)
            print(
                f"[partcl:softopt] {benchmark.name} "
                f"wl={costs['wirelength_cost']:.4f} "
                f"den={costs['density_cost']:.4f} "
                f"cong={costs['congestion_cost']:.4f} "
                f"proxy={costs['proxy_cost']:.4f}"
            )
        return result

    def _softopt_wirelength_loss(self, placement: torch.Tensor, benchmark: Benchmark, device: torch.device) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=placement.dtype)
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        ports = benchmark.port_positions.to(device)
        for net_idx, nodes in enumerate(benchmark.net_nodes):
            idx = nodes.to(device)
            macro_mask = idx < num_macros
            pts = []
            if macro_mask.any():
                pts.append(placement[idx[macro_mask]])
            if (~macro_mask).any() and num_ports > 0:
                port_idx = idx[~macro_mask] - num_macros
                valid = (port_idx >= 0) & (port_idx < num_ports)
                if valid.any():
                    pts.append(ports[port_idx[valid]])
            if not pts:
                continue
            pts_cat = torch.cat(pts, dim=0)
            w = float(benchmark.net_weights[net_idx].item()) if net_idx < len(benchmark.net_weights) else 1.0
            beta = 8.0
            x = pts_cat[:, 0]
            y = pts_cat[:, 1]
            wl_x = torch.logsumexp(beta * x, dim=0) / beta + torch.logsumexp(-beta * x, dim=0) / beta
            wl_y = torch.logsumexp(beta * y, dim=0) / beta + torch.logsumexp(-beta * y, dim=0) / beta
            total = total + w * (wl_x + wl_y)
        return total / max(1, benchmark.num_nets)

    def _softopt_density_loss(
        self,
        soft_pos: torch.Tensor,
        soft_sizes: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        cell_w: float,
        cell_h: float,
        tail_temp: float,
    ) -> torch.Tensor:
        dx = soft_pos[:, 0][:, None, None] - grid_x[None, :, :]
        dy = soft_pos[:, 1][:, None, None] - grid_y[None, :, :]
        sigma_x = (soft_sizes[:, 0][:, None, None] + cell_w) * 0.40
        sigma_y = (soft_sizes[:, 1][:, None, None] + cell_h) * 0.40
        occ = torch.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
        area = (soft_sizes[:, 0] * soft_sizes[:, 1])[:, None, None]
        density_map = (occ * area).sum(dim=0) / max(cell_w * cell_h, 1.0e-9)
        return 0.08 * density_map.mean() + self._softopt_tail_loss(density_map, tail_temp)

    def _softopt_congestion_loss(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        cell_w: float,
        cell_h: float,
        device: torch.device,
        tail_temp: float,
    ) -> torch.Tensor:
        num_macros = benchmark.num_macros
        num_ports = benchmark.port_positions.shape[0]
        ports = benchmark.port_positions.to(device)
        demand = torch.zeros_like(grid_x)
        beta = 6.0
        eps = 1.0e-3
        for net_idx, nodes in enumerate(benchmark.net_nodes):
            idx = nodes.to(device)
            macro_mask = idx < num_macros
            pts = []
            if macro_mask.any():
                pts.append(placement[idx[macro_mask]])
            if (~macro_mask).any() and num_ports > 0:
                port_idx = idx[~macro_mask] - num_macros
                valid = (port_idx >= 0) & (port_idx < num_ports)
                if valid.any():
                    pts.append(ports[port_idx[valid]])
            if not pts:
                continue
            pts_cat = torch.cat(pts, dim=0)
            x = pts_cat[:, 0]
            y = pts_cat[:, 1]
            x_max = torch.logsumexp(beta * x, dim=0) / beta
            x_min = -torch.logsumexp(-beta * x, dim=0) / beta
            y_max = torch.logsumexp(beta * y, dim=0) / beta
            y_min = -torch.logsumexp(-beta * y, dim=0) / beta
            span_x = torch.clamp(x_max - x_min, min=eps)
            span_y = torch.clamp(y_max - y_min, min=eps)
            box = torch.sigmoid((grid_x - x_min) / cell_w) * torch.sigmoid((x_max - grid_x) / cell_w)
            box = box * torch.sigmoid((grid_y - y_min) / cell_h) * torch.sigmoid((y_max - grid_y) / cell_h)
            weight = float(benchmark.net_weights[net_idx].item()) if net_idx < len(benchmark.net_weights) else 1.0
            demand = demand + weight * box / (span_x + span_y)
        return 0.10 * demand.mean() + self._softopt_tail_loss(demand, tail_temp)

    def _softopt_tail_loss(self, grid: torch.Tensor, tail_temp: float) -> torch.Tensor:
        flat = grid.reshape(-1)
        if flat.numel() == 0:
            return torch.zeros((), device=grid.device, dtype=grid.dtype)
        centered = flat - flat.max()
        weights = torch.softmax(centered / max(tail_temp, 1.0e-4), dim=0)
        return torch.sum(weights * flat)

    def _macrobase_soft_search(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
        search_stage: int = 0,
    ) -> torch.Tensor:
        if plc is None or benchmark.num_soft_macros == 0:
            return placement

        def operator_family(name: str) -> str:
            if "channel_subset" in name and name.endswith(":route"):
                return "channel_subset_route"
            if "channel_subset" in name:
                return "channel_subset"
            if name.endswith(":route_strong") or name == "route_strong":
                return "route_strong"
            if name.endswith(":route") or name in {"route", "local_then_route", "channel_then_route"}:
                return "route"
            if name.endswith(":whitespace") or name == "whitespace":
                return "whitespace"
            if name.endswith(":channel") or name in {"channel", "local_then_channel"}:
                return "channel"
            if name.endswith(":soft") or name in {"local_then_soft", "quad_a", "quad_b"}:
                return "soft"
            if name.endswith(":strong"):
                return "strong"
            if ":subset" in name or name == "base":
                return "subset"
            return "other"

        log_candidates = os.environ.get("PARTCL_SOFT_SEARCH_LOG", "0") == "1"
        lean_soft_pool = os.environ.get("PARTCL_LEAN_SOFT_POOL", "0") == "1"
        seed_candidates: list[tuple[str, torch.Tensor]] = [("base", placement.clone())]
        local = self._outline_soft_local_refine(placement.clone(), benchmark, graph, plc)
        seed_candidates.append(("local", local))
        seed_candidates.append(("local_then_soft", self._outline_soft_refine(local.clone(), benchmark, graph)))
        seed_candidates.append(("route", self._outline_congestion_soft_refine(placement.clone(), benchmark, graph, plc)))
        seed_candidates.append(("channel", self._outline_channel_soft_refine(placement.clone(), benchmark, graph, plc)))
        seed_candidates.append(("whitespace", self._outline_whitespace_soft_refine(placement.clone(), benchmark, graph, plc)))
        seed_candidates.append(("local_then_route", self._outline_congestion_soft_refine(local.clone(), benchmark, graph, plc)))
        seed_candidates.append((
            "channel_then_route",
            self._outline_congestion_soft_refine(
                self._outline_channel_soft_refine(placement.clone(), benchmark, graph, plc),
                benchmark,
                graph,
                plc,
            ),
        ))
        seed_candidates.append((
            "quad_a",
            self._quadratic_soft_macro_follow(
                placement=placement.clone(),
                benchmark=benchmark,
                graph=graph,
                anchor_weight=0.05,
                relax=0.72,
                solver_steps=16 if benchmark.num_macros <= 320 else 13,
            ),
        ))
        if not lean_soft_pool:
            seed_candidates.append(("route_strong", self._outline_congestion_soft_refine_strong(placement.clone(), benchmark, graph, plc)))
            seed_candidates.append(("local_then_channel", self._outline_channel_soft_refine(local.clone(), benchmark, graph, plc)))
            seed_candidates.append((
                "quad_b",
                self._quadratic_soft_macro_follow(
                    placement=placement.clone(),
                    benchmark=benchmark,
                    graph=graph,
                    anchor_weight=0.10,
                    relax=0.76,
                    solver_steps=18 if benchmark.num_macros <= 320 else 14,
                ),
            ))
        if search_stage >= 1:
            channel_seed = self._outline_channel_soft_refine(placement.clone(), benchmark, graph, plc)
            seed_candidates.append(("channel_then_route_strong", self._outline_congestion_soft_refine_strong(channel_seed.clone(), benchmark, graph, plc)))
            seed_candidates.append(("whitespace_then_route", self._outline_congestion_soft_refine(
                self._outline_whitespace_soft_refine(placement.clone(), benchmark, graph, plc),
                benchmark,
                graph,
                plc,
            )))
        if search_stage >= 2:
            seed_candidates.append(("route_strong_twice", self._outline_congestion_soft_refine_strong(
                self._outline_congestion_soft_refine_strong(placement.clone(), benchmark, graph, plc),
                benchmark,
                graph,
                plc,
            )))

        best_name, best = seed_candidates[0]
        best_cost = self._exact_costs(best, benchmark, plc)
        beam: list[tuple[float, float, float, str, torch.Tensor]] = []
        seed_seen: set[bytes] = set()
        for name, trial in seed_candidates:
            trial_key = self._placement_cache_key(trial)
            if trial_key in seed_seen:
                continue
            seed_seen.add(trial_key)
            costs = self._exact_costs(trial, benchmark, plc)
            proxy = float(costs["proxy_cost"])
            dc = float(costs["density_cost"] + costs["congestion_cost"])
            cong = float(costs["congestion_cost"])
            beam.append((proxy, dc, cong, name, trial))
            if proxy + 1.0e-4 < best_cost["proxy_cost"]:
                best_name = name
                best = trial
                best_cost = costs
        if log_candidates:
            top = ", ".join(f"{name}={proxy:.4f}" for proxy, _, _, name, _ in sorted(beam, key=lambda item: item[0])[:4])
            print(f"[partcl:macrosoft] {benchmark.name} seeds {top}")

        wide_soft_search = os.environ.get("PARTCL_WIDE_SOFT_SEARCH", "0") == "1"
        if benchmark.num_macros <= 320 and wide_soft_search:
            max_rounds = 4
            beam_width = 4
        elif benchmark.num_macros <= 320:
            max_rounds = 4
            beam_width = 3
        else:
            max_rounds = 3
            beam_width = 3
        if search_stage >= 1 and benchmark.num_macros <= 320:
            beam_width = max(beam_width, 4)
        favored_ops: set[str] | None = None
        for round_idx in range(max_rounds):
            next_candidates: list[tuple[str, torch.Tensor]] = []
            for _, _, _, seed_name, seed in sorted(beam, key=lambda item: item[0])[:beam_width]:
                anchor_pos = seed[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
                if search_stage >= 1 and benchmark.num_macros <= 320:
                    refine_schedules = (
                        (8, 0.025, 0.66, 20),
                        (12, 0.040, 0.68, 20),
                        (18, 0.055, 0.70, 22),
                        (28, 0.070, 0.72, 22),
                    )
                elif not lean_soft_pool:
                    refine_schedules = (
                        (8, 0.03, 0.68, 18),
                        (12, 0.05, 0.70, 18),
                        (18, 0.06, 0.72, 20),
                        (28, 0.08, 0.74, 20),
                        (36, 0.10, 0.75, 22),
                    )
                else:
                    refine_schedules = (
                        (8, 0.03, 0.68, 18),
                        (18, 0.06, 0.72, 20),
                        (28, 0.08, 0.74, 20),
                    )
                for limit, anchor_weight, relax, steps in refine_schedules:
                    trial = self._macrobase_soft_subset_refine(
                        placement=seed.clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                        limit=limit,
                        anchor_weight=anchor_weight,
                        relax=relax,
                        solver_steps=steps if benchmark.num_macros <= 320 else max(12, steps - 3),
                        anchor_pos=anchor_pos,
                    )
                    trial_tag = f"{seed_name}:subset{limit}"
                    next_candidates.append((trial_tag, trial))
                    next_candidates.append((f"{trial_tag}:soft", self._outline_soft_refine(trial.clone(), benchmark, graph)))
                    next_candidates.append((f"{trial_tag}:route", self._outline_congestion_soft_refine(trial.clone(), benchmark, graph, plc)))
                    if favored_ops is None or "channel" in favored_ops:
                        next_candidates.append((f"{trial_tag}:channel", self._outline_channel_soft_refine(trial.clone(), benchmark, graph, plc)))
                    if favored_ops is None or "whitespace" in favored_ops:
                        next_candidates.append((f"{trial_tag}:whitespace", self._outline_whitespace_soft_refine(trial.clone(), benchmark, graph, plc)))
                    channel_trial = self._macrobase_soft_channel_subset_refine(
                        placement=seed.clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                        limit=max(6, min(32, limit)),
                        anchor_weight=max(0.02, anchor_weight * 0.75),
                        relax=max(0.66, relax - 0.03),
                        solver_steps=max(14, steps - 1) if benchmark.num_macros <= 320 else max(11, steps - 4),
                        anchor_pos=anchor_pos,
                    )
                    next_candidates.append((f"{seed_name}:channel_subset{limit}", channel_trial))
                    if favored_ops is None or "channel_subset_route" in favored_ops or "route" in favored_ops:
                        next_candidates.append((f"{seed_name}:channel_subset{limit}:route", self._outline_congestion_soft_refine(channel_trial.clone(), benchmark, graph, plc)))
                    if search_stage >= 1:
                        next_candidates.append((f"{seed_name}:channel_subset{limit}:route_strong", self._outline_congestion_soft_refine_strong(channel_trial.clone(), benchmark, graph, plc)))
                    if benchmark.num_macros <= 320 and (favored_ops is None or "channel_subset" in favored_ops):
                        for extra_limit in (max(6, limit - 4), min(40, limit + 4)):
                            if extra_limit == limit:
                                continue
                            extra_trial = self._macrobase_soft_channel_subset_refine(
                                placement=channel_trial.clone(),
                                benchmark=benchmark,
                                graph=graph,
                                plc=plc,
                                limit=extra_limit,
                                anchor_weight=max(0.018, anchor_weight * 0.68),
                                relax=max(0.64, relax - 0.04),
                                solver_steps=max(14, steps) if benchmark.num_macros <= 320 else max(11, steps - 3),
                                anchor_pos=channel_trial[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64),
                            )
                            next_candidates.append((f"{seed_name}:channel_subset{limit}:channel_subset{extra_limit}", extra_trial))
                            if favored_ops is None or "channel_subset_route" in favored_ops or "route" in favored_ops:
                                next_candidates.append(
                                    (
                                        f"{seed_name}:channel_subset{limit}:channel_subset{extra_limit}:route",
                                        self._outline_congestion_soft_refine(extra_trial.clone(), benchmark, graph, plc),
                                    )
                                )
                            if search_stage >= 2:
                                next_candidates.append(
                                    (
                                        f"{seed_name}:channel_subset{limit}:channel_subset{extra_limit}:route_strong",
                                        self._outline_congestion_soft_refine_strong(extra_trial.clone(), benchmark, graph, plc),
                                    )
                                )
                    if not lean_soft_pool and (favored_ops is None or "route_strong" in favored_ops):
                        next_candidates.append((f"{trial_tag}:route_strong", self._outline_congestion_soft_refine_strong(trial.clone(), benchmark, graph, plc)))
                if favored_ops is None or "strong" in favored_ops:
                    next_candidates.append((
                        f"{seed_name}:strong",
                        self._strong_soft_macro_follow(
                            placement=seed.clone(),
                            benchmark=benchmark,
                            graph=graph,
                        ),
                    ))

            scored_round: list[tuple[float, float, float, str, torch.Tensor]] = []
            improved = False
            round_seen: set[bytes] = set()
            for name, trial in next_candidates:
                trial_key = self._placement_cache_key(trial)
                if trial_key in round_seen:
                    continue
                round_seen.add(trial_key)
                costs = self._exact_costs(trial, benchmark, plc)
                proxy = float(costs["proxy_cost"])
                dc = float(costs["density_cost"] + costs["congestion_cost"])
                cong = float(costs["congestion_cost"])
                scored_round.append((proxy, dc, cong, name, trial))
                if proxy + 1.0e-4 < best_cost["proxy_cost"]:
                    best_name = name
                    best = trial
                    best_cost = costs
                    improved = True
            if scored_round:
                proxy_ranked = sorted(scored_round, key=lambda item: item[0])
                dc_ranked = sorted(scored_round, key=lambda item: item[1])
                cong_ranked = sorted(scored_round, key=lambda item: item[2])
                beam = []
                seen_names: set[str] = set()
                for item in proxy_ranked[:beam_width]:
                    beam.append(item)
                    seen_names.add(item[3])
                dc_slots = 2 if search_stage >= 1 and benchmark.num_macros <= 320 else 1 if benchmark.num_macros <= 320 else 0
                for item in dc_ranked:
                    if dc_slots <= 0:
                        break
                    if item[3] in seen_names:
                        continue
                    beam.append(item)
                    seen_names.add(item[3])
                    dc_slots -= 1
                for family_name in ("whitespace", "channel", "channel_subset", "channel_subset_route"):
                    family_pick = None
                    for item in dc_ranked:
                        if operator_family(item[3]) == family_name:
                            family_pick = item
                            break
                    if family_pick is None:
                        continue
                    if family_pick[3] in seen_names:
                        continue
                    beam.append(family_pick)
                    seen_names.add(family_pick[3])
                cong_slots = 1 if benchmark.num_macros <= 320 else 0
                for item in cong_ranked:
                    if cong_slots <= 0:
                        break
                    if item[3] in seen_names:
                        continue
                    beam.append(item)
                    seen_names.add(item[3])
                    cong_slots -= 1
            ranked = sorted(scored_round, key=lambda item: item[0]) if scored_round else []
            if ranked:
                favored_ops = []
                seen_families = set()
                for _, _, _, name, _ in ranked[: min(8, len(ranked))]:
                    family = operator_family(name)
                    if family in seen_families:
                        continue
                    seen_families.add(family)
                    favored_ops.append(family)
                    if len(favored_ops) >= 4:
                        break
                favored_ops = set(favored_ops)
            if log_candidates and scored_round:
                top = ", ".join(f"{name}={proxy:.4f}" for proxy, _, _, name, _ in sorted(scored_round, key=lambda item: item[0])[:5])
                favored = ",".join(sorted(favored_ops)) if favored_ops else "none"
                print(f"[partcl:macrosoft] {benchmark.name} round={round_idx + 1} top {top} favored={favored}")
            print(
                f"[partcl:macrosoft] {benchmark.name} round={round_idx + 1} "
                f"wl={best_cost['wirelength_cost']:.4f} "
                f"den={best_cost['density_cost']:.4f} "
                f"cong={best_cost['congestion_cost']:.4f} "
                f"proxy={best_cost['proxy_cost']:.4f} "
                f"best={best_name}"
            )
            if not improved:
                break
        print(
            f"[partcl:macrosoft] {benchmark.name} "
            f"wl={best_cost['wirelength_cost']:.4f} "
            f"den={best_cost['density_cost']:.4f} "
            f"cong={best_cost['congestion_cost']:.4f} "
            f"proxy={best_cost['proxy_cost']:.4f} "
            f"best={best_name}"
        )
        return best

    def _iterated_soft_search_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None or benchmark.num_soft_macros == 0:
            return placement

        current = self._macrobase_soft_search(placement.clone(), benchmark, graph, plc, search_stage=0)
        current_costs = self._exact_costs(current, benchmark, plc)
        print(
            f"[partcl:soft2] {benchmark.name} pass=1 "
            f"wl={current_costs['wirelength_cost']:.4f} "
            f"den={current_costs['density_cost']:.4f} "
            f"cong={current_costs['congestion_cost']:.4f} "
            f"proxy={current_costs['proxy_cost']:.4f}"
        )

        small_case = benchmark.num_hard_macros <= 320 or benchmark.num_macros <= 320
        default_passes = 3 if small_case else 1
        pass_count = max(
            1,
            int(os.environ.get("PARTCL_SOFT_SEARCH_PASSES", str(default_passes))),
        )

        for pass_idx in range(1, pass_count):
            trial = self._macrobase_soft_search(current.clone(), benchmark, graph, plc, search_stage=pass_idx)
            trial_costs = self._exact_costs(trial, benchmark, plc)
            print(
                f"[partcl:soft2] {benchmark.name} pass={pass_idx + 1} "
                f"wl={trial_costs['wirelength_cost']:.4f} "
                f"den={trial_costs['density_cost']:.4f} "
                f"cong={trial_costs['congestion_cost']:.4f} "
                f"proxy={trial_costs['proxy_cost']:.4f}"
            )
            if trial_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
                current = trial
                current_costs = trial_costs
            else:
                break
        if small_case:
            hard_rebalanced = self._hard_outer_rebalance_polish(current.clone(), benchmark, graph, plc)
            hard_rebalanced_costs = self._exact_costs(hard_rebalanced, benchmark, plc)
            print(
                f"[partcl:hard-balance] {benchmark.name} "
                f"wl={hard_rebalanced_costs['wirelength_cost']:.4f} "
                f"den={hard_rebalanced_costs['density_cost']:.4f} "
                f"cong={hard_rebalanced_costs['congestion_cost']:.4f} "
                f"proxy={hard_rebalanced_costs['proxy_cost']:.4f}"
            )
            if hard_rebalanced_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
                current = hard_rebalanced
                current_costs = hard_rebalanced_costs
            rebalanced = self._soft_region_rebalance_polish(current.clone(), benchmark, graph, plc)
            rebalanced_costs = self._exact_costs(rebalanced, benchmark, plc)
            print(
                f"[partcl:soft-balance] {benchmark.name} "
                f"wl={rebalanced_costs['wirelength_cost']:.4f} "
                f"den={rebalanced_costs['density_cost']:.4f} "
                f"cong={rebalanced_costs['congestion_cost']:.4f} "
                f"proxy={rebalanced_costs['proxy_cost']:.4f}"
            )
            if rebalanced_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
                current = rebalanced
                current_costs = rebalanced_costs
            polished = self._soft_coord_descent_polish(current.clone(), benchmark, graph, plc)
            polished_costs = self._exact_costs(polished, benchmark, plc)
            print(
                f"[partcl:soft-cd] {benchmark.name} "
                f"wl={polished_costs['wirelength_cost']:.4f} "
                f"den={polished_costs['density_cost']:.4f} "
                f"cong={polished_costs['congestion_cost']:.4f} "
                f"proxy={polished_costs['proxy_cost']:.4f}"
            )
            if polished_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
                current = polished
                current_costs = polished_costs
            default_local_cycles = "3" if small_case else "1"
            local_cycles = max(1, int(os.environ.get("PARTCL_SOFT_LOCAL_CYCLES", default_local_cycles)))
            for cycle_idx in range(local_cycles):
                density_polished = self._soft_density_cd_polish(current.clone(), benchmark, graph, plc)
                density_polished_costs = self._exact_costs(density_polished, benchmark, plc)
                print(
                    f"[partcl:soft-dcd] {benchmark.name} cycle={cycle_idx + 1} "
                    f"wl={density_polished_costs['wirelength_cost']:.4f} "
                    f"den={density_polished_costs['density_cost']:.4f} "
                    f"cong={density_polished_costs['congestion_cost']:.4f} "
                    f"proxy={density_polished_costs['proxy_cost']:.4f}"
                )
                if density_polished_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
                    current = density_polished
                    current_costs = density_polished_costs
                swapped = self._soft_pair_swap_polish(current.clone(), benchmark, graph, plc)
                swapped_costs = self._exact_costs(swapped, benchmark, plc)
                print(
                    f"[partcl:soft-swap] {benchmark.name} cycle={cycle_idx + 1} "
                    f"wl={swapped_costs['wirelength_cost']:.4f} "
                    f"den={swapped_costs['density_cost']:.4f} "
                    f"cong={swapped_costs['congestion_cost']:.4f} "
                    f"proxy={swapped_costs['proxy_cost']:.4f}"
                )
                if swapped_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
                    current = swapped
                    current_costs = swapped_costs
                spread = self._soft_pair_spread_polish(current.clone(), benchmark, graph, plc)
                spread_costs = self._exact_costs(spread, benchmark, plc)
                print(
                    f"[partcl:soft-spread] {benchmark.name} cycle={cycle_idx + 1} "
                    f"wl={spread_costs['wirelength_cost']:.4f} "
                    f"den={spread_costs['density_cost']:.4f} "
                    f"cong={spread_costs['congestion_cost']:.4f} "
                    f"proxy={spread_costs['proxy_cost']:.4f}"
                )
                if spread_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
                    current = spread
                    current_costs = spread_costs
        return current

    def _hard_outer_rebalance_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        if plc is None or num_hard == 0:
            return placement

        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        if not np.any(movable):
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        hard = current[:num_hard].cpu().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)
        widths = sizes[:, 0]
        heights = sizes[:, 1]
        areas = widths * heights
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        center = np.array([0.5 * cw, 0.5 * ch], dtype=np.float64)
        canvas_scale = max(cw, ch)
        self._exact_costs(current, benchmark, plc)
        nrow = benchmark.grid_rows
        ncol = benchmark.grid_cols
        density = np.asarray(plc.grid_cells, dtype=np.float64).reshape(nrow, ncol)
        h_cong = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        congestion = np.maximum(h_cong, v_cong)

        region_rows = 4 if benchmark.grid_rows >= 24 else 3
        region_cols = 4 if benchmark.grid_cols >= 24 else 3
        region_w = cw / region_cols
        region_h = ch / region_rows
        region_area = region_w * region_h
        region_hard = np.zeros((region_rows, region_cols), dtype=np.float64)
        region_soft = np.zeros((region_rows, region_cols), dtype=np.float64)
        region_density = np.zeros((region_rows, region_cols), dtype=np.float64)
        region_congestion = np.zeros((region_rows, region_cols), dtype=np.float64)
        soft = current[num_hard :].cpu().numpy().astype(np.float64)
        soft_sizes = benchmark.macro_sizes[num_hard :].cpu().numpy().astype(np.float64)
        soft_areas = soft_sizes[:, 0] * soft_sizes[:, 1] if len(soft_sizes) > 0 else np.zeros(0, dtype=np.float64)
        for i, (x, y) in enumerate(hard):
            c = min(region_cols - 1, max(0, int(x / max(region_w, 1.0e-9))))
            r = min(region_rows - 1, max(0, int(y / max(region_h, 1.0e-9))))
            region_hard[r, c] += areas[i]
        for i, (x, y) in enumerate(soft):
            c = min(region_cols - 1, max(0, int(x / max(region_w, 1.0e-9))))
            r = min(region_rows - 1, max(0, int(y / max(region_h, 1.0e-9))))
            region_soft[r, c] += soft_areas[i]

        row_edges = np.linspace(0, nrow, region_rows + 1, dtype=int)
        col_edges = np.linspace(0, ncol, region_cols + 1, dtype=int)
        for r in range(region_rows):
            r0, r1 = row_edges[r], row_edges[r + 1]
            if r1 <= r0:
                continue
            for c in range(region_cols):
                c0, c1 = col_edges[c], col_edges[c + 1]
                if c1 <= c0:
                    continue
                region_density[r, c] = float(np.mean(density[r0:r1, c0:c1]))
                region_congestion[r, c] = float(np.mean(congestion[r0:r1, c0:c1]))

        soft_ratio = region_soft / max(region_area, 1.0e-9)
        hard_ratio = region_hard / max(region_area, 1.0e-9)
        overload_score = (
            1.45 * region_congestion
            + 0.95 * region_density
            + 0.55 * soft_ratio
            + 0.18 * hard_ratio
        )
        free_score = region_area - region_hard - 0.72 * region_soft - 0.22 * overload_score * region_area
        br_bonus = np.zeros_like(overload_score)
        br_bonus[-1, -1] = 0.60
        if region_cols >= 3:
            br_bonus[-1, -2] = 0.28
        if region_rows >= 3:
            br_bonus[-2, -1] = 0.28
        overloaded_regions = []
        for r in range(region_rows):
            for c in range(region_cols):
                overloaded_regions.append((float(overload_score[r, c] + br_bonus[r, c]), r, c))
        overloaded_regions.sort(reverse=True)
        outer_regions = []
        for r in range(region_rows):
            for c in range(region_cols):
                dist = abs(r - (region_rows - 1) * 0.5) + abs(c - (region_cols - 1) * 0.5)
                if dist < 1.0:
                    continue
                outer_regions.append((float(free_score[r, c]), r, c))
        outer_regions.sort(reverse=True)
        sink_regions = [(score, r, c) for score, r, c in overloaded_regions[:4] if score > 0.0]
        receiver_regions = [(score, r, c) for score, r, c in outer_regions if score > 0.0][:6]
        if not receiver_regions:
            return current

        hotspot_targets = self._get_hotspot_targets(
            placement=current,
            benchmark=benchmark,
            plc=plc,
            widths=widths,
            heights=heights,
            movable=movable,
            limit=12 if benchmark.num_macros <= 320 else 8,
        )
        hotspot_priority = {idx: rank for rank, (idx, _) in enumerate(hotspot_targets)}
        center_dist = np.linalg.norm(hard - center[None, :], axis=1)
        center_scale = max(0.25 * canvas_scale, 1.0e-9)
        area_scale = max(float(np.max(areas)), 1.0e-9)
        sink_lookup = {(r, c): rank for rank, (_, r, c) in enumerate(sink_regions)}
        local_soft_area = np.zeros(num_hard, dtype=np.float64)
        for idx in range(num_hard):
            margin_x = 0.80 * widths[idx] + 0.90 * region_w
            margin_y = 0.80 * heights[idx] + 0.90 * region_h
            for soft_idx in range(len(soft)):
                if abs(soft[soft_idx, 0] - hard[idx, 0]) <= margin_x and abs(soft[soft_idx, 1] - hard[idx, 1]) <= margin_y:
                    local_soft_area[idx] += soft_areas[soft_idx]
        soft_area_scale = max(float(np.max(local_soft_area)), 1.0e-9)
        candidate_order = []
        for idx in range(num_hard):
            if not movable[idx]:
                continue
            region_c = min(region_cols - 1, max(0, int(hard[idx, 0] / max(region_w, 1.0e-9))))
            region_r = min(region_rows - 1, max(0, int(hard[idx, 1] / max(region_h, 1.0e-9))))
            centrality = max(0.0, 1.0 - center_dist[idx] / center_scale)
            hotspot_bonus = 0.0
            if idx in hotspot_priority:
                hotspot_bonus = 1.0 - hotspot_priority[idx] / max(len(hotspot_targets), 1)
            sink_bonus = 0.0
            if (region_r, region_c) in sink_lookup:
                sink_bonus = 1.0 - sink_lookup[(region_r, region_c)] / max(len(sink_regions), 1)
            corner_bonus = 0.0
            if region_r >= region_rows - 2 and region_c >= region_cols - 2:
                corner_bonus = 0.55
            soft_bonus = local_soft_area[idx] / soft_area_scale
            if centrality <= 0.0 and hotspot_bonus <= 0.0 and sink_bonus <= 0.0:
                continue
            score = (areas[idx] / area_scale) * (
                0.22
                + 0.20 * centrality
                + 0.48 * hotspot_bonus
                + 0.55 * sink_bonus
                + 0.22 * corner_bonus
                + 0.28 * soft_bonus
            )
            candidate_order.append((score, idx))
        candidate_order.sort(reverse=True)
        candidate_indices = [idx for _, idx in candidate_order[:8]]
        if not candidate_indices:
            return current

        max_disp = 0.46 * canvas_scale if benchmark.num_macros <= 320 else 0.34 * canvas_scale
        best_trial = None
        best_costs = current_costs

        def _run_hard_trial(base_hard: np.ndarray, updates: list[tuple[int, np.ndarray]]) -> None:
            nonlocal best_trial, best_costs
            trial = current.clone()
            trial_hard = base_hard.copy()
            moved = False
            for idx, target in updates:
                base = hard[idx]
                proposed = 0.10 * base + 0.90 * target
                disp = proposed - base
                disp_norm = float(np.linalg.norm(disp))
                if disp_norm > max_disp:
                    proposed = base + disp * (max_disp / max(disp_norm, 1.0e-9))
                proposed[0] = np.clip(proposed[0], widths[idx] * 0.5, cw - widths[idx] * 0.5)
                proposed[1] = np.clip(proposed[1], heights[idx] * 0.5, ch - heights[idx] * 0.5)
                trial_hard[idx] = proposed
                moved = True
            if not moved:
                return
            trial_hard = self._legalize_hard_numpy(trial_hard, benchmark)
            total_move = 0.0
            for idx, _ in updates:
                total_move += float(np.linalg.norm(trial_hard[idx] - hard[idx]))
            if total_move < 0.08 * canvas_scale:
                return
            trial[:num_hard] = torch.tensor(trial_hard, dtype=trial.dtype)
            trial = self._outline_soft_local_refine(trial, benchmark, graph, plc)
            costs = self._exact_costs(trial, benchmark, plc)
            if costs["proxy_cost"] + 1.0e-4 < best_costs["proxy_cost"]:
                best_trial = trial
                best_costs = costs

        base_hard = hard.copy()

        # Stronger single-macro evacuations toward the best receiver regions.
        for idx in candidate_indices[:5]:
            for _, rr, rc in receiver_regions[:5]:
                target = np.array([(rc + 0.5) * region_w, (rr + 0.5) * region_h], dtype=np.float64)
                _run_hard_trial(base_hard, [(idx, target)])

        # Directed escape moves from the most overloaded sink, especially bottom-right congestion.
        if sink_regions:
            lead_score, lead_r, lead_c = sink_regions[0]
            sink_center = np.array([(lead_c + 0.5) * region_w, (lead_r + 0.5) * region_h], dtype=np.float64)
            bottom_right = np.array([0.90 * cw, 0.10 * ch], dtype=np.float64)
            preferred_receivers = receiver_regions[:4]
            for idx in candidate_indices[:5]:
                region_c = min(region_cols - 1, max(0, int(hard[idx, 0] / max(region_w, 1.0e-9))))
                region_r = min(region_rows - 1, max(0, int(hard[idx, 1] / max(region_h, 1.0e-9))))
                if abs(region_r - lead_r) > 1 or abs(region_c - lead_c) > 1:
                    continue
                away = hard[idx] - sink_center
                if np.linalg.norm(away) < 1.0e-9:
                    away = hard[idx] - center
                if region_r >= region_rows - 2 and region_c >= region_cols - 2:
                    away = away + 0.90 * (hard[idx] - bottom_right) + np.array([-0.55 * cw, 0.55 * ch], dtype=np.float64)
                norm = np.linalg.norm(away)
                if norm > 1.0e-9:
                    for frac in (0.22, 0.34, 0.46):
                        target = hard[idx] + away / norm * (frac * canvas_scale)
                        _run_hard_trial(base_hard, [(idx, target)])
                for _, rr, rc in preferred_receivers:
                    target = np.array([(rc + 0.5) * region_w, (rr + 0.5) * region_h], dtype=np.float64)
                    _run_hard_trial(base_hard, [(idx, target)])

        # Paired moves to open broader channels instead of just relocating one blocker.
        pair_regions = [(r, c) for _, r, c in receiver_regions[:4]]
        for a_pos, idx_a in enumerate(candidate_indices[:4]):
            for idx_b in candidate_indices[a_pos + 1 : a_pos + 4]:
                for region_a in pair_regions[:3]:
                    for region_b in pair_regions[:3]:
                        if region_a == region_b:
                            continue
                        target_a = np.array(
                            [(region_a[1] + 0.5) * region_w, (region_a[0] + 0.5) * region_h],
                            dtype=np.float64,
                        )
                        target_b = np.array(
                            [(region_b[1] + 0.5) * region_w, (region_b[0] + 0.5) * region_h],
                            dtype=np.float64,
                        )
                        _run_hard_trial(base_hard, [(idx_a, target_a), (idx_b, target_b)])

        # Sideways and diagonal evacuation for oversized hotspot blockers.
        for idx in candidate_indices[:4]:
            base = hard[idx]
            lateral_targets = [
                np.array([0.18 * cw, np.clip(base[1], 0.20 * ch, 0.80 * ch)], dtype=np.float64),
                np.array([0.82 * cw, np.clip(base[1], 0.20 * ch, 0.80 * ch)], dtype=np.float64),
                np.array([np.clip(base[0], 0.18 * cw, 0.82 * cw), 0.18 * ch], dtype=np.float64),
                np.array([np.clip(base[0], 0.18 * cw, 0.82 * cw), 0.82 * ch], dtype=np.float64),
                np.array([0.22 * cw, 0.78 * ch], dtype=np.float64),
                np.array([0.30 * cw, 0.70 * ch], dtype=np.float64),
            ]
            for target in lateral_targets:
                _run_hard_trial(base_hard, [(idx, target)])

        if best_trial is not None:
            return best_trial
        return current

    def _outline_soft_search_plus(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None or benchmark.num_soft_macros == 0:
            return placement

        best = self._macrobase_soft_search(placement.clone(), benchmark, graph, plc)
        best_cost = self._exact_costs(best, benchmark, plc)
        beam = [best.clone()]
        beam.append(self._outline_soft_local_refine(placement.clone(), benchmark, graph, plc))

        round_count = max(1, int(os.environ.get("PARTCL_SOFT_CD_ROUNDS", "2")))
        for round_idx in range(round_count):
            next_candidates: list[torch.Tensor] = []
            for seed in beam[:2]:
                anchor_pos = seed[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
                for limit, anchor_weight, relax, steps in (
                    (10, 0.025, 0.66, 20),
                    (16, 0.035, 0.68, 22),
                    (24, 0.050, 0.70, 22),
                    (36, 0.070, 0.72, 24),
                ):
                    trial = self._macrobase_soft_subset_refine(
                        placement=seed.clone(),
                        benchmark=benchmark,
                        graph=graph,
                        plc=plc,
                        limit=limit,
                        anchor_weight=anchor_weight,
                        relax=relax,
                        solver_steps=steps if benchmark.num_macros <= 320 else max(14, steps - 4),
                        anchor_pos=anchor_pos,
                    )
                    next_candidates.append(trial)
                    next_candidates.append(
                        self._quadratic_soft_macro_follow(
                            placement=trial.clone(),
                            benchmark=benchmark,
                            graph=graph,
                            anchor_weight=max(0.02, anchor_weight * 0.75),
                            relax=min(0.78, relax + 0.04),
                            solver_steps=max(12, steps - 4),
                        )
                    )

            scored: list[tuple[float, torch.Tensor]] = []
            improved = False
            for trial in next_candidates:
                costs = self._exact_costs(trial, benchmark, plc)
                proxy = float(costs["proxy_cost"])
                scored.append((proxy, trial))
                if proxy + 1.0e-4 < best_cost["proxy_cost"]:
                    best = trial
                    best_cost = costs
                    improved = True
            beam = [trial for _, trial in sorted(scored, key=lambda item: item[0])[:2]] if scored else beam
            print(
                f"[partcl:outline-soft+] {benchmark.name} round={round_idx + 1} "
                f"wl={best_cost['wirelength_cost']:.4f} "
                f"den={best_cost['density_cost']:.4f} "
                f"cong={best_cost['congestion_cost']:.4f} "
                f"proxy={best_cost['proxy_cost']:.4f}"
            )
            if not improved:
                break

        return best

    def _macrobase_soft_subset_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
        limit: int,
        anchor_weight: float,
        relax: float,
        solver_steps: int,
        anchor_pos: np.ndarray,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        targets = self._get_soft_hotspot_targets(placement, benchmark, plc, limit=limit)
        if not targets:
            return placement
        movable_soft = np.zeros(num_soft, dtype=bool)
        movable_soft[targets] = True
        return self._quadratic_soft_macro_follow(
            placement=placement,
            benchmark=benchmark,
            graph=graph,
            anchor_weight=anchor_weight,
            relax=relax,
            solver_steps=solver_steps,
            anchor_pos=anchor_pos,
            movable_soft=movable_soft,
        )

    def _macrobase_soft_channel_subset_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
        limit: int,
        anchor_weight: float,
        relax: float,
        solver_steps: int,
        anchor_pos: np.ndarray,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        targets = self._get_soft_channel_targets(placement, benchmark, plc, limit=limit)
        if not targets:
            return placement

        movable_soft = np.zeros(num_soft, dtype=bool)
        target_indices = [idx for idx, _ in targets]
        movable_soft[target_indices] = True
        current = self._quadratic_soft_macro_follow(
            placement=placement,
            benchmark=benchmark,
            graph=graph,
            anchor_weight=anchor_weight,
            relax=relax,
            solver_steps=solver_steps,
            anchor_pos=anchor_pos,
            movable_soft=movable_soft,
        )

        soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        base_soft = np.asarray(anchor_pos, dtype=np.float64)
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = (0.012 if benchmark.num_macros <= 320 else 0.009) * canvas_scale
        target_map = {idx: vec for idx, vec in targets}
        for soft_idx, channel_vec in target_map.items():
            norm = np.linalg.norm(channel_vec)
            if norm < 1.0e-9:
                continue
            direction = channel_vec / norm
            proposed = soft[soft_idx] + direction * (0.24 * max_disp)
            disp = proposed - base_soft[soft_idx]
            disp_norm = np.linalg.norm(disp)
            if disp_norm > max_disp:
                proposed = base_soft[soft_idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
            soft[soft_idx] = proposed

        soft[:, 0] = np.clip(soft[:, 0], widths * 0.5, benchmark.canvas_width - widths * 0.5)
        soft[:, 1] = np.clip(soft[:, 1], heights * 0.5, benchmark.canvas_height - heights * 0.5)
        current[benchmark.num_hard_macros :] = torch.tensor(soft, dtype=current.dtype)
        return current

    def _soft_coord_descent_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft == 0:
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = (0.010 if benchmark.num_macros <= 320 else 0.008) * canvas_scale

        for round_idx in range(2):
            route_targets = {idx: vec for idx, vec in self._get_soft_route_targets(current, benchmark, plc, limit=18)}
            channel_targets = {idx: vec for idx, vec in self._get_soft_channel_targets(current, benchmark, plc, limit=18)}
            region_targets = {
                idx: vec
                for idx, vec in self._get_soft_region_targets(
                    current,
                    benchmark,
                    plc,
                    limit=14,
                    region_rows=4 if benchmark.grid_rows >= 24 else 3,
                    region_cols=4 if benchmark.grid_cols >= 24 else 3,
                )
            }
            hotspot_targets = list(self._get_soft_hotspot_targets(current, benchmark, plc, limit=18))

            priority = []
            seen = set()
            for idx in hotspot_targets:
                if idx not in seen:
                    priority.append(idx)
                    seen.add(idx)
            for mapping in (route_targets, channel_targets, region_targets):
                for idx in mapping:
                    if idx not in seen:
                        priority.append(idx)
                        seen.add(idx)
            if not priority:
                break

            base_soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
            best_trial = None
            best_trial_costs = current_costs
            for soft_idx in priority[:12]:
                direction = np.zeros(2, dtype=np.float64)
                if soft_idx in route_targets:
                    direction += 1.20 * np.asarray(route_targets[soft_idx], dtype=np.float64)
                if soft_idx in channel_targets:
                    direction += 0.95 * np.asarray(channel_targets[soft_idx], dtype=np.float64)
                if soft_idx in region_targets:
                    direction += 0.70 * np.asarray(region_targets[soft_idx], dtype=np.float64)
                norm = np.linalg.norm(direction)
                if norm < 1.0e-9:
                    continue
                direction = direction / norm
                axis_dir = direction.copy()
                if abs(axis_dir[0]) >= abs(axis_dir[1]):
                    axis_dir[1] = 0.0
                    axis_dir[0] = np.sign(axis_dir[0]) if abs(axis_dir[0]) > 1.0e-9 else 0.0
                else:
                    axis_dir[0] = 0.0
                    axis_dir[1] = np.sign(axis_dir[1]) if abs(axis_dir[1]) > 1.0e-9 else 0.0
                perp_dir = np.array([-direction[1], direction[0]], dtype=np.float64)
                candidate_dirs = [direction]
                if np.linalg.norm(axis_dir) > 1.0e-9:
                    candidate_dirs.append(axis_dir)
                if np.linalg.norm(perp_dir) > 1.0e-9:
                    candidate_dirs.append(perp_dir / max(np.linalg.norm(perp_dir), 1.0e-9))

                candidate_steps = (
                    0.12 * max_disp,
                    0.20 * max_disp,
                    0.30 * max_disp,
                )
                for move_dir in candidate_dirs:
                    for step in candidate_steps:
                        for sign in (1.0, -0.45):
                            trial = current.clone()
                            soft = trial[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
                            proposed = soft[soft_idx] + sign * move_dir * step
                            proposed[0] = np.clip(
                                proposed[0],
                                widths[soft_idx] * 0.5,
                                benchmark.canvas_width - widths[soft_idx] * 0.5,
                            )
                            proposed[1] = np.clip(
                                proposed[1],
                                heights[soft_idx] * 0.5,
                                benchmark.canvas_height - heights[soft_idx] * 0.5,
                            )
                            disp = proposed - base_soft[soft_idx]
                            disp_norm = np.linalg.norm(disp)
                            if disp_norm > max_disp:
                                proposed = base_soft[soft_idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
                            soft[soft_idx] = proposed
                            trial[benchmark.num_hard_macros :] = torch.tensor(soft, dtype=trial.dtype)
                            costs = self._exact_costs(trial, benchmark, plc)
                            if costs["proxy_cost"] + 1.0e-4 < best_trial_costs["proxy_cost"]:
                                best_trial = trial
                                best_trial_costs = costs

            if best_trial is None:
                break
            current = best_trial
            current_costs = best_trial_costs
            print(
                f"[partcl:soft-cd] {benchmark.name} round={round_idx + 1} "
                f"wl={current_costs['wirelength_cost']:.4f} "
                f"den={current_costs['density_cost']:.4f} "
                f"cong={current_costs['congestion_cost']:.4f} "
                f"proxy={current_costs['proxy_cost']:.4f}"
            )
        return current

    def _soft_region_rebalance_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft == 0:
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        num_hard = benchmark.num_hard_macros
        soft = current[num_hard :].cpu().numpy().astype(np.float64)
        hard = current[:num_hard].cpu().numpy().astype(np.float64)
        soft_sizes = benchmark.macro_sizes[num_hard :].cpu().numpy().astype(np.float64)
        hard_sizes = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)
        soft_areas = soft_sizes[:, 0] * soft_sizes[:, 1]
        hard_areas = hard_sizes[:, 0] * hard_sizes[:, 1]

        region_rows = 4 if benchmark.grid_rows >= 24 else 3
        region_cols = 4 if benchmark.grid_cols >= 24 else 3
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        region_w = cw / region_cols
        region_h = ch / region_rows
        region_area = region_w * region_h

        region_hard = np.zeros((region_rows, region_cols), dtype=np.float64)
        region_soft = np.zeros((region_rows, region_cols), dtype=np.float64)
        soft_region = np.zeros((num_soft, 2), dtype=np.int64)

        for i, (x, y) in enumerate(hard):
            c = min(region_cols - 1, max(0, int(x / max(region_w, 1.0e-9))))
            r = min(region_rows - 1, max(0, int(y / max(region_h, 1.0e-9))))
            region_hard[r, c] += hard_areas[i]
        for i, (x, y) in enumerate(soft):
            c = min(region_cols - 1, max(0, int(x / max(region_w, 1.0e-9))))
            r = min(region_rows - 1, max(0, int(y / max(region_h, 1.0e-9))))
            soft_region[i] = (r, c)
            region_soft[r, c] += soft_areas[i]

        free_area = np.maximum(region_area - region_hard, 1.0e-9)
        total_soft_area = float(np.sum(soft_areas))
        target_soft = free_area / np.sum(free_area) * total_soft_area
        surplus = region_soft - target_soft

        donor_order = np.dstack(np.unravel_index(np.argsort(-surplus, axis=None), surplus.shape))[0]
        recv_order = np.dstack(np.unravel_index(np.argsort(surplus, axis=None), surplus.shape))[0]
        donors = [(int(r), int(c), float(surplus[r, c])) for r, c in donor_order if surplus[r, c] > 1.0e-6]
        receivers = [(int(r), int(c), float(-surplus[r, c])) for r, c in recv_order if surplus[r, c] < -1.0e-6]
        if not donors or not receivers:
            return current

        widths = soft_sizes[:, 0]
        heights = soft_sizes[:, 1]
        canvas_scale = max(cw, ch)
        small_case = benchmark.num_macros <= 320
        max_disp = (0.34 if small_case else 0.26) * canvas_scale
        donor_cap = 6 if small_case else 4
        recv_cap = 6 if small_case else 4
        donor_pick_cap = 8 if small_case else 6
        trial_cap = 40 if small_case else 24
        bundle_sizes = (3, 2, 1) if small_case else (2, 1)
        blend_schedule = (0.72, 0.86) if small_case else (0.68,)
        offset_scale = 0.18 if small_case else 0.12

        best_trial = None
        best_trial_costs = current_costs
        trials_checked = 0
        for dr, dc, donor_surplus in donors[:donor_cap]:
            donor_indices = [i for i, (r, c) in enumerate(soft_region) if int(r) == dr and int(c) == dc]
            if not donor_indices:
                continue
            donor_center = np.array([(dc + 0.5) * region_w, (dr + 0.5) * region_h], dtype=np.float64)
            donor_indices.sort(
                key=lambda i: float(np.linalg.norm(soft[i] - donor_center)) * (1.0 + 0.20 * soft_areas[i] / max(np.mean(soft_areas), 1.0e-9)),
                reverse=True,
            )
            donor_indices = donor_indices[:donor_pick_cap]
            for rr, rc, recv_deficit in receivers[:recv_cap]:
                recv_center = np.array([(rc + 0.5) * region_w, (rr + 0.5) * region_h], dtype=np.float64)
                transfer_goal = min(donor_surplus, recv_deficit)
                if transfer_goal <= 1.0e-6:
                    continue
                preferred = []
                area_accum = 0.0
                for soft_idx in donor_indices:
                    preferred.append(soft_idx)
                    area_accum += soft_areas[soft_idx]
                    if area_accum >= 1.15 * transfer_goal:
                        break
                if not preferred:
                    continue
                for bundle_size in bundle_sizes:
                    chosen = preferred[:bundle_size]
                    if len(chosen) < bundle_size:
                        continue
                    for blend in blend_schedule:
                        trial = current.clone()
                        trial_soft = trial[num_hard :].cpu().numpy().astype(np.float64)
                        for local_idx, soft_idx in enumerate(chosen):
                            angle = (2.0 * np.pi * local_idx) / max(len(chosen), 1)
                            offset = np.array(
                                [
                                    np.cos(angle) * offset_scale * region_w,
                                    np.sin(angle) * offset_scale * region_h,
                                ],
                                dtype=np.float64,
                            )
                            target = recv_center + offset
                            proposed = (1.0 - blend) * trial_soft[soft_idx] + blend * target
                            disp = proposed - soft[soft_idx]
                            disp_norm = float(np.linalg.norm(disp))
                            if disp_norm > max_disp:
                                proposed = soft[soft_idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
                            proposed[0] = np.clip(proposed[0], widths[soft_idx] * 0.5, cw - widths[soft_idx] * 0.5)
                            proposed[1] = np.clip(proposed[1], heights[soft_idx] * 0.5, ch - heights[soft_idx] * 0.5)
                            trial_soft[soft_idx] = proposed
                        trial[num_hard :] = torch.tensor(trial_soft, dtype=trial.dtype)
                        costs = self._exact_costs(trial, benchmark, plc)
                        trials_checked += 1
                        if costs["proxy_cost"] + 1.0e-4 < best_trial_costs["proxy_cost"]:
                            best_trial = trial
                            best_trial_costs = costs
                        if trials_checked >= trial_cap:
                            break
                    if trials_checked >= trial_cap:
                        break
                if trials_checked >= trial_cap:
                    break
            if trials_checked >= trial_cap:
                break

        if best_trial is not None:
            print(
                f"[partcl:soft-balance] {benchmark.name} round=1 "
                f"wl={best_trial_costs['wirelength_cost']:.4f} "
                f"den={best_trial_costs['density_cost']:.4f} "
                f"cong={best_trial_costs['congestion_cost']:.4f} "
                f"proxy={best_trial_costs['proxy_cost']:.4f}"
            )
            return best_trial
        return current

    def _soft_pair_swap_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft < 2:
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)

        swap_rounds = 2 if benchmark.num_macros <= 320 else 1
        candidate_cap = 12 if benchmark.num_macros <= 320 else 8
        for round_idx in range(swap_rounds):
            hotspot_targets = list(self._get_soft_hotspot_targets(current, benchmark, plc, limit=12))
            route_targets = [idx for idx, _ in self._get_soft_route_targets(current, benchmark, plc, limit=12)]
            channel_targets = [idx for idx, _ in self._get_soft_channel_targets(current, benchmark, plc, limit=12)]
            priority = []
            seen = set()
            for idx in hotspot_targets + route_targets + channel_targets:
                if idx not in seen:
                    priority.append(idx)
                    seen.add(idx)
            if len(priority) < 2:
                break

            soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
            best_trial = None
            best_trial_costs = current_costs
            candidate_indices = priority[:candidate_cap]
            for i in range(len(candidate_indices)):
                for j in range(i + 1, len(candidate_indices)):
                    a = candidate_indices[i]
                    b = candidate_indices[j]
                    trial = current.clone()
                    trial_soft = soft.copy()
                    pa = trial_soft[a].copy()
                    pb = trial_soft[b].copy()
                    swap_a = np.array(
                        [
                            np.clip(pb[0], widths[a] * 0.5, benchmark.canvas_width - widths[a] * 0.5),
                            np.clip(pb[1], heights[a] * 0.5, benchmark.canvas_height - heights[a] * 0.5),
                        ],
                        dtype=np.float64,
                    )
                    swap_b = np.array(
                        [
                            np.clip(pa[0], widths[b] * 0.5, benchmark.canvas_width - widths[b] * 0.5),
                            np.clip(pa[1], heights[b] * 0.5, benchmark.canvas_height - heights[b] * 0.5),
                        ],
                        dtype=np.float64,
                    )
                    trial_soft[a] = swap_a
                    trial_soft[b] = swap_b
                    trial[benchmark.num_hard_macros :] = torch.tensor(trial_soft, dtype=trial.dtype)
                    costs = self._exact_costs(trial, benchmark, plc)
                    if costs["proxy_cost"] + 1.0e-4 < best_trial_costs["proxy_cost"]:
                        best_trial = trial
                        best_trial_costs = costs

            if best_trial is None:
                break
            current = best_trial
            current_costs = best_trial_costs
            print(
                f"[partcl:soft-swap] {benchmark.name} round={round_idx + 1} "
                f"wl={best_trial_costs['wirelength_cost']:.4f} "
                f"den={best_trial_costs['density_cost']:.4f} "
                f"cong={best_trial_costs['congestion_cost']:.4f} "
                f"proxy={best_trial_costs['proxy_cost']:.4f}"
            )
        if current_costs["proxy_cost"] + 1.0e-4 < self._exact_costs(placement, benchmark, plc)["proxy_cost"]:
            return current
        return current

    def _soft_density_cd_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft == 0:
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = (0.009 if benchmark.num_macros <= 320 else 0.007) * canvas_scale

        for round_idx in range(2):
            hotspot_targets = list(self._get_soft_hotspot_targets(current, benchmark, plc, limit=20))
            region_targets = {
                idx: vec
                for idx, vec in self._get_soft_region_targets(
                    current,
                    benchmark,
                    plc,
                    limit=18,
                    region_rows=4 if benchmark.grid_rows >= 24 else 3,
                    region_cols=4 if benchmark.grid_cols >= 24 else 3,
                )
            }
            if not hotspot_targets and not region_targets:
                break

            priority = []
            seen = set()
            for idx in hotspot_targets:
                if idx not in seen:
                    priority.append(idx)
                    seen.add(idx)
            for idx in region_targets:
                if idx not in seen:
                    priority.append(idx)
                    seen.add(idx)

            base_soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
            best_trial = None
            best_trial_costs = current_costs
            for soft_idx in priority[:10]:
                direction = np.zeros(2, dtype=np.float64)
                if soft_idx in region_targets:
                    direction += np.asarray(region_targets[soft_idx], dtype=np.float64)
                if soft_idx in hotspot_targets:
                    center = base_soft[soft_idx]
                    direction += 0.6 * np.array(
                        [
                            center[0] - 0.5 * benchmark.canvas_width,
                            center[1] - 0.5 * benchmark.canvas_height,
                        ],
                        dtype=np.float64,
                    )
                norm = np.linalg.norm(direction)
                if norm < 1.0e-9:
                    continue
                direction = direction / norm
                candidate_dirs = [direction, np.array([np.sign(direction[0]), 0.0], dtype=np.float64), np.array([0.0, np.sign(direction[1])], dtype=np.float64)]
                for move_dir in candidate_dirs:
                    if np.linalg.norm(move_dir) < 1.0e-9:
                        continue
                    move_dir = move_dir / max(np.linalg.norm(move_dir), 1.0e-9)
                    for step in (0.10 * max_disp, 0.18 * max_disp, 0.28 * max_disp):
                        trial = current.clone()
                        soft = trial[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
                        proposed = soft[soft_idx] + move_dir * step
                        proposed[0] = np.clip(
                            proposed[0],
                            widths[soft_idx] * 0.5,
                            benchmark.canvas_width - widths[soft_idx] * 0.5,
                        )
                        proposed[1] = np.clip(
                            proposed[1],
                            heights[soft_idx] * 0.5,
                            benchmark.canvas_height - heights[soft_idx] * 0.5,
                        )
                        disp = proposed - base_soft[soft_idx]
                        disp_norm = np.linalg.norm(disp)
                        if disp_norm > max_disp:
                            proposed = base_soft[soft_idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
                        soft[soft_idx] = proposed
                        trial[benchmark.num_hard_macros :] = torch.tensor(soft, dtype=trial.dtype)
                        costs = self._exact_costs(trial, benchmark, plc)
                        if costs["proxy_cost"] + 1.0e-4 < best_trial_costs["proxy_cost"]:
                            best_trial = trial
                            best_trial_costs = costs
            if best_trial is None:
                break
            current = best_trial
            current_costs = best_trial_costs
            print(
                f"[partcl:soft-dcd] {benchmark.name} round={round_idx + 1} "
                f"wl={current_costs['wirelength_cost']:.4f} "
                f"den={current_costs['density_cost']:.4f} "
                f"cong={current_costs['congestion_cost']:.4f} "
                f"proxy={current_costs['proxy_cost']:.4f}"
            )
        return current

    def _soft_pair_spread_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        num_soft = benchmark.num_soft_macros
        if plc is None or num_soft < 2:
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        widths = benchmark.macro_sizes[benchmark.num_hard_macros :, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[benchmark.num_hard_macros :, 1].cpu().numpy().astype(np.float64)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = (0.008 if benchmark.num_macros <= 320 else 0.006) * canvas_scale

        hotspot_targets = list(self._get_soft_hotspot_targets(current, benchmark, plc, limit=12))
        route_targets = {idx: vec for idx, vec in self._get_soft_route_targets(current, benchmark, plc, limit=10)}
        channel_targets = {idx: vec for idx, vec in self._get_soft_channel_targets(current, benchmark, plc, limit=10)}
        priority = []
        seen = set()
        for idx in hotspot_targets:
            if idx not in seen:
                priority.append(idx)
                seen.add(idx)
        for idx in list(route_targets.keys()) + list(channel_targets.keys()):
            if idx not in seen:
                priority.append(idx)
                seen.add(idx)
        if len(priority) < 2:
            return current

        soft = current[benchmark.num_hard_macros :].cpu().numpy().astype(np.float64)
        best_trial = None
        best_trial_costs = current_costs
        candidate_indices = priority[:10]
        for i in range(len(candidate_indices)):
            for j in range(i + 1, len(candidate_indices)):
                a = candidate_indices[i]
                b = candidate_indices[j]
                pair_vec = soft[a] - soft[b]
                norm = np.linalg.norm(pair_vec)
                if norm < 1.0e-9:
                    pair_vec = np.zeros(2, dtype=np.float64)
                    pair_vec[0] = 1.0
                    norm = 1.0
                direction = pair_vec / norm
                if a in route_targets:
                    route_vec = np.asarray(route_targets[a], dtype=np.float64)
                    if np.linalg.norm(route_vec) > 1.0e-9:
                        direction = direction + 0.35 * route_vec / max(np.linalg.norm(route_vec), 1.0e-9)
                if b in route_targets:
                    route_vec = np.asarray(route_targets[b], dtype=np.float64)
                    if np.linalg.norm(route_vec) > 1.0e-9:
                        direction = direction - 0.35 * route_vec / max(np.linalg.norm(route_vec), 1.0e-9)
                direction = direction / max(np.linalg.norm(direction), 1.0e-9)
                for step in (0.18 * max_disp, 0.30 * max_disp, 0.42 * max_disp):
                    trial = current.clone()
                    trial_soft = soft.copy()
                    pa = trial_soft[a] + direction * step
                    pb = trial_soft[b] - direction * step
                    pa[0] = np.clip(pa[0], widths[a] * 0.5, benchmark.canvas_width - widths[a] * 0.5)
                    pa[1] = np.clip(pa[1], heights[a] * 0.5, benchmark.canvas_height - heights[a] * 0.5)
                    pb[0] = np.clip(pb[0], widths[b] * 0.5, benchmark.canvas_width - widths[b] * 0.5)
                    pb[1] = np.clip(pb[1], heights[b] * 0.5, benchmark.canvas_height - heights[b] * 0.5)
                    trial_soft[a] = pa
                    trial_soft[b] = pb
                    trial[benchmark.num_hard_macros :] = torch.tensor(trial_soft, dtype=trial.dtype)
                    costs = self._exact_costs(trial, benchmark, plc)
                    if costs["proxy_cost"] + 1.0e-4 < best_trial_costs["proxy_cost"]:
                        best_trial = trial
                        best_trial_costs = costs

        if best_trial is not None:
            print(
                f"[partcl:soft-spread] {benchmark.name} round=1 "
                f"wl={best_trial_costs['wirelength_cost']:.4f} "
                f"den={best_trial_costs['density_cost']:.4f} "
                f"cong={best_trial_costs['congestion_cost']:.4f} "
                f"proxy={best_trial_costs['proxy_cost']:.4f}"
            )
            return best_trial
        return current

    def _outline_incre_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None:
            return placement

        current = self._outline_soft_local_refine(placement, benchmark, graph, plc)
        current_costs = self._exact_costs(current, benchmark, plc)
        num_hard = benchmark.num_hard_macros
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        hard_i = graph["hard_i"]
        hard_j = graph["hard_j"]
        hard_w = graph["hard_w"]
        base_pos = placement[:num_hard].cpu().numpy().astype(np.float64)
        canvas_scale = max(benchmark.canvas_width, benchmark.canvas_height)
        max_disp = 0.006 * canvas_scale

        for _ in range(2):
            hot = self._get_hotspot_targets(
                placement=current,
                benchmark=benchmark,
                plc=plc,
                widths=widths,
                heights=heights,
                movable=movable,
                limit=4 if benchmark.num_macros <= 320 else 6,
            )
            if not hot:
                break

            pos = current[:num_hard].cpu().numpy().astype(np.float64)
            bary = pos.copy()
            if len(hard_w) > 0:
                nbr_sum = np.zeros_like(pos)
                nbr_w = np.zeros(num_hard, dtype=np.float64)
                np.add.at(nbr_sum, hard_i, pos[hard_j] * hard_w[:, None])
                np.add.at(nbr_sum, hard_j, pos[hard_i] * hard_w[:, None])
                np.add.at(nbr_w, hard_i, hard_w)
                np.add.at(nbr_w, hard_j, hard_w)
                mask = nbr_w > 0
                bary[mask] = nbr_sum[mask] / nbr_w[mask, None]

            best_candidate = None
            best_costs = current_costs
            best_disp = 0.0
            proposals = []
            for idx, hotspot_vec in hot:
                if not movable[idx]:
                    continue
                directions = []
                for vec in (
                    hotspot_vec,
                    hotspot_vec + 0.10 * (pos[idx] - bary[idx]),
                    hotspot_vec + 0.06 * (pos[idx] - base_pos[idx]),
                ):
                    norm = np.linalg.norm(vec)
                    if norm >= 1.0e-9:
                        directions.append(vec / norm)

                for direction in directions:
                    for step_frac in (0.001, 0.002, 0.003):
                        step = step_frac * canvas_scale
                        proposed = pos[idx] + direction * step
                        if np.linalg.norm(proposed - base_pos[idx]) > max_disp:
                            continue
                        if not self._preserves_local_order(
                            idx=idx,
                            old_pos=pos,
                            new_center=proposed,
                            widths=widths,
                            heights=heights,
                            movable=movable,
                        ):
                            continue
                        pre_score = self._score_incre_proposal(
                            idx=idx,
                            old_pos=pos,
                            new_center=proposed,
                            base_pos=base_pos,
                            bary=bary,
                            hotspot_vec=hotspot_vec,
                            widths=widths,
                            heights=heights,
                            movable=movable,
                            canvas_scale=canvas_scale,
                        )
                        proposals.append((pre_score, idx, proposed))

            for _, idx, proposed in sorted(proposals, key=lambda item: item[0], reverse=True)[:8]:
                trial = current.clone()
                trial_pos = trial[:num_hard].cpu().numpy().astype(np.float64)
                trial_pos[idx] = proposed
                trial_pos = self._legalize_hard_numpy(trial_pos, benchmark)
                if np.linalg.norm(trial_pos[idx] - base_pos[idx]) > max_disp + 1.0e-9:
                    continue
                if not self._preserves_local_order(
                    idx=idx,
                    old_pos=pos,
                    new_center=trial_pos[idx],
                    widths=widths,
                    heights=heights,
                    movable=movable,
                ):
                    continue
                trial[:num_hard] = torch.tensor(trial_pos, dtype=trial.dtype)
                trial = self._outline_soft_local_refine(trial, benchmark, graph, plc)
                costs = self._exact_costs(trial, benchmark, plc)
                disp = float(np.linalg.norm(trial_pos[idx] - base_pos[idx]))
                improved = costs["proxy_cost"] + 1.0e-4 < best_costs["proxy_cost"]
                tied = abs(costs["proxy_cost"] - best_costs["proxy_cost"]) <= 1.0e-4 and disp < best_disp
                if improved or (best_candidate is not None and tied):
                    best_candidate = trial
                    best_costs = costs
                    best_disp = disp

            if best_candidate is None:
                break

            current = best_candidate
            current_costs = best_costs
            print(
                f"[partcl:outline-incre] {benchmark.name} "
                f"wl={current_costs['wirelength_cost']:.4f} "
                f"den={current_costs['density_cost']:.4f} "
                f"cong={current_costs['congestion_cost']:.4f} "
                f"proxy={current_costs['proxy_cost']:.4f}"
            )

        return current

    def _outline_replite_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None:
            return placement

        current = self._outline_soft_local_refine(placement.clone(), benchmark, graph, plc)
        current_costs = self._exact_costs(current, benchmark, plc)
        print(
            f"[partcl:replite] {benchmark.name} start "
            f"wl={current_costs['wirelength_cost']:.4f} "
            f"den={current_costs['density_cost']:.4f} "
            f"cong={current_costs['congestion_cost']:.4f} "
            f"proxy={current_costs['proxy_cost']:.4f}"
        )
        num_hard = benchmark.num_hard_macros
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        base_pos = placement[:num_hard].cpu().numpy().astype(np.float64)
        smooth_pos = graph["smooth_pos"]
        hard_i = graph["hard_i"]
        hard_j = graph["hard_j"]
        hard_w = graph["hard_w"]
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        max_disp = 0.022 * canvas_scale if benchmark.num_macros <= 320 else 0.016 * canvas_scale

        for round_idx in range(4 if benchmark.num_macros <= 320 else 3):
            hot = self._get_hotspot_targets(
                placement=current,
                benchmark=benchmark,
                plc=plc,
                widths=widths,
                heights=heights,
                movable=movable,
                limit=14 if benchmark.num_macros <= 320 else 10,
            )
            if not hot:
                break

            pos = current[:num_hard].cpu().numpy().astype(np.float64)
            bary = pos.copy()
            if len(hard_w) > 0:
                nbr_sum = np.zeros_like(pos)
                nbr_w = np.zeros(num_hard, dtype=np.float64)
                np.add.at(nbr_sum, hard_i, pos[hard_j] * hard_w[:, None])
                np.add.at(nbr_sum, hard_j, pos[hard_i] * hard_w[:, None])
                np.add.at(nbr_w, hard_i, hard_w)
                np.add.at(nbr_w, hard_j, hard_w)
                mask = nbr_w > 0
                bary[mask] = nbr_sum[mask] / nbr_w[mask, None]

            best_round = None
            best_round_costs = current_costs
            proposals = []
            for idx, hotspot_vec in hot:
                if not movable[idx]:
                    continue
                directions = []
                for vec in (
                    1.45 * hotspot_vec,
                    1.25 * hotspot_vec + 0.18 * (bary[idx] - pos[idx]),
                    1.10 * hotspot_vec + 0.16 * (smooth_pos[idx] - pos[idx]),
                    1.00 * hotspot_vec + 0.14 * (base_pos[idx] - pos[idx]),
                ):
                    norm = np.linalg.norm(vec)
                    if norm >= 1.0e-9:
                        directions.append(vec / norm)

                for direction in directions:
                    for step_frac in (0.0020, 0.0035, 0.0050, 0.0065):
                        proposed = pos[idx] + direction * (step_frac * canvas_scale)
                        disp = proposed - base_pos[idx]
                        disp_norm = np.linalg.norm(disp)
                        if disp_norm > max_disp:
                            proposed = base_pos[idx] + disp * (max_disp / max(disp_norm, 1.0e-9))
                        if not self._preserves_local_order(
                            idx=idx,
                            old_pos=pos,
                            new_center=proposed,
                            widths=widths,
                            heights=heights,
                            movable=movable,
                        ):
                            continue
                        pre_score = self._score_incre_proposal(
                            idx=idx,
                            old_pos=pos,
                            new_center=proposed,
                            base_pos=base_pos,
                            bary=bary,
                            hotspot_vec=hotspot_vec,
                            widths=widths,
                            heights=heights,
                            movable=movable,
                            canvas_scale=canvas_scale,
                        )
                        proposals.append((pre_score, idx, proposed))

            for _, idx, proposed in sorted(proposals, key=lambda item: item[0], reverse=True)[:12]:
                trial = current.clone()
                trial_pos = trial[:num_hard].cpu().numpy().astype(np.float64)
                trial_pos[idx] = proposed
                trial_pos = self._legalize_hard_numpy(trial_pos, benchmark)
                if np.linalg.norm(trial_pos[idx] - base_pos[idx]) > max_disp + 1.0e-9:
                    continue
                if not self._preserves_local_order(
                    idx=idx,
                    old_pos=pos,
                    new_center=trial_pos[idx],
                    widths=widths,
                    heights=heights,
                    movable=movable,
                ):
                    continue
                trial[:num_hard] = torch.tensor(trial_pos, dtype=trial.dtype)
                trial = self._outline_soft_local_refine(trial, benchmark, graph, plc)
                costs = self._exact_costs(trial, benchmark, plc)
                if costs["proxy_cost"] + 1.0e-4 < best_round_costs["proxy_cost"]:
                    best_round = trial
                    best_round_costs = costs

            if best_round is None:
                print(f"[partcl:replite] {benchmark.name} round={round_idx + 1} accepted=0")
                break

            current = best_round
            current_costs = best_round_costs
            print(
                f"[partcl:replite] {benchmark.name} round={round_idx + 1} "
                f"accepted=1 "
                f"wl={current_costs['wirelength_cost']:.4f} "
                f"den={current_costs['density_cost']:.4f} "
                f"cong={current_costs['congestion_cost']:.4f} "
                f"proxy={current_costs['proxy_cost']:.4f}"
            )

        print(
            f"[partcl:replite] {benchmark.name} pre_polish "
            f"wl={current_costs['wirelength_cost']:.4f} "
            f"den={current_costs['density_cost']:.4f} "
            f"cong={current_costs['congestion_cost']:.4f} "
            f"proxy={current_costs['proxy_cost']:.4f}"
        )
        polished = self._macrobase_soft_search(current.clone(), benchmark, graph, plc)
        polished_costs = self._exact_costs(polished, benchmark, plc)
        if polished_costs["proxy_cost"] + 1.0e-4 < current_costs["proxy_cost"]:
            print(
                f"[partcl:replite] {benchmark.name} polish "
                f"wl={polished_costs['wirelength_cost']:.4f} "
                f"den={polished_costs['density_cost']:.4f} "
                f"cong={polished_costs['congestion_cost']:.4f} "
                f"proxy={polished_costs['proxy_cost']:.4f}"
            )
            return polished

        print(
            f"[partcl:replite] {benchmark.name} polish accepted=0 "
            f"wl={polished_costs['wirelength_cost']:.4f} "
            f"den={polished_costs['density_cost']:.4f} "
            f"cong={polished_costs['congestion_cost']:.4f} "
            f"proxy={polished_costs['proxy_cost']:.4f}"
        )

        return current

    def _preserves_local_order(
        self,
        idx: int,
        old_pos: np.ndarray,
        new_center: np.ndarray,
        widths: np.ndarray,
        heights: np.ndarray,
        movable: np.ndarray,
    ) -> bool:
        ref = old_pos[idx]
        for other in range(old_pos.shape[0]):
            if other == idx or not movable[other]:
                continue
            dx = old_pos[other, 0] - ref[0]
            dy = old_pos[other, 1] - ref[1]
            if abs(dy) <= 0.6 * (heights[idx] + heights[other]):
                old_sign = np.sign(dx)
                new_sign = np.sign(old_pos[other, 0] - new_center[0])
                if old_sign != 0.0 and new_sign != 0.0 and old_sign != new_sign:
                    return False
            if abs(dx) <= 0.6 * (widths[idx] + widths[other]):
                old_sign = np.sign(dy)
                new_sign = np.sign(old_pos[other, 1] - new_center[1])
                if old_sign != 0.0 and new_sign != 0.0 and old_sign != new_sign:
                    return False
        return True

    def _score_incre_proposal(
        self,
        idx: int,
        old_pos: np.ndarray,
        new_center: np.ndarray,
        base_pos: np.ndarray,
        bary: np.ndarray,
        hotspot_vec: np.ndarray,
        widths: np.ndarray,
        heights: np.ndarray,
        movable: np.ndarray,
        canvas_scale: float,
    ) -> float:
        move = new_center - old_pos[idx]
        move_norm = float(np.linalg.norm(move))
        if move_norm < 1.0e-12:
            return -1.0e9

        hot_norm = float(np.linalg.norm(hotspot_vec))
        congestion_relief = float(np.dot(move, hotspot_vec) / (move_norm * max(hot_norm, 1.0e-9)))

        crowd_penalty = 0.0
        for other in range(old_pos.shape[0]):
            if other == idx or not movable[other]:
                continue
            dx = abs(old_pos[other, 0] - new_center[0])
            dy = abs(old_pos[other, 1] - new_center[1])
            sep_x = 0.5 * (widths[idx] + widths[other]) + 0.02
            sep_y = 0.5 * (heights[idx] + heights[other]) + 0.02
            if dx < 1.35 * sep_x and dy < 1.35 * sep_y:
                crowd_penalty += max(0.0, 1.35 * sep_x - dx) + max(0.0, 1.35 * sep_y - dy)

        base_penalty = float(np.linalg.norm(new_center - base_pos[idx]) / max(canvas_scale, 1.0e-9))
        bary_penalty = float(np.linalg.norm(new_center - bary[idx]) / max(canvas_scale, 1.0e-9))
        return 2.4 * congestion_relief - 3.2 * crowd_penalty - 0.9 * base_penalty - 0.35 * bary_penalty

    def _compute_rudy_map(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> np.ndarray:
        """Per-net pin-bbox RUDY demand on the contest grid.

        Each net contributes weight / max(bbox_w, eps) horizontally and
        weight / max(bbox_h, eps) vertically, smeared uniformly over the
        bounding box's grid cells. Returns the L2-combined demand.
        """
        nrow = max(int(benchmark.grid_rows), 1)
        ncol = max(int(benchmark.grid_cols), 1)
        cell_w = float(benchmark.canvas_width) / ncol
        cell_h = float(benchmark.canvas_height) / nrow
        pos = placement.cpu().numpy().astype(np.float64)
        port_pos = benchmark.port_positions.cpu().numpy().astype(np.float64)
        port_start = benchmark.num_macros
        h_demand = np.zeros((nrow, ncol), dtype=np.float64)
        v_demand = np.zeros((nrow, ncol), dtype=np.float64)
        for net_idx, nodes in enumerate(benchmark.net_nodes):
            if len(nodes) < 2:
                continue
            xs = []
            ys = []
            for node in nodes.tolist():
                if node < benchmark.num_macros:
                    xs.append(pos[node, 0])
                    ys.append(pos[node, 1])
                else:
                    pidx = node - port_start
                    if 0 <= pidx < port_pos.shape[0]:
                        xs.append(port_pos[pidx, 0])
                        ys.append(port_pos[pidx, 1])
            if len(xs) < 2:
                continue
            xmin = max(0.0, min(xs))
            xmax = max(xmin, max(xs))
            ymin = max(0.0, min(ys))
            ymax = max(ymin, max(ys))
            c0 = min(ncol - 1, max(0, int(xmin / max(cell_w, 1.0e-9))))
            c1 = min(ncol - 1, max(0, int(xmax / max(cell_w, 1.0e-9))))
            r0 = min(nrow - 1, max(0, int(ymin / max(cell_h, 1.0e-9))))
            r1 = min(nrow - 1, max(0, int(ymax / max(cell_h, 1.0e-9))))
            weight = (
                float(benchmark.net_weights[net_idx].item())
                if net_idx < len(benchmark.net_weights)
                else 1.0
            )
            bbox_w = max(xmax - xmin, cell_w * 0.25)
            bbox_h = max(ymax - ymin, cell_h * 0.25)
            h_per_cell = weight / bbox_w
            v_per_cell = weight / bbox_h
            h_demand[r0 : r1 + 1, c0 : c1 + 1] += h_per_cell
            v_demand[r0 : r1 + 1, c0 : c1 + 1] += v_per_cell
        return np.sqrt(h_demand * h_demand + v_demand * v_demand)

    def _outline_rudy_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        """Push hard and soft macros away from top-5% RUDY hotspots.

        Accept-only-if-better at the exact proxy. Combines a small hard-macro
        shift with a soft Laplacian re-solve.
        """
        if plc is None:
            return placement

        rudy = self._compute_rudy_map(placement, benchmark)
        if not np.isfinite(rudy).all() or float(rudy.max()) < 1.0e-9:
            return placement

        nrow, ncol = rudy.shape
        threshold = float(np.percentile(rudy, 95.0))
        hot_mask = rudy >= threshold
        hot_count = int(np.count_nonzero(hot_mask))
        if hot_count == 0:
            return placement

        # Smooth gradient field: descent direction = -∇RUDY (repel from hotspots).
        # Use central differences with a small Gaussian smoothing to avoid bin noise.
        smoothed = rudy.copy()
        if min(nrow, ncol) >= 4:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
            smoothed = np.apply_along_axis(
                lambda a: np.convolve(a, kernel, mode="same"), 0, smoothed
            )
            smoothed = np.apply_along_axis(
                lambda a: np.convolve(a, kernel, mode="same"), 1, smoothed
            )
        gy = np.zeros_like(smoothed)
        gx = np.zeros_like(smoothed)
        gy[1:-1, :] = 0.5 * (smoothed[2:, :] - smoothed[:-2, :])
        gx[:, 1:-1] = 0.5 * (smoothed[:, 2:] - smoothed[:, :-2])
        rudy_max = float(smoothed.max() + 1.0e-9)

        cell_w = float(benchmark.canvas_width) / ncol
        cell_h = float(benchmark.canvas_height) / nrow
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))

        num_hard = int(benchmark.num_hard_macros)
        num_soft = int(benchmark.num_soft_macros)
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        movable_hard = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        max_hard_disp = 0.012 * canvas_scale
        max_soft_disp_frac = 0.010

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        baseline_proxy = float(current_costs["proxy_cost"])
        baseline_dc = float(current_costs["density_cost"] + current_costs["congestion_cost"])

        # Score hard macros by RUDY exposure within their bbox, push the worst K.
        hard_exposure = np.zeros(num_hard, dtype=np.float64)
        push_dir = np.zeros((num_hard, 2), dtype=np.float64)
        pos_arr = current[:num_hard].cpu().numpy().astype(np.float64)
        for idx in range(num_hard):
            if not movable_hard[idx]:
                continue
            x = pos_arr[idx, 0]
            y = pos_arr[idx, 1]
            hw = widths[idx] * 0.5
            hh = heights[idx] * 0.5
            c0 = min(ncol - 1, max(0, int((x - hw) / max(cell_w, 1.0e-9))))
            c1 = min(ncol - 1, max(0, int((x + hw) / max(cell_w, 1.0e-9))))
            r0 = min(nrow - 1, max(0, int((y - hh) / max(cell_h, 1.0e-9))))
            r1 = min(nrow - 1, max(0, int((y + hh) / max(cell_h, 1.0e-9))))
            patch = smoothed[r0 : r1 + 1, c0 : c1 + 1]
            patch_hot = hot_mask[r0 : r1 + 1, c0 : c1 + 1]
            hard_exposure[idx] = float(patch.mean()) * (1.0 + 0.5 * float(patch_hot.mean()))
            gx_local = float(np.mean(gx[r0 : r1 + 1, c0 : c1 + 1]))
            gy_local = float(np.mean(gy[r0 : r1 + 1, c0 : c1 + 1]))
            norm = max(np.hypot(gx_local, gy_local), 1.0e-9)
            push_dir[idx, 0] = -gx_local / norm
            push_dir[idx, 1] = -gy_local / norm

        top_k = 12 if num_hard <= 320 else 8 if num_hard <= 430 else 6
        order = np.argsort(-hard_exposure)
        active = [int(i) for i in order if hard_exposure[i] > 0.0][:top_k]
        if not active:
            return current

        # Try a step ladder. The first improvement wins.
        for step_frac in (0.012, 0.008, 0.005):
            trial_hard = pos_arr.copy()
            step = step_frac * canvas_scale
            for idx in active:
                trial_hard[idx, 0] += push_dir[idx, 0] * step
                trial_hard[idx, 1] += push_dir[idx, 1] * step
            trial_hard[:, 0] = np.clip(
                trial_hard[:, 0], widths * 0.5, benchmark.canvas_width - widths * 0.5
            )
            trial_hard[:, 1] = np.clip(
                trial_hard[:, 1], heights * 0.5, benchmark.canvas_height - heights * 0.5
            )
            trial_hard = self._legalize_hard_numpy(trial_hard, benchmark)
            disp = trial_hard - pos_arr
            if float(np.linalg.norm(disp, axis=1).max()) > max_hard_disp * 1.5:
                continue

            trial = current.clone()
            trial[:num_hard] = torch.tensor(trial_hard, dtype=trial.dtype)

            if num_soft > 0:
                anchor_pos = current[num_hard:].cpu().numpy().astype(np.float64)
                trial = self._quadratic_soft_macro_follow(
                    placement=trial,
                    benchmark=benchmark,
                    graph=graph,
                    anchor_weight=0.050,
                    relax=0.74,
                    solver_steps=14,
                    anchor_pos=anchor_pos,
                )
                # Soft RUDY push: nudge soft macros sitting in hot bins toward
                # the smoothed gradient direction, capped tightly.
                soft = trial[num_hard:].cpu().numpy().astype(np.float64)
                soft_widths = (
                    benchmark.macro_sizes[num_hard:, 0].cpu().numpy().astype(np.float64)
                )
                soft_heights = (
                    benchmark.macro_sizes[num_hard:, 1].cpu().numpy().astype(np.float64)
                )
                fixed_soft = benchmark.macro_fixed[num_hard:].cpu().numpy()
                max_soft = max_soft_disp_frac * canvas_scale
                for s_idx in range(num_soft):
                    if fixed_soft[s_idx]:
                        continue
                    sx = soft[s_idx, 0]
                    sy = soft[s_idx, 1]
                    c = min(ncol - 1, max(0, int(sx / max(cell_w, 1.0e-9))))
                    r = min(nrow - 1, max(0, int(sy / max(cell_h, 1.0e-9))))
                    if not hot_mask[r, c]:
                        continue
                    dx = -float(gx[r, c])
                    dy = -float(gy[r, c])
                    norm = max(np.hypot(dx, dy), 1.0e-9)
                    dx /= norm
                    dy /= norm
                    soft[s_idx, 0] += dx * 0.35 * max_soft
                    soft[s_idx, 1] += dy * 0.35 * max_soft
                soft[:, 0] = np.clip(
                    soft[:, 0],
                    soft_widths * 0.5,
                    benchmark.canvas_width - soft_widths * 0.5,
                )
                soft[:, 1] = np.clip(
                    soft[:, 1],
                    soft_heights * 0.5,
                    benchmark.canvas_height - soft_heights * 0.5,
                )
                trial[num_hard:] = torch.tensor(soft, dtype=trial.dtype)

            trial_costs = self._exact_costs(trial, benchmark, plc)
            trial_proxy = float(trial_costs["proxy_cost"])
            trial_dc = float(trial_costs["density_cost"] + trial_costs["congestion_cost"])
            if int(trial_costs.get("overlap_count", 0)) != 0:
                continue
            proxy_gain = baseline_proxy - trial_proxy
            dc_gain = baseline_dc - trial_dc
            accept = (
                proxy_gain >= 1.0e-4
                or (dc_gain >= 0.02 and trial_proxy <= baseline_proxy + 0.005)
            )
            if accept:
                if os.environ.get("PARTCL_DEBUG_RUDY", "0") == "1":
                    print(
                        f"[partcl:rudy] {benchmark.name} "
                        f"step={step_frac:.3f} hot={hot_count} active={len(active)} "
                        f"proxy {baseline_proxy:.4f}->{trial_proxy:.4f} "
                        f"dc {baseline_dc:.4f}->{trial_dc:.4f}"
                    )
                return trial

        return current

    def _outline_route_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None:
            return placement

        current = self._outline_soft_refine(placement, benchmark, graph)
        current_costs = self._exact_costs(current, benchmark, plc)
        num_hard = benchmark.num_hard_macros
        widths = benchmark.macro_sizes[:num_hard, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[:num_hard, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[:num_hard]).cpu().numpy()
        hard_i = graph["hard_i"]
        hard_j = graph["hard_j"]
        hard_w = graph["hard_w"]
        base_pos = placement[:num_hard].cpu().numpy().astype(np.float64)
        canvas_scale = max(benchmark.canvas_width, benchmark.canvas_height)

        for _ in range(2):
            hot = self._get_route_hotspot_targets(
                placement=current,
                benchmark=benchmark,
                plc=plc,
                widths=widths,
                heights=heights,
                movable=movable,
                limit=5,
            )
            if not hot:
                break

            pos = current[:num_hard].cpu().numpy().astype(np.float64)
            bary = pos.copy()
            if len(hard_w) > 0:
                nbr_sum = np.zeros_like(pos)
                nbr_w = np.zeros(num_hard, dtype=np.float64)
                np.add.at(nbr_sum, hard_i, pos[hard_j] * hard_w[:, None])
                np.add.at(nbr_sum, hard_j, pos[hard_i] * hard_w[:, None])
                np.add.at(nbr_w, hard_i, hard_w)
                np.add.at(nbr_w, hard_j, hard_w)
                mask = nbr_w > 0
                bary[mask] = nbr_sum[mask] / nbr_w[mask, None]

            best_candidate = None
            best_costs = current_costs
            for idx, hotspot_vec in hot:
                directions = []
                for vec in (
                    hotspot_vec,
                    hotspot_vec + 0.12 * (pos[idx] - bary[idx]),
                    hotspot_vec + 0.08 * (pos[idx] - base_pos[idx]),
                ):
                    norm = np.linalg.norm(vec)
                    if norm >= 1.0e-9:
                        directions.append(vec / norm)

                for direction in directions:
                    for step_frac in (0.0015, 0.003, 0.0045):
                        trial = current.clone()
                        trial_pos = trial[:num_hard].cpu().numpy().astype(np.float64)
                        trial_pos[idx] += direction * (step_frac * canvas_scale)
                        trial_pos = self._legalize_hard_numpy(trial_pos, benchmark)
                        trial[:num_hard] = torch.tensor(trial_pos, dtype=trial.dtype)
                        trial = self._outline_soft_refine(trial, benchmark, graph)
                        costs = self._exact_costs(trial, benchmark, plc)
                        if costs["proxy_cost"] + 1.0e-4 < best_costs["proxy_cost"]:
                            best_candidate = trial
                            best_costs = costs

            if best_candidate is None:
                break

            current = best_candidate
            current_costs = best_costs
            print(
                f"[partcl:outline-route] {benchmark.name} "
                f"wl={current_costs['wirelength_cost']:.4f} "
                f"den={current_costs['density_cost']:.4f} "
                f"cong={current_costs['congestion_cost']:.4f} "
                f"proxy={current_costs['proxy_cost']:.4f}"
            )

        return current

    def _exact_soft_hotspot_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        graph: dict,
        plc,
    ) -> torch.Tensor:
        if plc is None or benchmark.num_soft_macros == 0:
            return placement

        current = placement.clone()
        current_costs = self._exact_costs(current, benchmark, plc)
        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        soft_sizes = benchmark.macro_sizes[num_hard:, :].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[num_hard:]).cpu().numpy()
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height))

        for round_idx in range(3):
            route_targets = self._get_soft_route_targets(
                current,
                benchmark,
                plc,
                limit=10 if benchmark.num_macros <= 320 else 7,
            )
            region_targets = self._get_soft_region_targets(
                current,
                benchmark,
                plc,
                limit=10 if benchmark.num_macros <= 320 else 7,
                region_rows=5 if benchmark.grid_rows >= 24 else 3,
                region_cols=5 if benchmark.grid_cols >= 24 else 3,
            )
            vectors: dict[int, np.ndarray] = {}
            for idx, vec in route_targets:
                vectors[idx] = vectors.get(idx, np.zeros(2, dtype=np.float64)) + 0.9 * vec
            for idx, vec in region_targets:
                vectors[idx] = vectors.get(idx, np.zeros(2, dtype=np.float64)) + 1.1 * vec

            ranked = sorted(
                (
                    (float(np.linalg.norm(vec)), idx, vec)
                    for idx, vec in vectors.items()
                    if idx < num_soft and movable[idx] and np.linalg.norm(vec) > 1.0e-9
                ),
                reverse=True,
            )[:8]
            if not ranked:
                break

            base_soft = current[num_hard:].cpu().numpy().astype(np.float64)
            best_trial = None
            best_costs = current_costs
            for _, soft_idx, vec in ranked:
                direction = vec / max(np.linalg.norm(vec), 1.0e-9)
                for step_frac in (0.0018, 0.0032, 0.0050):
                    trial = current.clone()
                    soft = base_soft.copy()
                    soft[soft_idx] = soft[soft_idx] + direction * (step_frac * canvas_scale)
                    soft[:, 0] = np.clip(
                        soft[:, 0],
                        soft_sizes[:, 0] * 0.5,
                        benchmark.canvas_width - soft_sizes[:, 0] * 0.5,
                    )
                    soft[:, 1] = np.clip(
                        soft[:, 1],
                        soft_sizes[:, 1] * 0.5,
                        benchmark.canvas_height - soft_sizes[:, 1] * 0.5,
                    )
                    trial[num_hard:] = torch.tensor(soft, dtype=trial.dtype)
                    costs = self._exact_costs(trial, benchmark, plc)
                    if costs["proxy_cost"] + 1.0e-4 < best_costs["proxy_cost"]:
                        best_trial = trial
                        best_costs = costs

            if best_trial is None:
                break
            current = best_trial
            current_costs = best_costs
            print(
                f"[partcl:exact-soft] {benchmark.name} round={round_idx + 1} "
                f"wl={current_costs['wirelength_cost']:.4f} "
                f"den={current_costs['density_cost']:.4f} "
                f"cong={current_costs['congestion_cost']:.4f} "
                f"proxy={current_costs['proxy_cost']:.4f}"
            )

        return current

    def _get_soft_hotspot_targets(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        limit: int,
    ) -> list[int]:
        self._exact_costs(placement, benchmark, plc)
        nrow = benchmark.grid_rows
        ncol = benchmark.grid_cols
        density = np.asarray(plc.grid_cells, dtype=np.float64).reshape(nrow, ncol)
        h_cong = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        congestion = np.maximum(h_cong, v_cong)

        cong_thr = float(np.percentile(congestion, 96.0))
        dens_thr = float(np.percentile(density, 88.0))
        hotspot_scores = 1.9 * np.maximum(0.0, congestion - cong_thr) + 1.2 * np.maximum(
            0.0, density - dens_thr
        )
        flat = np.argsort(hotspot_scores.reshape(-1))[::-1]
        if hotspot_scores.reshape(-1)[flat[0]] <= 0.0:
            return []

        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        soft = placement[num_hard:].cpu().numpy().astype(np.float64)
        widths = benchmark.macro_sizes[num_hard:, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[num_hard:, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[num_hard:]).cpu().numpy()
        grid_w = benchmark.canvas_width / max(ncol, 1)
        grid_h = benchmark.canvas_height / max(nrow, 1)

        scores: dict[int, float] = {}
        for flat_idx in flat[: max(12, 3 * limit)]:
            score = hotspot_scores.reshape(-1)[flat_idx]
            if score <= 0.0:
                break
            row = int(flat_idx // ncol)
            col = int(flat_idx % ncol)
            cx = (col + 0.5) * grid_w
            cy = (row + 0.5) * grid_h
            for soft_idx in range(num_soft):
                if not movable[soft_idx]:
                    continue
                dx = abs(soft[soft_idx, 0] - cx)
                dy = abs(soft[soft_idx, 1] - cy)
                if dx > 0.55 * widths[soft_idx] + grid_w or dy > 0.55 * heights[soft_idx] + grid_h:
                    continue
                scores[soft_idx] = scores.get(soft_idx, 0.0) + score

        return sorted(scores, key=lambda idx: scores[idx], reverse=True)[:limit]

    def _get_soft_route_targets(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        limit: int,
    ) -> list[tuple[int, np.ndarray]]:
        self._exact_costs(placement, benchmark, plc)
        nrow = benchmark.grid_rows
        ncol = benchmark.grid_cols
        h_cong = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        congestion = np.maximum(h_cong, v_cong)
        density = np.asarray(plc.grid_cells, dtype=np.float64).reshape(nrow, ncol)

        cong_thr = float(np.percentile(congestion, 97.5))
        dens_thr = float(np.percentile(density, 90.0))
        hotspot_scores = 2.8 * np.maximum(0.0, congestion - cong_thr) - 0.20 * np.maximum(0.0, density - dens_thr)
        flat = np.argsort(hotspot_scores.reshape(-1))[::-1]
        if hotspot_scores.reshape(-1)[flat[0]] <= 0.0:
            return []

        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        soft = placement[num_hard:].cpu().numpy().astype(np.float64)
        widths = benchmark.macro_sizes[num_hard:, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[num_hard:, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[num_hard:]).cpu().numpy()
        grid_w = benchmark.canvas_width / max(ncol, 1)
        grid_h = benchmark.canvas_height / max(nrow, 1)

        vectors: dict[int, np.ndarray] = {}
        scores: dict[int, float] = {}
        for flat_idx in flat[: max(16, 3 * limit)]:
            score = hotspot_scores.reshape(-1)[flat_idx]
            if score <= 0.0:
                break
            row = int(flat_idx // ncol)
            col = int(flat_idx % ncol)
            cx = (col + 0.5) * grid_w
            cy = (row + 0.5) * grid_h
            center = np.array([cx, cy], dtype=np.float64)
            for soft_idx in range(num_soft):
                if not movable[soft_idx]:
                    continue
                dx = abs(soft[soft_idx, 0] - cx)
                dy = abs(soft[soft_idx, 1] - cy)
                if dx > 0.60 * widths[soft_idx] + 1.5 * grid_w or dy > 0.60 * heights[soft_idx] + 1.5 * grid_h:
                    continue
                away = soft[soft_idx] - center
                norm = np.linalg.norm(away)
                if norm < 1.0e-9:
                    continue
                vectors[soft_idx] = vectors.get(soft_idx, np.zeros(2, dtype=np.float64)) + score * away / norm
                scores[soft_idx] = scores.get(soft_idx, 0.0) + score

        ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
        return [(idx, vectors[idx]) for idx in ordered]

    def _get_soft_channel_targets(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        limit: int,
    ) -> list[tuple[int, np.ndarray]]:
        self._exact_costs(placement, benchmark, plc)
        nrow = benchmark.grid_rows
        ncol = benchmark.grid_cols
        density = np.asarray(plc.grid_cells, dtype=np.float64).reshape(nrow, ncol)
        h_cong = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nrow, ncol)

        row_route = np.maximum(np.percentile(h_cong, 90.0, axis=1), h_cong.mean(axis=1))
        col_route = np.maximum(np.percentile(v_cong, 90.0, axis=0), v_cong.mean(axis=0))
        row_density = density.mean(axis=1)
        col_density = density.mean(axis=0)
        row_thr = float(np.percentile(row_route, 88.0))
        col_thr = float(np.percentile(col_route, 88.0))
        row_d_thr = float(np.percentile(row_density, 80.0))
        col_d_thr = float(np.percentile(col_density, 80.0))
        row_scores = 2.4 * np.maximum(0.0, row_route - row_thr) - 0.15 * np.maximum(0.0, row_density - row_d_thr)
        col_scores = 2.4 * np.maximum(0.0, col_route - col_thr) - 0.15 * np.maximum(0.0, col_density - col_d_thr)

        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        soft = placement[num_hard:].cpu().numpy().astype(np.float64)
        widths = benchmark.macro_sizes[num_hard:, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[num_hard:, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[num_hard:]).cpu().numpy()
        grid_w = benchmark.canvas_width / max(ncol, 1)
        grid_h = benchmark.canvas_height / max(nrow, 1)

        vectors: dict[int, np.ndarray] = {}
        scores: dict[int, float] = {}
        top_rows = np.argsort(row_scores)[::-1][: max(4, limit // 3)]
        top_cols = np.argsort(col_scores)[::-1][: max(4, limit // 3)]

        for row in top_rows:
            score = float(row_scores[row])
            if score <= 0.0:
                break
            cy = (float(row) + 0.5) * grid_h
            stripe_half = 1.6 * grid_h
            for soft_idx in range(num_soft):
                if not movable[soft_idx]:
                    continue
                if abs(soft[soft_idx, 1] - cy) > 0.55 * heights[soft_idx] + stripe_half:
                    continue
                away_y = soft[soft_idx, 1] - cy
                direction = 1.0 if away_y >= 0.0 else -1.0
                vec = np.array([0.0, direction], dtype=np.float64)
                vectors[soft_idx] = vectors.get(soft_idx, np.zeros(2, dtype=np.float64)) + score * vec
                scores[soft_idx] = scores.get(soft_idx, 0.0) + score

        for col in top_cols:
            score = float(col_scores[col])
            if score <= 0.0:
                break
            cx = (float(col) + 0.5) * grid_w
            stripe_half = 1.6 * grid_w
            for soft_idx in range(num_soft):
                if not movable[soft_idx]:
                    continue
                if abs(soft[soft_idx, 0] - cx) > 0.55 * widths[soft_idx] + stripe_half:
                    continue
                away_x = soft[soft_idx, 0] - cx
                direction = 1.0 if away_x >= 0.0 else -1.0
                vec = np.array([direction, 0.0], dtype=np.float64)
                vectors[soft_idx] = vectors.get(soft_idx, np.zeros(2, dtype=np.float64)) + score * vec
                scores[soft_idx] = scores.get(soft_idx, 0.0) + score

        ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
        return [(idx, vectors[idx]) for idx in ordered if np.linalg.norm(vectors[idx]) > 1.0e-9]

    def _get_soft_region_targets(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        limit: int,
        region_rows: int,
        region_cols: int,
    ) -> list[tuple[int, np.ndarray]]:
        self._exact_costs(placement, benchmark, plc)
        nrow = benchmark.grid_rows
        ncol = benchmark.grid_cols
        density = np.asarray(plc.grid_cells, dtype=np.float64).reshape(nrow, ncol)
        h_cong = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        congestion = np.maximum(h_cong, v_cong)

        row_edges = np.linspace(0, nrow, region_rows + 1, dtype=int)
        col_edges = np.linspace(0, ncol, region_cols + 1, dtype=int)
        region_list: list[tuple[float, float, float, float, float]] = []
        for r in range(region_rows):
            r0, r1 = row_edges[r], row_edges[r + 1]
            if r1 <= r0:
                continue
            for c in range(region_cols):
                c0, c1 = col_edges[c], col_edges[c + 1]
                if c1 <= c0:
                    continue
                region_den = float(np.mean(density[r0:r1, c0:c1]))
                region_cong = float(np.mean(congestion[r0:r1, c0:c1]))
                center_x = ((c0 + c1) * 0.5 / max(ncol, 1)) * benchmark.canvas_width
                center_y = ((r0 + r1) * 0.5 / max(nrow, 1)) * benchmark.canvas_height
                region_list.append((region_den + 1.35 * region_cong, center_x, center_y, region_den, region_cong))

        if not region_list:
            return []
        scores = np.array([item[0] for item in region_list], dtype=np.float64)
        score_thr = float(np.percentile(scores, 82.0))

        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        soft = placement[num_hard:].cpu().numpy().astype(np.float64)
        widths = benchmark.macro_sizes[num_hard:, 0].cpu().numpy().astype(np.float64)
        heights = benchmark.macro_sizes[num_hard:, 1].cpu().numpy().astype(np.float64)
        movable = (~benchmark.macro_fixed[num_hard:]).cpu().numpy()
        vectors: dict[int, np.ndarray] = {}
        accum_scores: dict[int, float] = {}
        region_half_w = benchmark.canvas_width / max(region_cols * 2.0, 1.0)
        region_half_h = benchmark.canvas_height / max(region_rows * 2.0, 1.0)
        for region_score, cx, cy, _, _ in sorted(region_list, reverse=True):
            if region_score < score_thr:
                break
            center = np.array([cx, cy], dtype=np.float64)
            for soft_idx in range(num_soft):
                if not movable[soft_idx]:
                    continue
                dx = abs(soft[soft_idx, 0] - cx)
                dy = abs(soft[soft_idx, 1] - cy)
                if dx > 0.60 * widths[soft_idx] + region_half_w or dy > 0.60 * heights[soft_idx] + region_half_h:
                    continue
                away = soft[soft_idx] - center
                norm = np.linalg.norm(away)
                if norm < 1.0e-9:
                    continue
                vectors[soft_idx] = vectors.get(soft_idx, np.zeros(2, dtype=np.float64)) + region_score * away / norm
                accum_scores[soft_idx] = accum_scores.get(soft_idx, 0.0) + region_score

        ordered = sorted(accum_scores, key=accum_scores.get, reverse=True)[:limit]
        return [(idx, vectors[idx]) for idx in ordered if np.linalg.norm(vectors[idx]) > 1.0e-9]

    def _get_route_hotspot_targets(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        widths: np.ndarray,
        heights: np.ndarray,
        movable: np.ndarray,
        limit: int,
    ) -> list[tuple[int, np.ndarray]]:
        from macro_place.objective import _set_placement

        _set_placement(plc, placement, benchmark)
        nrow = benchmark.grid_rows
        ncol = benchmark.grid_cols
        density = np.asarray(plc.grid_cells, dtype=np.float64).reshape(nrow, ncol)
        h_cong = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        congestion = np.maximum(h_cong, v_cong)

        cong_thr = float(np.percentile(congestion, 98.0))
        dens_thr = float(np.percentile(density, 90.0))
        hotspot_scores = 2.8 * np.maximum(0.0, congestion - cong_thr) - 0.35 * np.maximum(0.0, density - dens_thr)
        flat = np.argsort(hotspot_scores.reshape(-1))[::-1]
        if hotspot_scores.reshape(-1)[flat[0]] <= 0.0:
            return []

        pos = placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64)
        grid_w = benchmark.canvas_width / max(ncol, 1)
        grid_h = benchmark.canvas_height / max(nrow, 1)
        macro_vectors: dict[int, np.ndarray] = {}
        macro_scores: dict[int, float] = {}
        for flat_idx in flat[: max(8, 2 * limit)]:
            score = hotspot_scores.reshape(-1)[flat_idx]
            if score <= 0.0:
                break
            row = int(flat_idx // ncol)
            col = int(flat_idx % ncol)
            cx = (col + 0.5) * grid_w
            cy = (row + 0.5) * grid_h
            for idx in range(benchmark.num_hard_macros):
                if not movable[idx]:
                    continue
                dx = abs(pos[idx, 0] - cx)
                dy = abs(pos[idx, 1] - cy)
                if dx > 0.55 * widths[idx] + grid_w or dy > 0.55 * heights[idx] + grid_h:
                    continue
                vec = pos[idx] - np.array([cx, cy], dtype=np.float64)
                if idx in macro_vectors:
                    macro_vectors[idx] += score * vec
                    macro_scores[idx] += score
                else:
                    macro_vectors[idx] = score * vec
                    macro_scores[idx] = score

        ranked = sorted(macro_scores, key=lambda idx: macro_scores[idx], reverse=True)[:limit]
        return [(idx, macro_vectors[idx]) for idx in ranked]

    def _get_hotspot_targets(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        widths: np.ndarray,
        heights: np.ndarray,
        movable: np.ndarray,
        limit: int,
    ) -> list[tuple[int, np.ndarray]]:
        from macro_place.objective import _set_placement

        _set_placement(plc, placement, benchmark)
        nrow = benchmark.grid_rows
        ncol = benchmark.grid_cols
        density = np.asarray(plc.grid_cells, dtype=np.float64).reshape(nrow, ncol)
        h_cong = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nrow, ncol)
        congestion = np.maximum(h_cong, v_cong)

        cong_thr = float(np.percentile(congestion, 97.0))
        dens_thr = float(np.percentile(density, 92.0))
        hotspot_scores = 2.2 * np.maximum(0.0, congestion - cong_thr) + 0.5 * np.maximum(0.0, density - dens_thr)
        flat = np.argsort(hotspot_scores.reshape(-1))[::-1]
        if hotspot_scores.reshape(-1)[flat[0]] <= 0.0:
            return []

        pos = placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64)
        grid_w = benchmark.canvas_width / max(ncol, 1)
        grid_h = benchmark.canvas_height / max(nrow, 1)
        macro_vectors: dict[int, np.ndarray] = {}
        macro_scores: dict[int, float] = {}
        for flat_idx in flat[: max(10, 2 * limit)]:
            score = hotspot_scores.reshape(-1)[flat_idx]
            if score <= 0.0:
                break
            row = int(flat_idx // ncol)
            col = int(flat_idx % ncol)
            cx = (col + 0.5) * grid_w
            cy = (row + 0.5) * grid_h
            for idx in range(benchmark.num_hard_macros):
                if not movable[idx]:
                    continue
                xmin = pos[idx, 0] - widths[idx] * 0.5
                xmax = pos[idx, 0] + widths[idx] * 0.5
                ymin = pos[idx, 1] - heights[idx] * 0.5
                ymax = pos[idx, 1] + heights[idx] * 0.5
                if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                    continue
                vec = pos[idx] - np.array([cx, cy], dtype=np.float64)
                if np.linalg.norm(vec) < 1.0e-9:
                    continue
                macro_vectors[idx] = macro_vectors.get(idx, np.zeros(2, dtype=np.float64)) + score * vec
                macro_scores[idx] = macro_scores.get(idx, 0.0) + score

        ordered = sorted(macro_scores, key=macro_scores.get, reverse=True)[:limit]
        return [(idx, macro_vectors[idx]) for idx in ordered]

        try:
            spec = importlib.util.spec_from_file_location("partcl_submission_outline", outline_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("failed to load outline module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            placer_cls = getattr(module, "SubmissionOutlinePlacer", None)
            if placer_cls is None:
                raise RuntimeError("outline placer class missing")
            placer = placer_cls()
            return placer.place(benchmark)
        except Exception:
            placement = benchmark.macro_positions.clone()
            hard_pos = self._legalize_hard_numpy(
                placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64),
                benchmark,
            )
            placement[: benchmark.num_hard_macros] = torch.tensor(
                hard_pos,
                dtype=placement.dtype,
            )
            return placement
