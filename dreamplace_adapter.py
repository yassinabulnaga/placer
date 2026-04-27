"""
Thin DREAMPlace adapter for the Partcl challenge benchmark format.

This module converts the in-memory challenge benchmark + PlacementCost state
into a temporary LEF/DEF design, runs DREAMPlace through its public CLI, and
maps the resulting DEF placement back to hard-macro center coordinates.

Environment variables:
    PARTCL_DREAMPLACE_ROOT   Path to a DREAMPlace checkout or install root.
    PARTCL_DREAMPLACE_ENTRY  Override the Placer.py entrypoint path.
    PARTCL_DREAMPLACE_PYTHON Python executable for the DREAMPlace run.
    PARTCL_DREAMPLACE_GPU    "1" or "0" to force GPU/CPU.
    PARTCL_DREAMPLACE_KEEP   "1" keeps generated temp directories for debugging.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Dict

import numpy as np
import torch


DB_UNITS = 2000


def run_dreamplace(
    benchmark,
    plc,
    seed,
    config=None,
    hard_macro_indices=None,
    soft_macro_indices=None,
):
    if plc is None:
        print("[partcl:dreamplace] skipped: plc unavailable")
        return None

    dreamplace_root = _resolve_dreamplace_root()
    if dreamplace_root is None:
        print("[partcl:dreamplace] skipped: no DREAMPlace checkout found")
        return None

    entrypoint = _resolve_dreamplace_entrypoint(dreamplace_root)
    if entrypoint is None:
        print(f"[partcl:dreamplace] skipped: no Placer.py entrypoint under {dreamplace_root}")
        return None

    seed = np.asarray(seed, dtype=np.float64)
    config = dict(config or {})

    keep = os.environ.get("PARTCL_DREAMPLACE_KEEP", "0") == "1"
    tmp_ctx = (
        tempfile.TemporaryDirectory(prefix="partcl-dreamplace-")
        if not keep
        else _PersistentTempDir(prefix="partcl-dreamplace-")
    )
    with tmp_ctx as tmpdir:
        workdir = Path(tmpdir)
        design_dir = workdir / benchmark.name
        results_dir = workdir / "results"
        design_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        exported = _export_design(
            benchmark=benchmark,
            plc=plc,
            seed=seed,
            design_dir=design_dir,
            config=config,
        )
        params_path = design_dir / f"{benchmark.name}.json"
        for attempt_name, attempt_cfg in _dreamplace_attempts(benchmark, config):
            _write_params_json(
                benchmark=benchmark,
                config=attempt_cfg,
                exported=exported,
                params_path=params_path,
                results_dir=results_dir,
            )
            out_def = _run_placer_once(
                benchmark=benchmark,
                dreamplace_root=dreamplace_root,
                entrypoint=entrypoint,
                params_path=params_path,
            )
            if out_def is None:
                continue
            if bool(int(attempt_cfg.get("return_all_macros", 0))):
                result_pos = _parse_output_def_all(out_def, exported["instances"], benchmark)
            else:
                result_pos = _parse_output_def(out_def, exported["instances"], benchmark)
            if result_pos is not None:
                print(f"[partcl:dreamplace] success: used {dreamplace_root} mode={attempt_name}")
                return result_pos
            print(f"[partcl:dreamplace] retry: invalid output in mode={attempt_name}")
        return None


def place(*args, **kwargs):
    return run_dreamplace(*args, **kwargs)


def _dreamplace_attempts(benchmark, config: dict):
    primary = dict(config)
    yield "macro", primary

    allow_flat_fallback = bool(int(primary.get("allow_flat_fallback", 1)))
    if "macro_place_flag" in primary or not allow_flat_fallback:
        return

    flat = dict(primary)
    complexity = max(benchmark.num_hard_macros, benchmark.num_macros)
    flat.update(
        {
            "macro_place_flag": 0,
            "single_stage": 1,
            "enable_fillers": 0,
            "density_weight": min(float(flat.get("density_weight", 8.0e-5)), 1.5e-5),
            "target_density": max(float(flat.get("target_density", 0.60)), 0.88),
            "lr": min(float(flat.get("lr", 0.05)), 0.008),
            "steps": 22 if complexity <= 320 else 24,
            "num_bins_scale": 0.5,
            "stage1_bins_scale": 1.0,
            "stage1_iter_scale": 18,
            "gamma": min(float(flat.get("gamma", 4.0)), 4.0),
            "stop_overflow": max(float(flat.get("stop_overflow", 0.10)), 0.10),
            "adjust_rudy_area_flag": 0,
            "adjust_pin_area_flag": 0,
            "max_num_area_adjust": 0,
            "random_center_init_flag": 0,
        }
    )
    yield "flat", flat


def _run_placer_once(benchmark, dreamplace_root: Path, entrypoint: Path, params_path: Path) -> Path | None:
    cmd = [
        os.environ.get("PARTCL_DREAMPLACE_PYTHON", sys.executable),
        str(entrypoint),
        str(params_path),
    ]
    env = os.environ.copy()
    root_str = str(dreamplace_root)
    py_paths = [root_str, str(dreamplace_root / "dreamplace")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_paths)
    proc = subprocess.run(
        cmd,
        cwd=str(dreamplace_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        if _should_retry_cpu(stderr, params_path):
            print("[partcl:dreamplace] retry: switching to CPU mode for this build")
            _force_cpu_params(params_path)
            proc = subprocess.run(
                cmd,
                cwd=str(dreamplace_root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        if proc.returncode != 0:
            print(
                "[partcl:dreamplace] failed: "
                f"root={dreamplace_root} entry={entrypoint} returncode={proc.returncode}"
            )
            stderr = (proc.stderr or "").strip()
            if stderr:
                stderr_lines = stderr.splitlines()
                tail = stderr_lines[-8:] if os.environ.get("PARTCL_DREAMPLACE_VERBOSE", "0") == "1" else stderr_lines[-1:]
                print(f"[partcl:dreamplace] stderr: {' | '.join(tail)}")
            return None

    out_def = params_path.parent.parent / "results" / benchmark.name / f"{benchmark.name}.gp.def"
    if not out_def.exists():
        print(f"[partcl:dreamplace] failed: expected output DEF not found at {out_def}")
        return None
    return out_def


def _should_retry_cpu(stderr: str, params_path: Path) -> bool:
    if "CANNOT enable GPU without CUDA compiled" not in stderr:
        return False
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(params.get("gpu", 0)) != 0


def _force_cpu_params(params_path: Path) -> None:
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["gpu"] = 0
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def _resolve_dreamplace_root() -> Path | None:
    candidates = []

    def add_root_options(path: Path) -> None:
        install = path / "install"
        if install not in candidates:
            candidates.append(install)
        if path not in candidates:
            candidates.append(path)

    env_root = os.environ.get("PARTCL_DREAMPLACE_ROOT")
    if env_root:
        add_root_options(Path(env_root))
    here = Path(__file__).resolve().parent
    add_root_options(here / "dreamplace")
    add_root_options(here.parent / "dreamplace")
    add_root_options(Path("/submission/dreamplace"))
    for path in candidates:
        if path.exists() and _resolve_dreamplace_entrypoint(path) is not None:
            return path.resolve()
    return None


def _resolve_dreamplace_entrypoint(root: Path) -> Path | None:
    env_entry = os.environ.get("PARTCL_DREAMPLACE_ENTRY")
    if env_entry:
        path = Path(env_entry)
        if path.exists():
            return path.resolve()
    candidates = [
        root / "dreamplace" / "Placer.py",
        root / "Placer.py",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _export_design(benchmark, plc, seed: np.ndarray, design_dir: Path, config: dict | None = None) -> Dict[str, object]:
    pin_map = _collect_instance_pins(plc)
    instances, virtual_ports = _build_instances(benchmark, plc, seed, pin_map, config=config)
    ports = _build_ports(plc, virtual_ports)
    nets = _build_nets(plc, instances, ports, virtual_ports)

    tech_lef = design_dir / f"{benchmark.name}.tech.lef"
    cells_lef = design_dir / f"{benchmark.name}.cells.lef"
    init_def = design_dir / f"{benchmark.name}.def"

    die_width_db = max(_db(benchmark.canvas_width), 1)
    die_height_db = max(_db(benchmark.canvas_height), 1)
    # Use a std-cell-pitch site/row grid rather than the coarse contest-grid pitch.
    # DREAMPlace's electrostatic spreading and legalizer assume sites much finer
    # than macros; one grid-cell-per-site under-resolves spreading by ~4-16x for
    # typical IBM benchmark macro sizes.
    cfg = config or {}
    # Env override lets us regress to the legacy coarse-grid export without
    # threading a config flag through every call site.
    if os.environ.get("PARTCL_DREAMPLACE_FINE_SITE", "1") == "0":
        fine_site_grid = False
    else:
        fine_site_grid = bool(int(cfg.get("fine_site_grid", 1)))
    if fine_site_grid:
        try:
            sizes = benchmark.macro_sizes.cpu().numpy().astype(float)
            num_macros = int(benchmark.num_macros)
            min_w = float(sizes[:num_macros, 0].min()) if num_macros > 0 else 0.0
            min_h = float(sizes[:num_macros, 1].min()) if num_macros > 0 else 0.0
        except Exception:
            min_w = 0.0
            min_h = 0.0
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        coarse_site_w = canvas_w / max(benchmark.grid_cols, 1)
        coarse_row_h = canvas_h / max(benchmark.grid_rows, 1)
        # Target site/row at least 4x finer than the smallest macro, but bounded
        # so we do not blow up LEF/DEF size: at most ~2048 sites across the die.
        target_site_w = max(min_w * 0.25, canvas_w / 2048.0)
        target_row_h = max(min_h * 0.25, canvas_h / 2048.0)
        # Never coarser than the previous contest-grid pitch (regression safety).
        target_site_w = min(target_site_w, coarse_site_w)
        target_row_h = min(target_row_h, coarse_row_h)
        # Avoid pathological values.
        target_site_w = max(target_site_w, canvas_w / 4096.0)
        target_row_h = max(target_row_h, canvas_h / 4096.0)
        site_width_db = max(_db(target_site_w), 1)
        row_height_db = max(_db(target_row_h), 1)
    else:
        site_width_db = max(die_width_db // max(benchmark.grid_cols, 1), 1)
        row_height_db = max(die_height_db // max(benchmark.grid_rows, 1), 1)
    site_width = site_width_db / DB_UNITS
    row_height = row_height_db / DB_UNITS

    env_layers = os.environ.get("PARTCL_DREAMPLACE_LAYERS")
    if env_layers is not None and env_layers.strip():
        try:
            num_routing_layers = max(1, int(env_layers))
        except ValueError:
            num_routing_layers = max(1, int((config or {}).get("num_routing_layers", 4)))
    else:
        num_routing_layers = max(1, int((config or {}).get("num_routing_layers", 4)))
    _write_tech_lef(
        tech_lef,
        site_width=site_width,
        row_height=row_height,
        num_routing_layers=num_routing_layers,
    )
    _write_cells_lef(cells_lef, instances)
    _write_def(
        init_def,
        benchmark,
        instances,
        ports,
        nets,
        site_width=site_width,
        row_height=row_height,
    )

    return {
        "tech_lef": tech_lef,
        "cells_lef": cells_lef,
        "init_def": init_def,
        "instances": instances,
        "site_width_db": site_width_db,
        "row_height_db": row_height_db,
        "num_routing_layers": num_routing_layers,
    }


def _collect_instance_pins(plc) -> Dict[str, dict]:
    pin_map: Dict[str, dict] = {}
    for idx, mod in enumerate(plc.modules_w_pins):
        mod_type = mod.get_type()
        if mod_type != "MACRO_PIN" or not hasattr(mod, "get_macro_name"):
            continue
        macro_name = mod.get_macro_name()
        pin_name = mod.get_name().split("/")[-1]
        pin_map.setdefault(macro_name, {})[pin_name] = {
            "x_offset": float(getattr(mod, "x_offset", 0.0)),
            "y_offset": float(getattr(mod, "y_offset", 0.0)),
            "idx": idx,
        }
    return pin_map


def _build_instances(
    benchmark,
    plc,
    seed: np.ndarray,
    pin_map: Dict[str, dict],
    config: dict | None = None,
) -> tuple[Dict[str, dict], Dict[str, dict]]:
    instances: Dict[str, dict] = {}
    virtual_ports: Dict[str, dict] = {}
    num_hard = benchmark.num_hard_macros
    config = config or {}
    fix_soft_macros = bool(int(config.get("fix_soft_macros", 0)))
    omit_soft_macros = bool(int(config.get("omit_soft_macros", 0)))
    all_indices = list(plc.hard_macro_indices) + list(plc.soft_macro_indices)

    for bench_idx, plc_idx in enumerate(all_indices):
        node = plc.modules_w_pins[plc_idx]
        orig_name = node.get_name()
        inst_name = _sanitize(orig_name, prefix="inst")
        master_name = f"{inst_name}_MASTER"
        width = float(node.get_width())
        height = float(node.get_height())
        if bench_idx < num_hard:
            x, y = float(seed[bench_idx, 0]), float(seed[bench_idx, 1])
            fixed = bool(node.get_fix_flag())
        else:
            x0, y0 = node.get_pos()
            x, y = float(x0), float(y0)
            fixed = bool(node.get_fix_flag()) or fix_soft_macros

        pin_defs = {}
        for pin_name, pin_info in pin_map.get(orig_name, {}).items():
            pin_defs[_sanitize(pin_name, prefix="pin")] = {
                "orig_name": pin_name,
                "x_offset": float(pin_info["x_offset"]),
                "y_offset": float(pin_info["y_offset"]),
            }

        if not pin_defs:
            pin_defs["P0"] = {"orig_name": "P0", "x_offset": 0.0, "y_offset": 0.0}

        if bench_idx >= num_hard and omit_soft_macros:
            for pin_name, pin in pin_defs.items():
                port_name = _sanitize(f"{orig_name}_{pin_name}", prefix="vport")
                virtual_ports[f"{orig_name}/{pin['orig_name']}"] = {
                    "name": port_name,
                    "x": float(x + pin["x_offset"]),
                    "y": float(y + pin["y_offset"]),
                }
            continue

        instances[orig_name] = {
            "bench_idx": bench_idx,
            "plc_idx": plc_idx,
            "inst_name": inst_name,
            "master_name": master_name,
            "width": width,
            "height": height,
            "x": x,
            "y": y,
            "fixed": fixed,
            "pin_defs": pin_defs,
        }
    return instances, virtual_ports


def _build_ports(plc, virtual_ports: Dict[str, dict] | None = None) -> Dict[str, dict]:
    ports = {}
    for idx in plc.port_indices:
        node = plc.modules_w_pins[idx]
        name = node.get_name()
        x, y = node.get_pos()
        ports[name] = {
            "name": _sanitize(name, prefix="port"),
            "x": float(x),
            "y": float(y),
        }
    for key, value in (virtual_ports or {}).items():
        ports[key] = dict(value)
    return ports


def _build_nets(
    plc,
    instances: Dict[str, dict],
    ports: Dict[str, dict],
    virtual_ports: Dict[str, dict] | None = None,
) -> list[dict]:
    nets = []
    virtual_ports = virtual_ports or {}
    for net_idx, (driver, sinks) in enumerate(plc.nets.items()):
        terms = []
        seen = set()
        for pin_ref in [driver] + list(sinks):
            if "/" in pin_ref:
                inst_orig, pin_orig = pin_ref.split("/", 1)
                inst = instances.get(inst_orig)
                if inst is None:
                    port = virtual_ports.get(pin_ref)
                    if port is not None:
                        term = ("pin", port["name"])
                    else:
                        continue
                else:
                    pin_name = _match_pin_name(inst["pin_defs"], pin_orig)
                    term = ("inst", inst["inst_name"], pin_name)
            else:
                port = ports.get(pin_ref)
                if port is None:
                    continue
                term = ("pin", port["name"])
            if term not in seen:
                seen.add(term)
                terms.append(term)
        if len(terms) >= 2:
            nets.append({"name": f"NET_{net_idx}", "terms": terms})
    return nets


def _match_pin_name(pin_defs: Dict[str, dict], pin_orig: str) -> str:
    pin_orig_s = _sanitize(pin_orig, prefix="pin")
    if pin_orig_s in pin_defs:
        return pin_orig_s
    for pin_name, pin_def in pin_defs.items():
        if pin_def["orig_name"] == pin_orig:
            return pin_name
    return next(iter(pin_defs))


def _write_tech_lef(path: Path, site_width: float, row_height: float, num_routing_layers: int = 4) -> None:
    """Tech LEF with multiple alternating routing layers.

    Direction alternates HORIZONTAL/VERTICAL; pitch is set to a fraction of
    site_width so the layer count is meaningful relative to the placement grid.
    DREAMPlace's routability optimization indexes per-layer capacities, so a
    single-layer LEF (the prior behavior) starves the routability gradient.
    """
    num_routing_layers = max(1, int(num_routing_layers))
    pitch = max(site_width * 0.5, 0.05)
    width = max(pitch * 0.5, 0.025)
    layers = []
    for layer_idx in range(num_routing_layers):
        direction = "HORIZONTAL" if layer_idx % 2 == 0 else "VERTICAL"
        layers.append(
            f"LAYER metal{layer_idx + 1}\n"
            f"  TYPE ROUTING ;\n"
            f"  DIRECTION {direction} ;\n"
            f"  PITCH {pitch:.6f} ;\n"
            f"  WIDTH {width:.6f} ;\n"
            f"END metal{layer_idx + 1}"
        )
    layers_block = "\n".join(layers)
    content = (
        f"VERSION 5.8 ;\n"
        f"BUSBITCHARS \"[]\" ;\n"
        f"DIVIDERCHAR \"/\" ;\n"
        f"UNITS\n"
        f"  DATABASE MICRONS 2000 ;\n"
        f"END UNITS\n"
        f"MANUFACTURINGGRID 0.001 ;\n"
        f"{layers_block}\n"
        f"SITE CORE\n"
        f"  CLASS CORE ;\n"
        f"  SIZE {site_width:.6f} BY {row_height:.6f} ;\n"
        f"END CORE\n"
        f"END LIBRARY\n"
    )
    path.write_text(content, encoding="utf-8")


def _write_cells_lef(path: Path, instances: Dict[str, dict]) -> None:
    lines = [
        'VERSION 5.8 ;',
        'BUSBITCHARS "[]" ;',
        'DIVIDERCHAR "/" ;',
        'UNITS',
        f'  DATABASE MICRONS {DB_UNITS} ;',
        'END UNITS',
    ]
    for inst in instances.values():
        lines.extend(
            [
                f"MACRO {inst['master_name']}",
                "  CLASS BLOCK ;",
                "  ORIGIN 0 0 ;",
                f"  SIZE {inst['width']:.6f} BY {inst['height']:.6f} ;",
                "  SYMMETRY X Y ;",
                "  SITE CORE ;",
            ]
        )
        for pin_name, pin in inst["pin_defs"].items():
            px = inst["width"] * 0.5 + pin["x_offset"]
            py = inst["height"] * 0.5 + pin["y_offset"]
            half = 0.01
            xl = max(0.0, px - half)
            yl = max(0.0, py - half)
            xh = min(inst["width"], px + half)
            yh = min(inst["height"], py + half)
            lines.extend(
                [
                    f"  PIN {pin_name}",
                    "    DIRECTION INOUT ;",
                    "    USE SIGNAL ;",
                    "    PORT",
                    "      LAYER metal1 ;",
                    f"      RECT {xl:.6f} {yl:.6f} {xh:.6f} {yh:.6f} ;",
                    "    END",
                    f"  END {pin_name}",
                ]
            )
        lines.extend([f"END {inst['master_name']}"])
    lines.append("END LIBRARY")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_def(
    path: Path,
    benchmark,
    instances: Dict[str, dict],
    ports: Dict[str, dict],
    nets: list[dict],
    site_width: float,
    row_height: float,
) -> None:
    site_width_db = max(_db(site_width), 1)
    row_height_db = max(_db(row_height), 1)
    lines = [
        "VERSION 5.8 ;",
        'BUSBITCHARS "[]" ;',
        'DIVIDERCHAR "/" ;',
        f"DESIGN {benchmark.name} ;",
        f"UNITS DISTANCE MICRONS {DB_UNITS} ;",
        f"DIEAREA ( 0 0 ) ( {_db(benchmark.canvas_width)} {_db(benchmark.canvas_height)} ) ;",
    ]
    die_width_db = _db(benchmark.canvas_width)
    die_height_db = _db(benchmark.canvas_height)
    sites_per_row = max(1, die_width_db // max(site_width_db, 1))
    num_rows = max(1, die_height_db // max(row_height_db, 1))
    # Cap row count at 2048 so DEF size stays manageable on large benchmarks.
    if num_rows > 2048:
        num_rows = 2048
        row_height_db = max(1, die_height_db // num_rows)
    if sites_per_row > 4096:
        sites_per_row = 4096
        site_width_db = max(1, die_width_db // sites_per_row)
    for row in range(num_rows):
        y_db = row * row_height_db
        orient = "N" if row % 2 == 0 else "FS"
        lines.append(
            f"ROW ROW_{row} CORE 0 {y_db} {orient} DO {sites_per_row} BY 1 STEP {site_width_db} 0 ;"
        )

    lines.append(f"COMPONENTS {len(instances)} ;")
    for inst in instances.values():
        x_ll = inst["x"] - inst["width"] * 0.5
        y_ll = inst["y"] - inst["height"] * 0.5
        status = "FIXED" if inst["fixed"] else "PLACED"
        lines.append(
            f"  - {inst['inst_name']} {inst['master_name']} + {status} ( {_db(x_ll)} {_db(y_ll)} ) N ;"
        )
    lines.append("END COMPONENTS")

    port_to_net = {}
    for net in nets:
        for term in net["terms"]:
            if term[0] == "pin":
                port_to_net[term[1]] = net["name"]

    lines.append(f"PINS {len(ports)} ;")
    pin_box = max(_db(0.02), 1)
    for port in ports.values():
        net_name = port_to_net.get(port["name"], port["name"])
        lines.extend(
            [
                f"  - {port['name']} + NET {net_name} + DIRECTION INPUT + USE SIGNAL",
                f"    + LAYER metal1 ( {-pin_box} {-pin_box} ) ( {pin_box} {pin_box} )",
                f"    + FIXED ( {_db(port['x'])} {_db(port['y'])} ) N ;",
            ]
        )
    lines.append("END PINS")

    lines.append(f"NETS {len(nets)} ;")
    for net in nets:
        terms = []
        for term in net["terms"]:
            if term[0] == "pin":
                terms.append(f"( PIN {term[1]} )")
            else:
                terms.append(f"( {term[1]} {term[2]} )")
        lines.append(f"  - {net['name']} {' '.join(terms)} ;")
    lines.append("END NETS")
    lines.append("END DESIGN")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_params_json(benchmark, config: dict, exported: Dict[str, object], params_path: Path, results_dir: Path) -> None:
    complexity = max(benchmark.num_hard_macros, benchmark.num_macros)
    util = float(
        torch.sum(benchmark.macro_sizes[:, 0] * benchmark.macro_sizes[:, 1]).item()
        / max(benchmark.canvas_width * benchmark.canvas_height, 1.0e-6)
    )
    base_bins = 128 if complexity <= 320 else 192 if complexity <= 430 else 256
    num_bins = int(
        np.clip(
            round(base_bins * float(config.get("num_bins_scale", 1.0))),
            64,
            320,
        )
    )
    target_density = float(
        np.clip(
            float(
                config.get(
                    "target_density",
                    np.clip(max(util * 1.10, 0.55), 0.55, 0.92),
                )
            ),
            0.45,
            0.92,
        )
    )
    stage1_bins = int(
        np.clip(
            round(num_bins * float(config.get("stage1_bins_scale", 0.75))),
            64,
            max(64, num_bins),
        )
    )
    stage2_bins = int(
        np.clip(
            round(num_bins * float(config.get("stage2_bins_scale", 1.0))),
            64,
            384,
        )
    )
    stage1_iter = max(200, int(int(config.get("steps", 24)) * float(config.get("stage1_iter_scale", 18))))
    stage2_iter = max(200, int(int(config.get("steps", 24)) * float(config.get("stage2_iter_scale", 28))))
    route_bins_x = int(np.clip(round(stage2_bins * float(config.get("route_bins_scale", 1.0))), 64, 512))
    route_bins_y = int(np.clip(round(stage2_bins * float(config.get("route_bins_scale", 1.0))), 64, 512))
    macro_halo_x = float(config.get("macro_halo_x", 0.0))
    macro_halo_y = float(config.get("macro_halo_y", 0.0))
    if macro_halo_x == 0.0 or macro_halo_y == 0.0:
        avg_w = float(torch.mean(benchmark.macro_sizes[: benchmark.num_hard_macros, 0]).item()) if benchmark.num_hard_macros else 0.0
        avg_h = float(torch.mean(benchmark.macro_sizes[: benchmark.num_hard_macros, 1]).item()) if benchmark.num_hard_macros else 0.0
        default_halo_scale = float(config.get("macro_halo_scale", 0.0))
        macro_halo_x = macro_halo_x or avg_w * default_halo_scale
        macro_halo_y = macro_halo_y or avg_h * default_halo_scale
    single_stage = bool(int(config.get("single_stage", 0)))
    global_place_stages = [
        {
            "num_bins_x": stage1_bins,
            "num_bins_y": stage1_bins,
            "iteration": stage1_iter,
            "learning_rate": float(config.get("lr", 0.05)) * float(config.get("stage1_lr_scale", 1.0)),
            "wirelength": "weighted_average",
            "optimizer": "nesterov",
            "Llambda_density_weight_iteration": int(config.get("stage1_lambda_iter", 1)),
            "Lsub_iteration": int(config.get("stage1_sub_iter", 1)),
        },
    ]
    if not single_stage:
        global_place_stages.append(
            {
                "num_bins_x": stage2_bins,
                "num_bins_y": stage2_bins,
                "iteration": stage2_iter,
                "learning_rate": float(config.get("lr", 0.05)) * float(config.get("stage2_lr_scale", 0.65)),
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": int(config.get("stage2_lambda_iter", 2)),
                "Lsub_iteration": int(config.get("stage2_sub_iter", 1)),
            },
        )

    params = {
        "lef_input": [
            str(exported["tech_lef"]),
            str(exported["cells_lef"]),
        ],
        "def_input": str(exported["init_def"]),
        "gpu": 1 if os.environ.get("PARTCL_DREAMPLACE_GPU", "1") != "0" else 0,
        "num_bins_x": num_bins,
        "num_bins_y": num_bins,
        "global_place_stages": global_place_stages,
        "target_density": target_density,
        "density_weight": float(config.get("density_weight", 8.0e-5)),
        "gamma": float(config.get("gamma", 4.0)),
        "random_seed": int(config.get("random_seed", 1000 + sum(ord(ch) for ch in benchmark.name))),
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": int(config.get("enable_fillers", 1)),
        "gp_noise_ratio": float(config.get("gp_noise_ratio", 0.01)),
        "global_place_flag": 1,
        "legalize_flag": 0,
        "detailed_place_flag": 0,
        "detailed_place_engine": "",
        "detailed_place_command": "",
        "stop_overflow": float(config.get("stop_overflow", 0.10)),
        "dtype": "float32",
        "plot_flag": 0,
        "random_center_init_flag": int(config.get("random_center_init_flag", 0)),
        "gift_init_flag": int(config.get("gift_init_flag", 0)),
        "sort_nets_by_degree": 0,
        "num_threads": int(os.environ.get("PARTCL_DREAMPLACE_THREADS", "8")),
        "deterministic_flag": 1,
        "use_bb": int(config.get("use_bb", 1)),
        "macro_place_flag": int(config.get("macro_place_flag", 1)),
        "two_stage_density_scaler": float(config.get("two_stage_density_scaler", 350.0)),
        "macro_halo_x": macro_halo_x,
        "macro_halo_y": macro_halo_y,
        "routability_opt_flag": int(config.get("routability_opt_flag", 0)),
        "route_num_bins_x": route_bins_x,
        "route_num_bins_y": route_bins_y,
        "adjust_nctugr_area_flag": 0,
        "adjust_rudy_area_flag": int(config.get("adjust_rudy_area_flag", 1)),
        "adjust_pin_area_flag": int(config.get("adjust_pin_area_flag", 1)),
        "max_num_area_adjust": int(config.get("max_num_area_adjust", 2)),
        "route_area_adjust_stop_ratio": float(config.get("route_area_adjust_stop_ratio", 0.01)),
        "pin_area_adjust_stop_ratio": float(config.get("pin_area_adjust_stop_ratio", 0.05)),
        "node_area_adjust_overflow": float(config.get("node_area_adjust_overflow", 0.12)),
        # With a multi-layer tech LEF, DREAMPlace builds one capacity array per
        # routing layer. Split the total per-direction capacity across the
        # available H/V layers so the integral is preserved while routability
        # gradients have meaningful per-layer headroom to allocate.
        "unit_horizontal_capacity": float(
            config.get(
                "unit_horizontal_capacity",
                float(benchmark.hroutes_per_micron) / max(1, (exported["num_routing_layers"] + 1) // 2),
            )
        ),
        "unit_vertical_capacity": float(
            config.get(
                "unit_vertical_capacity",
                float(benchmark.vroutes_per_micron) / max(1, exported["num_routing_layers"] // 2),
            )
        ),
        "result_dir": str(results_dir),
    }
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def _parse_output_def(path: Path, instances: Dict[str, dict], benchmark) -> np.ndarray | None:
    all_pos = _parse_output_def_all(path, instances, benchmark)
    if all_pos is None:
        return None
    hard_pos = all_pos[: benchmark.num_hard_macros].copy()
    if not _validate_hard_pos(hard_pos, benchmark):
        return None
    return hard_pos


def _parse_output_def_all(path: Path, instances: Dict[str, dict], benchmark) -> np.ndarray | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"-\s+(?P<inst>\S+)\s+\S+\s+\+\s+(?:FIXED|PLACED)\s+\(\s*(?P<x>-?\d+)\s+(?P<y>-?\d+)\s*\)",
        re.MULTILINE,
    )
    by_inst = {inst["inst_name"]: inst for inst in instances.values()}
    macro_pos = np.zeros((benchmark.num_macros, 2), dtype=np.float64)
    found = np.zeros(benchmark.num_macros, dtype=bool)
    sentinel_count = 0
    for match in pattern.finditer(text):
        inst_name = match.group("inst")
        inst = by_inst.get(inst_name)
        if inst is None:
            continue
        bench_idx = inst["bench_idx"]
        if bench_idx >= benchmark.num_macros:
            continue
        x_db = int(match.group("x"))
        y_db = int(match.group("y"))
        if x_db <= -2_000_000_000 or y_db <= -2_000_000_000:
            sentinel_count += 1
            continue
        x_ll = x_db / DB_UNITS
        y_ll = y_db / DB_UNITS
        macro_pos[bench_idx, 0] = x_ll + inst["width"] * 0.5
        macro_pos[bench_idx, 1] = y_ll + inst["height"] * 0.5
        found[bench_idx] = True
    if not np.all(found):
        if sentinel_count:
            print(
                "[partcl:dreamplace] rejected: "
                f"{sentinel_count} macros had sentinel DEF coordinates"
            )
        return None
    if not _validate_macro_pos(macro_pos, benchmark):
        return None
    return macro_pos


def _validate_macro_pos(macro_pos: np.ndarray, benchmark) -> bool:
    if not np.all(np.isfinite(macro_pos)):
        print("[partcl:dreamplace] rejected: non-finite macro coordinates")
        return False

    widths = benchmark.macro_sizes[: benchmark.num_macros, 0].cpu().numpy().astype(np.float64)
    heights = benchmark.macro_sizes[: benchmark.num_macros, 1].cpu().numpy().astype(np.float64)
    min_x = widths * 0.5
    max_x = benchmark.canvas_width - widths * 0.5
    min_y = heights * 0.5
    max_y = benchmark.canvas_height - heights * 0.5
    out_of_bounds = (
        (macro_pos[:, 0] < (min_x - 1.0e-3))
        | (macro_pos[:, 0] > (max_x + 1.0e-3))
        | (macro_pos[:, 1] < (min_y - 1.0e-3))
        | (macro_pos[:, 1] > (max_y + 1.0e-3))
    )
    if np.any(out_of_bounds):
        print(
            "[partcl:dreamplace] rejected: "
            f"{int(np.count_nonzero(out_of_bounds))} macros out of canvas bounds"
        )
        return False

    return _validate_hard_pos(macro_pos[: benchmark.num_hard_macros], benchmark)


def _validate_hard_pos(hard_pos: np.ndarray, benchmark) -> bool:
    if not np.all(np.isfinite(hard_pos)):
        print("[partcl:dreamplace] rejected: non-finite macro coordinates")
        return False

    widths = benchmark.macro_sizes[: benchmark.num_hard_macros, 0].cpu().numpy().astype(np.float64)
    heights = benchmark.macro_sizes[: benchmark.num_hard_macros, 1].cpu().numpy().astype(np.float64)
    min_x = widths * 0.5
    max_x = benchmark.canvas_width - widths * 0.5
    min_y = heights * 0.5
    max_y = benchmark.canvas_height - heights * 0.5
    out_of_bounds = (
        (hard_pos[:, 0] < (min_x - 1.0e-3))
        | (hard_pos[:, 0] > (max_x + 1.0e-3))
        | (hard_pos[:, 1] < (min_y - 1.0e-3))
        | (hard_pos[:, 1] > (max_y + 1.0e-3))
    )
    if np.any(out_of_bounds):
        print(
            "[partcl:dreamplace] rejected: "
            f"{int(np.count_nonzero(out_of_bounds))} hard macros out of canvas bounds"
        )
        return False

    spread_x = float(np.std(hard_pos[:, 0]))
    spread_y = float(np.std(hard_pos[:, 1]))
    min_spread_x = max(benchmark.canvas_width * 0.01, 1.0)
    min_spread_y = max(benchmark.canvas_height * 0.01, 1.0)
    if spread_x < min_spread_x and spread_y < min_spread_y:
        print(
            "[partcl:dreamplace] rejected: "
            f"collapsed macro spread std=({spread_x:.4f}, {spread_y:.4f})"
        )
        return False

    return True


def _sanitize(text: str, prefix: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", text)
    if not clean or clean[0].isdigit():
        clean = f"{prefix}_{clean}"
    return clean


def _db(value: float) -> int:
    return int(round(float(value) * DB_UNITS))


class _PersistentTempDir:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.path = None

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix=self.prefix)
        return self.path

    def __exit__(self, exc_type, exc, tb):
        return False
