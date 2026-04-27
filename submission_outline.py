"""
Submission Outline - Runnable starter submission.

This file is meant to be edited into your real competition submission.
It is structured in phases so you can replace each stage independently:

1. Start from a randomized placement.
2. Apply your optimization logic to hard macros.
3. Legalize hard macros to remove overlaps.
4. Optionally update soft macros.

The default implementation is intentionally simple:
- It starts movable macros from random in-canvas positions.
- It uses a minimum-displacement style legalization pass for hard macros.
- It leaves soft macros at their randomized starting locations.

Usage:
    uv run evaluate submissions/submission_outline.py
    uv run evaluate submissions/submission_outline.py --all
    uv run evaluate submissions/submission_outline.py -b ibm03
"""

import os

import torch

from macro_place.benchmark import Benchmark


class SubmissionOutlinePlacer:
    """
    Runnable submission scaffold.

    Replace the helper methods one by one as your algorithm improves.
    """

    def __init__(self, gap: float = 0.001, search_rings: int = 48, seed: int = 17):
        self.gap = gap
        self.search_rings = search_rings
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = self._initialize_placement(benchmark)
        placement = self._optimize_hard_macros(placement, benchmark)
        placement = self._legalize_hard_macros(placement, benchmark)
        placement = self._update_soft_macros(placement, benchmark)
        return placement

    def _initialize_placement(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Starting point for your algorithm.

        Good replacements for this stage:
        - smarter random restarts
        - analytical/global placement initialization
        - learned initialization
        """
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask()
        if not movable.any():
            return placement

        env_seed = os.environ.get("PARTCL_RANDOM_INIT_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = self.seed
        else:
            seed = self.seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(benchmark.name))

        gen = torch.Generator(device=placement.device if placement.device.type != "cpu" else "cpu")
        gen.manual_seed(seed)

        hard_seed = self._seed_hard_macros(benchmark, gen)
        placement[: benchmark.num_hard_macros] = hard_seed
        placement = self._seed_soft_macros(placement, benchmark, gen)
        placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return placement

    def _seed_hard_macros(self, benchmark: Benchmark, gen: torch.Generator) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        hard = benchmark.macro_positions[:num_hard].clone()
        if num_hard == 0:
            return hard

        widths = benchmark.macro_sizes[:num_hard, 0]
        heights = benchmark.macro_sizes[:num_hard, 1]
        x_min = widths * 0.5
        x_max = benchmark.canvas_width - widths * 0.5
        y_min = heights * 0.5
        y_max = benchmark.canvas_height - heights * 0.5

        areas = widths * heights
        conn = torch.zeros((num_hard, num_hard), dtype=torch.float32)
        degree = torch.zeros(num_hard, dtype=torch.float32)
        for net_nodes, weight in zip(benchmark.net_nodes, benchmark.net_weights):
            hard_nodes = net_nodes[net_nodes < num_hard]
            if hard_nodes.numel() == 0:
                continue
            hard_nodes = torch.unique(hard_nodes)
            degree[hard_nodes] += float(weight)
            if hard_nodes.numel() < 2:
                continue
            pairs = float(weight) / max(int(hard_nodes.numel()) - 1, 1)
            for i in range(hard_nodes.numel()):
                a = int(hard_nodes[i])
                for j in range(i + 1, hard_nodes.numel()):
                    b = int(hard_nodes[j])
                    conn[a, b] += pairs
                    conn[b, a] += pairs

        aspect = benchmark.canvas_width / max(benchmark.canvas_height, 1.0e-9)
        cols = max(2, int(round((num_hard * aspect) ** 0.5)))
        rows = max(2, (num_hard + cols - 1) // cols)
        cell_w = benchmark.canvas_width / cols
        cell_h = benchmark.canvas_height / rows
        anchors = []
        for idx in range(num_hard):
            r = idx // cols
            c = idx % cols
            center_x = min(max((c + 0.5) * cell_w, float(x_min[idx])), float(x_max[idx]))
            center_y = min(max((r + 0.5) * cell_h, float(y_min[idx])), float(y_max[idx]))
            jitter_x = (torch.rand((), generator=gen).item() - 0.5) * 0.35 * cell_w
            jitter_y = (torch.rand((), generator=gen).item() - 0.5) * 0.35 * cell_h
            center_x = min(max(center_x + jitter_x, float(x_min[idx])), float(x_max[idx]))
            center_y = min(max(center_y + jitter_y, float(y_min[idx])), float(y_max[idx]))
            anchors.append((center_x, center_y))
        anchor_points = torch.tensor(anchors, dtype=hard.dtype)

        center = torch.tensor(
            [0.5 * benchmark.canvas_width, 0.5 * benchmark.canvas_height],
            dtype=hard.dtype,
        )
        anchor_perm = torch.argsort(
            (anchor_points[:, 0] - center[0]).abs() + 1.15 * (anchor_points[:, 1] - center[1]).abs()
        ).tolist()
        order = torch.argsort(-(areas + 0.35 * degree)).tolist()
        movable = benchmark.get_movable_mask()[:num_hard]
        used = [False] * num_hard
        placed = []
        for idx in order:
            if not movable[idx]:
                placed.append(idx)
                continue

            best_anchor = None
            best_score = None
            for anchor_idx in anchor_perm:
                if used[anchor_idx]:
                    continue
                target = anchor_points[anchor_idx]
                score = 0.0
                if placed:
                    for other in placed:
                        w = float(conn[idx, other])
                        if w <= 0.0:
                            continue
                        score += w * float(torch.norm(target - hard[other], p=1))
                score += 0.08 * float(torch.norm(target - center, p=1))
                score += 0.02 * float(torch.norm(target - hard[idx], p=1))
                if best_score is None or score < best_score:
                    best_score = score
                    best_anchor = anchor_idx
            if best_anchor is None:
                continue
            hard[idx] = anchor_points[best_anchor]
            used[best_anchor] = True
            placed.append(idx)

        fixed = benchmark.macro_fixed[:num_hard]
        hard[fixed] = benchmark.macro_positions[:num_hard][fixed]
        return hard

    def _seed_soft_macros(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        gen: torch.Generator,
    ) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        if num_soft == 0:
            return placement

        soft_start = num_hard
        widths = benchmark.macro_sizes[:, 0]
        heights = benchmark.macro_sizes[:, 1]
        x_min = widths * 0.5
        x_max = benchmark.canvas_width - widths * 0.5
        y_min = heights * 0.5
        y_max = benchmark.canvas_height - heights * 0.5

        soft_bary = [torch.zeros(2, dtype=placement.dtype) for _ in range(num_soft)]
        soft_weight = torch.zeros(num_soft, dtype=placement.dtype)
        for net_nodes, weight in zip(benchmark.net_nodes, benchmark.net_weights):
            soft_nodes = net_nodes[(net_nodes >= soft_start) & (net_nodes < benchmark.num_macros)]
            if soft_nodes.numel() == 0:
                continue
            anchors = net_nodes[net_nodes < soft_start]
            if anchors.numel() == 0:
                continue
            center = placement[anchors].mean(dim=0)
            for node in torch.unique(soft_nodes):
                soft_idx = int(node) - soft_start
                soft_bary[soft_idx] = soft_bary[soft_idx] + float(weight) * center
                soft_weight[soft_idx] += float(weight)

        for local_idx in range(num_soft):
            idx = soft_start + local_idx
            if benchmark.macro_fixed[idx]:
                placement[idx] = benchmark.macro_positions[idx]
                continue
            if soft_weight[local_idx] > 0:
                target = soft_bary[local_idx] / soft_weight[local_idx]
            else:
                target = torch.tensor(
                    [
                        x_min[idx] + torch.rand((), generator=gen) * torch.clamp(x_max[idx] - x_min[idx], min=0.0),
                        y_min[idx] + torch.rand((), generator=gen) * torch.clamp(y_max[idx] - y_min[idx], min=0.0),
                    ],
                    dtype=placement.dtype,
                )
            jitter = torch.tensor(
                [
                    (torch.rand((), generator=gen).item() - 0.5) * 0.08 * benchmark.canvas_width,
                    (torch.rand((), generator=gen).item() - 0.5) * 0.08 * benchmark.canvas_height,
                ],
                dtype=placement.dtype,
            )
            target = target + jitter
            target[0] = torch.clamp(target[0], min=x_min[idx], max=x_max[idx])
            target[1] = torch.clamp(target[1], min=y_min[idx], max=y_max[idx])
            placement[idx] = target
        return placement

    def _optimize_hard_macros(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> torch.Tensor:
        """
        Main optimization stage for hard macros.

        This outline runs a lightweight analytical refinement from the seeded
        start so we are not relying on the benchmark's provided placement.
        """
        num_hard = benchmark.num_hard_macros
        if num_hard == 0:
            return placement

        result = placement.clone()
        hard = result[:num_hard].clone()
        fixed = benchmark.macro_fixed[:num_hard]
        movable = ~fixed
        if not movable.any():
            return result

        widths = benchmark.macro_sizes[:num_hard, 0]
        heights = benchmark.macro_sizes[:num_hard, 1]
        half_w = widths * 0.5
        half_h = heights * 0.5
        degree, edge_i, edge_j, edge_w = self._build_hard_graph(benchmark)
        port_pull = self._build_port_pulls(benchmark)
        base_seed = hard.clone()

        stage_cfgs = (
            {"steps": 36, "lr": 0.32, "graph_w": 0.95, "seed_w": 0.10, "repel_w": 0.18},
            {"steps": 28, "lr": 0.18, "graph_w": 0.70, "seed_w": 0.06, "repel_w": 0.24},
        )

        for stage in stage_cfgs:
            for step in range(stage["steps"]):
                grad = torch.zeros_like(hard)

                if edge_w.numel() > 0:
                    nbr_sum = torch.zeros_like(hard)
                    nbr_w = torch.zeros(num_hard, dtype=hard.dtype)
                    nbr_sum.index_add_(0, edge_i, hard[edge_j] * edge_w[:, None])
                    nbr_sum.index_add_(0, edge_j, hard[edge_i] * edge_w[:, None])
                    nbr_w.index_add_(0, edge_i, edge_w)
                    nbr_w.index_add_(0, edge_j, edge_w)
                    mask = nbr_w > 0
                    bary = hard.clone()
                    bary[mask] = nbr_sum[mask] / nbr_w[mask, None]
                    grad += stage["graph_w"] * (hard - bary)

                grad += stage["seed_w"] * (hard - base_seed)
                grad += 0.18 * port_pull
                grad += self._repulsion_gradient(hard, widths, heights, movable, stage["repel_w"])
                grad[~movable] = 0.0

                scale = 1.0 + degree[:, None]
                hard[movable] -= stage["lr"] * grad[movable] / scale[movable]
                hard[:, 0].clamp_(half_w, benchmark.canvas_width - half_w)
                hard[:, 1].clamp_(half_h, benchmark.canvas_height - half_h)
                hard[fixed] = benchmark.macro_positions[:num_hard][fixed]

                if (step + 1) % 16 == 0 or step + 1 == stage["steps"]:
                    result[:num_hard] = hard
                    result = self._legalize_hard_macros(result, benchmark)
                    hard = result[:num_hard].clone()

        result[:num_hard] = hard
        return result

    def _legalize_hard_macros(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> torch.Tensor:
        """
        Remove hard-macro overlaps while staying close to the current layout.

        The algorithm places larger macros first and searches outward in rings
        until it finds the nearest legal position.
        """
        result = placement.clone()
        num_hard = benchmark.num_hard_macros
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        sizes = benchmark.macro_sizes[:num_hard]

        order = list(range(num_hard))
        order.sort(key=lambda i: -(sizes[i, 0] * sizes[i, 1]).item())

        placed = []
        for idx in order:
            if benchmark.macro_fixed[idx]:
                placed.append(idx)
                continue

            current = result[idx].clone()
            legal = self._find_nearest_legal_position(
                idx=idx,
                target=current,
                placed=placed,
                placement=result,
                benchmark=benchmark,
            )
            result[idx] = legal
            placed.append(idx)

        fixed_mask = benchmark.macro_fixed
        result[fixed_mask] = benchmark.macro_positions[fixed_mask]
        return result

    def _update_soft_macros(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> torch.Tensor:
        """
        Optional soft-macro update stage.

        Leaving soft macros unchanged is a reasonable baseline. Better
        submissions usually reposition soft macros after hard macro moves.
        """
        if benchmark.num_soft_macros == 0:
            return placement
        return self._seed_soft_macros(placement.clone(), benchmark, self._make_generator(benchmark))

    def _build_hard_graph(self, benchmark: Benchmark):
        num_hard = benchmark.num_hard_macros
        edge_dict = {}
        degree = torch.zeros(num_hard, dtype=torch.float32)
        for net_nodes, weight in zip(benchmark.net_nodes, benchmark.net_weights):
            hard_nodes = torch.unique(net_nodes[net_nodes < num_hard])
            if hard_nodes.numel() < 2 or hard_nodes.numel() > 28:
                continue
            pair_w = float(weight) / max(int(hard_nodes.numel()) - 1, 1)
            hard_list = hard_nodes.tolist()
            for i in range(len(hard_list)):
                a = int(hard_list[i])
                for j in range(i + 1, len(hard_list)):
                    b = int(hard_list[j])
                    key = (a, b) if a < b else (b, a)
                    edge_dict[key] = edge_dict.get(key, 0.0) + pair_w

        if not edge_dict:
            return degree, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0)

        edge_i = torch.tensor([a for a, _ in edge_dict.keys()], dtype=torch.long)
        edge_j = torch.tensor([b for _, b in edge_dict.keys()], dtype=torch.long)
        edge_w = torch.tensor([w for w in edge_dict.values()], dtype=torch.float32)
        degree.index_add_(0, edge_i, edge_w)
        degree.index_add_(0, edge_j, edge_w)
        return degree, edge_i, edge_j, edge_w

    def _build_port_pulls(self, benchmark: Benchmark) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        pulls = torch.zeros((num_hard, 2), dtype=benchmark.macro_positions.dtype)
        weights = torch.zeros(num_hard, dtype=benchmark.macro_positions.dtype)
        port_start = benchmark.num_macros
        for net_nodes, weight in zip(benchmark.net_nodes, benchmark.net_weights):
            hard_nodes = torch.unique(net_nodes[net_nodes < num_hard])
            port_nodes = net_nodes[net_nodes >= port_start] - port_start
            if hard_nodes.numel() == 0 or port_nodes.numel() == 0:
                continue
            port_center = benchmark.port_positions[port_nodes].mean(dim=0)
            w = float(weight)
            for node in hard_nodes.tolist():
                pulls[node] += w * (benchmark.macro_positions[node] - port_center)
                weights[node] += w
        mask = weights > 0
        pulls[mask] = pulls[mask] / weights[mask, None]
        return pulls

    def _repulsion_gradient(
        self,
        pos: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        movable: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        num_hard = pos.shape[0]
        grad = torch.zeros_like(pos)
        for i in range(num_hard):
            if not movable[i]:
                continue
            for j in range(i + 1, num_hard):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                min_sep_x = 0.55 * (widths[i] + widths[j])
                min_sep_y = 0.55 * (heights[i] + heights[j])
                if abs(dx) > min_sep_x or abs(dy) > min_sep_y:
                    continue
                sx = max(0.0, float(min_sep_x - abs(dx)))
                sy = max(0.0, float(min_sep_y - abs(dy)))
                if sx <= 0.0 and sy <= 0.0:
                    continue
                vec = torch.tensor(
                    [
                        dx if abs(float(dx)) > 1.0e-6 else (1.0 if i < j else -1.0),
                        dy if abs(float(dy)) > 1.0e-6 else (1.0 if i % 2 == 0 else -1.0),
                    ],
                    dtype=pos.dtype,
                )
                norm = torch.norm(vec, p=2)
                if float(norm) < 1.0e-9:
                    continue
                push = weight * (sx + sy) * (vec / norm)
                grad[i] += push
                grad[j] -= push
        grad[~movable] = 0.0
        return grad

    def _make_generator(self, benchmark: Benchmark) -> torch.Generator:
        env_seed = os.environ.get("PARTCL_RANDOM_INIT_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = self.seed
        else:
            seed = self.seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(benchmark.name))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        return gen

    def _find_nearest_legal_position(
        self,
        idx: int,
        target: torch.Tensor,
        placed: list[int],
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> torch.Tensor:
        width = benchmark.macro_sizes[idx, 0].item()
        height = benchmark.macro_sizes[idx, 1].item()

        x_min = width / 2
        x_max = benchmark.canvas_width - width / 2
        y_min = height / 2
        y_max = benchmark.canvas_height - height / 2

        clamped = torch.tensor(
            [
                min(max(target[0].item(), x_min), x_max),
                min(max(target[1].item(), y_min), y_max),
            ],
            dtype=placement.dtype,
        )
        if self._is_legal_hard_position(idx, clamped, placed, placement, benchmark):
            return clamped

        step = max(width, height) * 0.25 + self.gap
        best = None
        best_dist2 = float("inf")

        for ring in range(1, self.search_rings + 1):
            found = False
            for dx_ring in range(-ring, ring + 1):
                for dy_ring in range(-ring, ring + 1):
                    if abs(dx_ring) != ring and abs(dy_ring) != ring:
                        continue

                    candidate = torch.tensor(
                        [
                            min(max(target[0].item() + dx_ring * step, x_min), x_max),
                            min(max(target[1].item() + dy_ring * step, y_min), y_max),
                        ],
                        dtype=placement.dtype,
                    )
                    if not self._is_legal_hard_position(
                        idx,
                        candidate,
                        placed,
                        placement,
                        benchmark,
                    ):
                        continue

                    dist2 = torch.sum((candidate - clamped) ** 2).item()
                    if dist2 < best_dist2:
                        best = candidate
                        best_dist2 = dist2
                        found = True
            if found:
                return best

        return self._fallback_row_pack_position(idx, placed, placement, benchmark)

    def _is_legal_hard_position(
        self,
        idx: int,
        candidate: torch.Tensor,
        placed: list[int],
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> bool:
        width = benchmark.macro_sizes[idx, 0].item()
        height = benchmark.macro_sizes[idx, 1].item()
        x = candidate[0].item()
        y = candidate[1].item()

        if x - width / 2 < 0 or x + width / 2 > benchmark.canvas_width:
            return False
        if y - height / 2 < 0 or y + height / 2 > benchmark.canvas_height:
            return False

        for other in placed:
            if other >= benchmark.num_hard_macros:
                continue

            other_x = placement[other, 0].item()
            other_y = placement[other, 1].item()
            other_w = benchmark.macro_sizes[other, 0].item()
            other_h = benchmark.macro_sizes[other, 1].item()

            overlap_x = abs(x - other_x) < (width + other_w) / 2 + self.gap
            overlap_y = abs(y - other_y) < (height + other_h) / 2 + self.gap
            if overlap_x and overlap_y:
                return False

        return True

    def _fallback_row_pack_position(
        self,
        idx: int,
        placed: list[int],
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> torch.Tensor:
        """
        Last-resort legal placement if local search fails.
        """
        width = benchmark.macro_sizes[idx, 0].item()
        height = benchmark.macro_sizes[idx, 1].item()

        x = width / 2
        y = height / 2
        row_height = 0.0

        hard_placed = [p for p in placed if p < benchmark.num_hard_macros]
        if hard_placed:
            extents = []
            for other in hard_placed:
                other_x = placement[other, 0].item()
                other_y = placement[other, 1].item()
                other_w = benchmark.macro_sizes[other, 0].item()
                other_h = benchmark.macro_sizes[other, 1].item()
                extents.append(
                    (
                        other_x - other_w / 2,
                        other_x + other_w / 2,
                        other_y - other_h / 2,
                        other_y + other_h / 2,
                    )
                )
            extents.sort(key=lambda e: (e[2], e[0]))

            cursor_x = 0.0
            cursor_y = 0.0
            row_height = 0.0
            for left, right, bottom, top in extents:
                if bottom > cursor_y + self.gap:
                    break
                if left > cursor_x + width + self.gap:
                    break
                cursor_x = max(cursor_x, right + self.gap)
                row_height = max(row_height, top - cursor_y)
                if cursor_x + width > benchmark.canvas_width:
                    cursor_x = 0.0
                    cursor_y += row_height + self.gap
                    row_height = 0.0

            x = min(max(cursor_x + width / 2, width / 2), benchmark.canvas_width - width / 2)
            y = min(max(cursor_y + height / 2, height / 2), benchmark.canvas_height - height / 2)

        return torch.tensor([x, y], dtype=placement.dtype)
