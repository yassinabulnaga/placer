# PartCL Macro Placement: Next-Step Plan After Latest Progress

**Purpose:** This plan translates the latest progress notes into a concrete next engineering roadmap.

> **Status note, 2026-04-25:** The original version of this file assumed
> `outline_soft_search = 1.1218` was still the active frontier. That is now
> superseded by the later `rebalance_first_search` work recorded in
> `progress-notes.md`. Older sections below are still useful as historical
> context, but the roadmap immediately below is the current one.

Current best verified `ibm01` result:

```text
family  = dreamplace_full_outline_balanced
proxy   = 0.9583
wl      = 0.072
density = 0.601
cong    = 1.171
```

The immediate goal is no longer "add more late outer cycles." The repeated
rebalance cycles are real, but they plateaued near `1.1119` on `ibm01`.
The breakthrough came from preserving the full DREAMPlace analytical basin,
including soft-macro coordinates, instead of keeping only hard-macro positions.

### Current superseding diagnosis

```text
current best ibm01 proxy = 0.9583
RePlAce ibm01 baseline   = 0.9976
gain versus RePlAce      = 0.0393
```

Current component sum:

```text
wl       = 0.072
density  = 0.601
cong     = 1.171
D + C    = 1.772
```

If wirelength stays near `0.072`, beating the RePlAce proxy requires:

```text
0.072 + 0.5 * (D + C) < 0.9976
D + C < 1.8512
current D+C margin = about 0.0792
```

That target is now beaten on `ibm01`. The next risk is generalization: the
full DREAMPlace basin must be checked on `ibm02` and `ibm03`, then integrated
with a lighter downstream polish.

### Immediate priority order, superseding older sections

```text
P0. Validate dreamplace_full_outline_balanced on ibm02 and ibm03.
P1. Generate visuals for the new ibm01 below-1.0 placement.
P2. Keep outline_soft_search / rebalance_first as fallback candidates.
P3. Design a lighter exact-gated polish for the DREAMPlace basin.
P4. Run component-frontier exact selection across ibm01/ibm02/ibm03.
P5. Promote a cross-benchmark selector only after exact evidence.
```

### DREAMPlace / analytical-basin checklist

Use this checklist before adding another round count or another local repair:

```text
1. Locate the exact command, config, seed, and output placement for the 0.90xx run.
2. Confirm it used the same exact proxy scorer and benchmark instance.
3. Compare fixed-node, hard-macro, soft-macro, and outline semantics.
4. Compare DREAMPlace export choices: omit_soft_macros, virtual anchors,
   macro flags, density target, density weights, bins, halos, fillers,
   routability flags, and single-stage versus multi-stage placement.
5. Convert the best DREAMPlace/analytical output into a PartCL candidate.
6. Run adaptive rebalance_first downstream from that candidate.
7. Validate exact score and zero hard overlaps on ibm01, then ibm02/ibm03.
```

### What to stop optimizing first

Do not spend the primary budget on additional late outer cycles from the same
`1.1119` basin unless a new upstream candidate gives them fresh room to work.
The current code should treat `rebalance_first_search` as the best downstream
refiner, not as proof that the initial placement problem is solved.

---

## Table of Contents

1. [Updated Diagnosis](#1-updated-diagnosis)
2. [What This Means Strategically](#2-what-this-means-strategically)
3. [Immediate Priority Order](#3-immediate-priority-order)
4. [Priority 0: Run a Broader Component-Level Suite](#4-priority-0-run-a-broader-component-level-suite)
5. [Priority 1: Make Runtime Predictable Before Expanding Search](#5-priority-1-make-runtime-predictable-before-expanding-search)
6. [Priority 2: Upgrade `outline_soft_search` Into an Adaptive Beam](#6-priority-2-upgrade-outline_soft_search-into-an-adaptive-beam)
7. [Priority 3: Keep Whitespace/Corridor Candidates Inside the Beam](#7-priority-3-keep-whitespacecorridor-candidates-inside-the-beam)
8. [Priority 4: Track a Density+Congestion Frontier, Not Just Proxy](#8-priority-4-track-a-densitycongestion-frontier-not-just-proxy)
9. [Priority 5: Build a Cross-Benchmark Family Selector](#9-priority-5-build-a-cross-benchmark-family-selector)
10. [What to Pause or Demote](#10-what-to-pause-or-demote)
11. [Concrete Experiment Matrix](#11-concrete-experiment-matrix)
12. [Implementation Pseudocode](#12-implementation-pseudocode)
13. [Runtime Budgeting Plan](#13-runtime-budgeting-plan)
14. [7-Day Roadmap](#14-7-day-roadmap)
15. [Final Decision Rules](#15-final-decision-rules)

---

## 1. Updated Diagnosis

The latest result is real progress: `outline_soft_search` has become the first verified family that consistently improves over `outline_legal` on `ibm01`, and it also improved outline baselines on `ibm02` and `ibm03`.

However, the `ibm01` gap to RePlAce is still large.

```text
current best ibm01 proxy = 1.1218
RePlAce ibm01 baseline   = 0.9976
gap                      = 0.1242
```

The current best breakdown is:

```text
wl       = 0.0704
density  = 0.9137
cong     = 1.1890
D + C    = 2.1027
```

If wirelength stays at `0.0704`, then to beat the RePlAce proxy of `0.9976`, the required density-plus-congestion sum is:

```text
0.0704 + 0.5 * (D + C) < 0.9976
D + C < 1.8544
```

So the remaining required reduction is:

```text
current D+C  = 2.1027
target D+C   = 1.8544
needed drop  = 0.2483
```

That is the central fact. The current soft-search improvement from `outline_legal` to `outline_soft_search` is useful, but it is not yet changing the density/congestion basin enough.

### Current improvement versus `outline_legal`

```text
outline_legal       = 1.1293
outline_soft_search = 1.1218
improvement         = 0.0075
```

The improvement is real, but closing the `ibm01` RePlAce gap requires a much larger density/congestion shift. This means the next step is not simply “add more soft-search rounds.” The next step is to make the search more targeted, more component-aware, and more runtime-efficient.

---

## 2. What This Means Strategically

The current evidence says:

1. **The outline basin is still the correct anchor.**  
   It preserves a density structure that your DREAMPlace/macrobase families still fail to reproduce on `ibm01`.

2. **Soft-macro optimization is the main live lever.**  
   It is the only approach that has clearly beaten `outline_legal` on `ibm01` so far.

3. **Generic local hard-macro repair should not be the default path.**  
   Both `outline_replite` and discrete one-macro repair accepted zero profitable hard-macro moves on `ibm01`.

4. **Macro-mode DREAMPlace is now a working side branch, not the quality leader.**  
   It is valuable because it produces valid candidate diversity, but it should not consume the main debug budget until it wins on at least one benchmark family.

5. **The search needs adaptive operator selection, not blind widening.**  
   Wider search helped quality but hurt runtime. Leaner search helped runtime but regressed quality. The correct next step is to make the beam smarter, not simply wider.

6. **Do not overfit only to `ibm01`.**  
   The fact that `ibm02` improved from `1.6132` to `1.5165` suggests the outline-soft family may be much more valuable on some benchmarks than on `ibm01`. You need a full-suite picture before deciding what to optimize next.

---

## 3. Immediate Priority Order

Use this order for the next implementation cycle:

```text
P0. Run a broader exact component suite.
P1. Add runtime controls, exact-score caching, and operator-level logs.
P2. Turn outline_soft_search into an adaptive exact-gated beam.
P3. Integrate whitespace/corridor operators inside the beam, not standalone.
P4. Track density+congestion Pareto candidates, not only proxy winners.
P5. Build a simple cross-benchmark selector.
P6. Keep macrobase, DREAMPlace, hard repair, and PyTorch SoftOpt as side branches.
```

The main principle:

> Keep the outline-soft path as the production path, and use everything else as candidate generators or diagnostic branches until they prove themselves by exact score.

---

## 4. Priority 0: Run a Broader Component-Level Suite

Before adding another major algorithm, run a controlled benchmark sweep. The current results are too `ibm01`-centric.

### Required benchmark set

Run at least:

```text
ibm01, ibm02, ibm03, ibm04, ibm06, ibm09, ibm11, ibm12, ibm17, ibm18
```

Then run all IBM benchmarks once the candidate list is stable.

### Candidate families to include

Start with a narrow but informative suite:

```text
outline_legal
outline_soft_search
outline_soft_search_lean
outline_soft_search_wide
channel_then_cong
whitespace_shape
whitespace_then_soft_search
macrobase_best_softsearch
pytorch_softopt_tail
```

If runtime is too high, split into two passes:

#### Pass A: fast/default candidates

```text
outline_legal
outline_soft_search
channel_then_cong
whitespace_shape
```

#### Pass B: expensive candidates only on promising cases

```text
outline_soft_search_wide
whitespace_then_soft_search
macrobase_best_softsearch
pytorch_softopt_tail
```

### What to record

For every benchmark and candidate, log:

```text
benchmark
candidate_name
family
proxy
wl
density
congestion
density_plus_congestion
runtime_seconds
overlap_count
hard_displacement_from_outline
soft_displacement_from_outline
num_exact_scores
num_surrogate_scores
operator_lineage
```

### Key output table

Produce this table after the sweep:

| Benchmark | RePlAce | Outline Legal | Best Current | Best Family | Δ vs Outline | Δ vs RePlAce | Runtime |
|---|---:|---:|---:|---|---:|---:|---:|
| ibm01 | 0.9976 | 1.1293 | 1.1218 | outline_soft_search | -0.0075 | +0.1242 | TBD |
| ibm02 | 1.8370 | 1.6132 | 1.5165 | outline_soft_search | -0.0967 | -0.3205 | TBD |
| ibm03 | 1.3222 | 1.4036 | 1.3969 | outline_soft_search | -0.0067 | +0.0747 | TBD |

This table will tell you whether you are trying to win by:

1. beating RePlAce on every benchmark,
2. beating RePlAce on average,
3. qualifying by proxy and then winning through NG45/OpenROAD,
4. or simply improving your own average enough to place well.

Right now, `ibm02` is the most encouraging signal. Do not ignore it.

---

## 5. Priority 1: Make Runtime Predictable Before Expanding Search

You noted that exact-gated `ibm01` soft-search is now heavy and no longer a fast debug loop. Fixing that is more urgent than adding more candidate families.

### Add exact-score caching

Hash the placement tensor after rounding to a small tolerance.

Recommended hash:

```python
rounded = torch.round(pos / 1e-4).to(torch.int64)
key = sha1(rounded.cpu().numpy().tobytes()).hexdigest()
```

Cache:

```python
{
    "proxy": float,
    "wl": float,
    "density": float,
    "congestion": float,
    "overlaps": int,
    "runtime": float,
}
```

This matters because beam search can generate near-duplicates.

### Add per-candidate timing

Every candidate should print a one-line summary:

```text
[partcl:candidate] name=... family=... exact=... wl=... den=... cong=... runtime=... parent=... operator=...
```

### Add early timeout budget

Give each benchmark a time budget and enforce it inside the runner:

```python
BENCH_BUDGET_SECONDS = 300      # debug target
SUBMISSION_BUDGET_SECONDS = 3000 # leave margin under 1 hour
```

When budget is low:

1. finish current exact score,
2. skip expensive side branches,
3. return best exact-scored candidate.

### Use progressive scoring

Use four score levels:

```text
L0: validity only
    bounds, hard overlap, NaN/sentinel rejection

L1: cheap surrogate
    local density/congestion approximation, no exact evaluator

L2: exact proxy
    true evaluator, only for shortlisted candidates

L3: full benchmark sweep / ORFS
    only for final families
```

Do not let L1 choose the winner. L1 only decides what reaches L2.

---

## 6. Priority 2: Upgrade `outline_soft_search` Into an Adaptive Beam

The current winner is `outline_soft_search`, but it is still hand-scheduled. The next upgrade should be an operator-credit beam search.

### Core idea

Each search round maintains a beam of exact-scored candidates. Operators propose new candidates. Exact scoring gives each operator a credit signal. Future rounds sample operators based on credit.

### Operator set

Start with these operators:

| Operator | Purpose | Current evidence |
|---|---|---|
| `soft_local` | baseline local soft repair | stable but not always enough |
| `congestion_refine` | reduce top congestion bins | standalone weak, useful inside beam |
| `corridor_open` | open routing corridors | useful as beam ingredient |
| `whitespace_shape` | redistribute soft whitespace | standalone not winning, may help beam |
| `density_tail_evict` | move soft macros out of top density bins | should target real bottleneck |
| `dc_balanced` | optimize density+congestion jointly | default exploitation operator |
| `random_micro_jitter` | diversity, tiny moves only | prevents beam collapse |
| `net_centroid_reanchor` | repair wirelength after spreading | prevents WL drift |

### Initial operator probabilities

```python
operator_probs = {
    "dc_balanced": 0.25,
    "density_tail_evict": 0.20,
    "congestion_refine": 0.15,
    "corridor_open": 0.15,
    "whitespace_shape": 0.10,
    "soft_local": 0.10,
    "net_centroid_reanchor": 0.03,
    "random_micro_jitter": 0.02,
}
```

### Update rule

After each exact-scored child:

```python
gain = parent.proxy - child.proxy
operator_score[op] = 0.8 * operator_score[op] + 0.2 * gain
```

Then update probabilities:

```python
p(op) = epsilon / N + (1 - epsilon) * softmax(operator_score / tau)
```

Recommended:

```python
epsilon = 0.15
tau = 0.005 to 0.015
```

### Beam structure

Keep multiple survivor buckets:

```text
bucket A: best exact proxy
bucket B: best density + congestion
bucket C: best congestion
bucket D: best density
bucket E: best diversity / displacement-limited candidate
```

Do not keep only the best proxy. Some candidates with slightly worse proxy but much better `D+C` may become winners after another operator.

---

## 7. Priority 3: Keep Whitespace/Corridor Candidates Inside the Beam

Standalone results:

```text
channel_then_cong   ≈ 1.1278
whitespace_shape    ≈ 1.1309
best soft beam      ≈ 1.1218
```

These numbers say:

> Corridor and whitespace shaping are not final candidates by themselves, but they may be useful as intermediate states.

So the next change should be:

```text
outline_legal
  -> whitespace/corridor proposal
  -> exact gate or component-frontier gate
  -> soft_search continuation
  -> exact gate
```

Not:

```text
outline_legal
  -> whitespace/corridor proposal
  -> final candidate
```

### Add these composite candidates

```text
outline_ws_then_soft_round1
outline_ws_then_soft_round2
outline_channel_then_soft_round1
outline_channel_then_soft_round2
outline_ws_channel_then_soft
outline_channel_ws_then_soft
```

### Acceptance condition for intermediate whitespace/corridor states

Allow a candidate to continue even if proxy is slightly worse, but only if it improves the right components.

Suggested rule:

```python
allow_continue = (
    child.proxy <= parent.proxy + 0.020
    and child.density + child.congestion <= parent.density + parent.congestion - 0.020
)
```

Or:

```python
allow_continue = (
    child.proxy <= parent.proxy + 0.030
    and child.congestion <= parent.congestion - 0.040
)
```

This prevents the beam from killing candidates that create routing space before wirelength repair.

---

## 8. Priority 4: Track a Density+Congestion Frontier, Not Just Proxy

The current best has excellent wirelength, but the gap to RePlAce is density/congestion. Therefore your candidate survivor logic should explicitly track `D+C`.

### Add derived metrics

For every exact-scored candidate:

```python
candidate.dc_sum = candidate.density + candidate.congestion
candidate.dc_proxy = 0.5 * candidate.dc_sum
candidate.wl_share = candidate.wirelength / candidate.proxy
candidate.dc_share = candidate.dc_proxy / candidate.proxy
```

On current best `ibm01`:

```text
D+C = 2.1027
0.5*(D+C) = 1.05135
WL = 0.0704
```

That means the proxy is overwhelmingly dominated by density/congestion.

### New survivor rule

```python
def should_keep_candidate(c, incumbent, frontier):
    if c.overlaps != 0:
        return False

    # Always keep exact proxy improvement.
    if c.proxy < incumbent.proxy - 1e-4:
        return True

    # Keep candidates that materially reduce D+C without blowing up proxy.
    if c.dc_sum < incumbent.dc_sum - 0.03 and c.proxy < incumbent.proxy + 0.03:
        return True

    # Keep congestion specialists.
    if c.congestion < incumbent.congestion - 0.05 and c.proxy < incumbent.proxy + 0.04:
        return True

    # Keep density specialists.
    if c.density < incumbent.density - 0.05 and c.proxy < incumbent.proxy + 0.04:
        return True

    # Keep Pareto frontier candidates.
    if is_component_pareto(c, frontier):
        return True

    return False
```

### Why this matters

If you only keep proxy-best candidates, the search may keep slightly improving wirelength while never discovering lower density/congestion basins. You need to deliberately keep candidates that reduce the bottleneck components, even before they become proxy winners.

---

## 9. Priority 5: Build a Cross-Benchmark Family Selector

The progress notes show different behavior across benchmarks:

```text
ibm01: outline_soft_search improves 1.1293 -> 1.1218
ibm02: outline_soft_search improves 1.6132 -> 1.5165
ibm03: outline_soft_search improves 1.4036 -> 1.3969
```

This is not uniform. It strongly suggests a family selector will beat a single fixed recipe.

### Benchmark features to compute

```python
features = {
    "num_hard_macros": ...,
    "num_soft_macros": ...,
    "num_nets": ...,
    "area_util": ...,
    "outline_proxy": ...,
    "outline_wl": ...,
    "outline_density": ...,
    "outline_congestion": ...,
    "outline_dc_sum": ...,
    "top_density_mean": ...,
    "top_congestion_mean": ...,
    "density_congestion_overlap": ...,
    "hard_macro_area_cv": ...,
    "soft_to_hard_area_ratio": ...,
}
```

### Simple selector rules first

Before training anything, use hand-coded rules:

```python
if outline_congestion > 1.3:
    enable_corridor_ops = True

if outline_density > 0.95:
    enable_density_tail_ops = True

if soft_to_hard_area_ratio is high:
    use_wider_soft_search = True

if macrobase_after_soft is within 8% of outline_soft_search:
    keep_macrobase_branch = True
else:
    skip_macrobase_branch = True
```

### Later: train a lightweight selector

Once you have 100+ candidate records:

- Use XGBoost, LightGBM, random forest, or even logistic regression.
- Target: which family wins per benchmark.
- Inputs: benchmark features + outline score components.
- Output: candidate family budget allocation.

Do not start with a complex model. The dataset is small.

---

## 10. What to Pause or Demote

### Pause hard-macro repair on `ibm01`

The evidence is strong:

```text
outline_replite: zero accepted hard moves
one-macro repair: zero accepted hard moves
```

Do not keep spending default runtime here on `ibm01`.

Re-enable hard repair only when:

```text
benchmark has high congestion
AND hotspot map overlaps hard macro blockages
AND soft-only search plateaus
AND candidate remains above RePlAce by large margin
```

### Demote macrobase on `ibm01`

Macrobase is now valid, but not competitive:

```text
best macrobase trajectory ≈ 1.2024
best outline-soft         ≈ 1.1218
```

Use macrobase as:

1. a cross-benchmark diversity source,
2. a debug branch for DREAMPlace representation,
3. a fallback if outline basin is poor on another benchmark.

Do not use it as the main path on `ibm01`.

### Demote PyTorch SoftOpt until it passes parity checks

Current PyTorch SoftOpt results are worse:

```text
earlier torch SoftOpt                        ≈ 1.1306
congestion-heavy retune                      ≈ 1.1349
two-phase density-then-congestion SoftOpt    ≈ 1.1388
best heuristic outline_soft_search           ≈ 1.1218
```

Before tuning weights again, check whether its maps match the evaluator.

Parity checks:

```text
1. Do PyTorch top-density bins match exact top-density bins?
2. Do PyTorch top-congestion bins match exact top-congestion bins?
3. Does a PyTorch-predicted improving move actually improve exact score?
4. Are gradients pushing soft macros into legal/useful whitespace?
```

If these fail, do not tune weights. Fix the surrogate.

### Keep DREAMPlace stable, not experimental

Aggressive DREAMPlace density/routability variants caused allocator crashes. Keep them behind:

```text
PARTCL_ENABLE_EXPERIMENTAL_DREAMPLACE=1
```

Default submission path should use only stable variants.

---

## 11. Concrete Experiment Matrix

### Experiment A: Full-suite outline-soft validation

**Goal:** Determine whether `outline_soft_search` is generally strong or only locally useful.

| Candidate | Required? | Notes |
|---|---:|---|
| `outline_legal` | yes | baseline |
| `outline_soft_search` | yes | current best family |
| `outline_soft_search_lean` | yes | runtime/quality tradeoff |
| `outline_soft_search_wide` | optional | gated expensive run |

Success criteria:

```text
average improvement over outline_legal >= 2%
OR at least 5 benchmarks improve by >= 3%
OR at least 2 benchmarks beat RePlAce
```

### Experiment B: Whitespace/corridor as intermediate states

**Goal:** Test whether these are useful inside the beam.

| Candidate | Description |
|---|---|
| `ws_only` | current standalone whitespace shaping |
| `ws_then_soft1` | whitespace then one soft round |
| `ws_then_soft2` | whitespace then two soft rounds |
| `channel_then_soft1` | corridor/channel then one soft round |
| `channel_ws_then_soft` | combined route-space proposal |

Success criteria:

```text
beats outline_soft_search on at least 2 of 5 test benchmarks
OR improves D+C by >= 0.05 while proxy stays within +0.02
```

### Experiment C: Adaptive beam versus static beam

**Goal:** See whether operator credit improves quality/runtime.

| Version | Rounds | Beam | Operator choice |
|---|---:|---:|---|
| static_default | current | current | fixed |
| adaptive_small | 2 | 3 | credit-based |
| adaptive_medium | 3 | 4 | credit-based |
| adaptive_runtime_cap | dynamic | dynamic | stop by budget |

Success criteria:

```text
same or better quality than current rich pool
AND lower exact-score count
AND lower runtime
```

### Experiment D: Density+congestion frontier retention

**Goal:** Avoid proxy-only myopia.

Compare:

```text
current exact-gated survivor policy
vs
proxy + D+C frontier policy
```

Success criteria:

```text
finds at least one candidate with D+C lower by >= 0.05
that later becomes a proxy winner after continuation
```

### Experiment E: Macrobase cross-benchmark value

**Goal:** Decide whether macrobase should stay in default portfolio.

Run on 5-10 benchmarks:

```text
macrobase_best_raw
macrobase_best_softlocal
macrobase_best_softsearch
outline_soft_search
```

Success criteria:

```text
macrobase wins at least 1 benchmark
OR macrobase is within 3% of outline-soft on at least 3 benchmarks
```

If not, keep macrobase out of the default runtime path.

---

## 12. Implementation Pseudocode

### Candidate object

```python
@dataclass
class Candidate:
    name: str
    family: str
    placement: torch.Tensor
    parent: str | None
    operator: str | None
    surrogate_score: float | None = None
    proxy: float | None = None
    wl: float | None = None
    density: float | None = None
    congestion: float | None = None
    dc_sum: float | None = None
    overlaps: int | None = None
    runtime_s: float | None = None
    metadata: dict = field(default_factory=dict)
```

### Exact scoring wrapper with cache

```python
def exact_score_cached(candidate, benchmark, plc, cache):
    key = placement_hash(candidate.placement)
    if key in cache:
        metrics = cache[key]
    else:
        t0 = time.time()
        metrics = exact_score(candidate.placement, benchmark, plc)
        metrics["runtime_s"] = time.time() - t0
        cache[key] = metrics

    candidate.proxy = metrics["proxy"]
    candidate.wl = metrics["wirelength"]
    candidate.density = metrics["density"]
    candidate.congestion = metrics["congestion"]
    candidate.dc_sum = candidate.density + candidate.congestion
    candidate.overlaps = metrics.get("overlaps", 0)
    candidate.runtime_s = metrics["runtime_s"]
    return candidate
```

### Adaptive beam

```python
def outline_adaptive_soft_beam(outline_candidate, benchmark, plc, budget_s):
    beam = [exact_score_cached(outline_candidate, benchmark, plc, cache)]
    frontier = ComponentFrontier()
    op_scores = defaultdict(float)
    op_counts = defaultdict(int)
    start = time.time()

    for round_idx in range(MAX_ROUNDS):
        if time.time() - start > budget_s:
            break

        children = []
        probs = operator_probabilities(op_scores, epsilon=0.15, tau=0.01)

        for parent in select_parents(beam, frontier):
            for op in sample_ops(probs, k=OPS_PER_PARENT):
                child_pos = apply_soft_operator(op, parent.placement, benchmark, plc)
                child = Candidate(
                    name=f"{parent.name}_{op}_r{round_idx}",
                    family="outline_adaptive_soft",
                    placement=child_pos,
                    parent=parent.name,
                    operator=op,
                )

                if not quick_valid(child):
                    continue

                child.surrogate_score = cheap_surrogate(child)
                children.append(child)

        shortlist = shortlist_by_surrogate_and_diversity(children, k=EXACT_PER_ROUND)

        for child in shortlist:
            child = exact_score_cached(child, benchmark, plc, cache)
            parent = find_candidate(child.parent)
            gain = parent.proxy - child.proxy
            op_scores[child.operator] = 0.8 * op_scores[child.operator] + 0.2 * gain
            op_counts[child.operator] += 1
            frontier.add(child)

        beam = select_survivors(
            old_beam=beam,
            new_candidates=shortlist,
            frontier=frontier,
            max_beam=BEAM_SIZE,
        )

    return min(beam + frontier.proxy_candidates(), key=lambda c: c.proxy)
```

### Component-aware survivor selection

```python
def select_survivors(old_beam, new_candidates, frontier, max_beam):
    pool = [c for c in old_beam + new_candidates if c.overlaps == 0]

    survivors = []

    # 1. Always keep best proxy candidates.
    survivors += sorted(pool, key=lambda c: c.proxy)[:max(1, max_beam // 3)]

    # 2. Keep best D+C candidates within a proxy margin.
    best_proxy = min(c.proxy for c in pool)
    dc_candidates = [c for c in pool if c.proxy <= best_proxy + 0.04]
    survivors += sorted(dc_candidates, key=lambda c: c.dc_sum)[:max(1, max_beam // 3)]

    # 3. Keep congestion specialists.
    survivors += sorted(dc_candidates, key=lambda c: c.congestion)[:1]

    # 4. Keep density specialists.
    survivors += sorted(dc_candidates, key=lambda c: c.density)[:1]

    # 5. Deduplicate and trim.
    survivors = dedupe_by_hash(survivors)
    survivors = sorted(survivors, key=lambda c: c.proxy)[:max_beam]
    return survivors
```

---

## 13. Runtime Budgeting Plan

The competition hard limit is 1 hour per benchmark, but the default algorithm should leave a large safety margin.

### Recommended budgets

```text
Debug loop target:        2-5 minutes per benchmark
Default submission path:  8-15 minutes per benchmark
Expensive gated path:     20-35 minutes per benchmark
Hard timeout safety:      50 minutes per benchmark
```

### Runtime allocation for default path

```text
outline_legal + exact score             5%
outline_soft adaptive beam             55%
whitespace/corridor composite tests     15%
macrobase/DREAMPlace side branch        10%
validation + final exact rescoring      10%
slack                                    5%
```

### Runtime allocation for expensive path

```text
outline adaptive beam                   45%
wide soft/corridor/whitespace beam      25%
macrobase diversity                     10%
PyTorch SoftOpt diagnostic branch        5%
validation + final exact rescoring      10%
slack                                    5%
```

### Early stop rules

Stop search early if:

```text
no exact improvement for 2 rounds
AND no D+C frontier improvement for 2 rounds
AND runtime > 60% of budget
```

Continue search even if proxy plateaus when:

```text
D+C frontier continues improving
AND proxy is within 0.03 of incumbent
```

---

## 14. 7-Day Roadmap

### Day 1: Full-suite measurement infrastructure

Deliverables:

```text
results.csv with component scores
candidate lineage logs
exact-score cache
benchmark summary table
```

Run:

```text
outline_legal
outline_soft_search
channel_then_cong
whitespace_shape
```

on at least 10 IBM benchmarks.

### Day 2: Runtime and caching cleanup

Deliverables:

```text
placement hash cache
per-candidate timing
budget-aware runner
operator lineage logs
```

Goal:

```text
make ibm01 debug runs predictable again
```

### Days 3-4: Adaptive beam implementation

Deliverables:

```text
operator registry
operator credit scoring
component-aware survivor buckets
D+C frontier retention
```

Compare:

```text
current rich soft pool
vs
adaptive beam small
vs
adaptive beam medium
```

### Day 5: Whitespace/corridor composite integration

Deliverables:

```text
ws_then_soft candidates
channel_then_soft candidates
component-frontier continuation rules
```

Goal:

```text
determine whether whitespace/corridor states help after continuation
```

### Day 6: Cross-benchmark selector v0

Deliverables:

```text
benchmark feature extraction
hand-coded selector rules
family budget selection
```

Goal:

```text
stop using the same budget on every benchmark
```

### Day 7: Decision gate

Generate:

```text
full table: current default vs adaptive default
full table: best per benchmark
runtime table
family winner histogram
```

Decision:

```text
promote adaptive outline-soft as default
or keep current rich pool and continue operator work
```

---

## 15. Final Decision Rules

### Promote a candidate family to default only if:

```text
average proxy improves across the suite
AND ibm01 does not regress by more than 0.003
AND runtime remains under the default budget
AND zero hard overlaps are maintained
```

### Keep a candidate family as optional only if:

```text
it wins at least one benchmark
OR it gives useful D+C frontier candidates
OR it is needed for diversity in the selector
```

### Remove or gate a candidate family if:

```text
it never wins exact score
AND it consumes more than 10% runtime
AND it does not generate useful frontier candidates
```

### Current recommended default path

Based on the progress notes, the best next default should be:

```text
1. outline_legal
2. outline_soft_search adaptive beam
3. whitespace/corridor candidates inside the beam
4. component-frontier exact survivor selection
5. optional macrobase side branch only when selector enables it
6. final exact-score validation and zero-overlap check
```

### Current recommended research side branches

Keep these outside the default runtime path:

```text
macro-mode DREAMPlace representation improvements
PyTorch SoftOpt parity/debugging
hard-macro repair / RePlAce-lite
aggressive DREAMPlace density/routability variants
```

---

## Bottom Line

The most important update is that the winning path has changed again:

> The main production strategy should now be stronger analytical/DREAMPlace
> starting basins, followed by adaptive `rebalance_first` downstream refinement
> and exact component-aware survivor selection.

Do not keep adding late outer cycles as the primary strategy from the current
`1.1119` basin. Those cycles helped, but the last measured outer gain collapsed
to about `0.0004`, so they are no longer the most promising way to close the
remaining `ibm01` gap.

The next breakthrough is most likely to come from:

1. reproducing or isolating the stronger DREAMPlace-style `0.90xx` path,
2. auditing adapter/export/scoring semantics against that path,
3. using that better initial basin as input to `rebalance_first_search`,
4. keeping `outline_soft_search` as a fallback and diversity candidate,
5. and selecting by exact component frontier across multiple IBM benchmarks.

The next code target should be upstream basin quality: find why the stronger
DREAMPlace-style method can reach much lower proxy, reproduce that path in this
repo, then hand its placement to the downstream refiner that is already working.
