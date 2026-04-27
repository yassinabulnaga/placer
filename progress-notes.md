# PartCL Progress Notes

## Current Status

- Latest checked best `ibm01` result from `submissions/partcl/runner.py`: `proxy=1.1176`
- Winner on `ibm01`: `outline_soft_search`
- Breakdown: `wl=0.0705`, `den=0.9180`, `cong=1.1761`
- Important comparison points:
  - `outline_legal`: `1.1293`
  - lean soft-search pool: `1.1233`
  - RePlAce proxy baseline from repo README: `0.9976`
- Runtime remains a real issue: exact-gated `ibm01` soft-search runs are still heavy and are no longer a fast debug loop

## Main Conclusion So Far

We successfully turned the intended report-style flow into a real, runnable pipeline:

- real `mtkahypar` partitioning is active
- real DREAMPlace is installed and used
- macro-mode DREAMPlace can now return valid placements in the integrated flow

The important shift is that the best verified path is no longer plain `outline_legal`. A stronger exact-gated soft search on top of the outline basin now beats it on `ibm01`, and also improves isolated outline baselines on `ibm02` and `ibm03`.

## What The Deep Research Got Right

The second deep-research report was directionally correct on the most important points:

- density and congestion are the real bottleneck on `ibm01`, not wirelength
- `outline_legal` is the right incumbent basin to anchor on
- soft optimization is a first-order lever
- local hard-macro repair should be bounded and secondary
- exact scoring must decide survivors; surrogates are only for ranking

The experiments now support those claims pretty strongly.

## Report-Guided Ideas That Worked

### 1. Outline-anchored exact-gated soft search

This is the highest-value idea from the report that actually paid off in our code.

Verified progression on `ibm01`:

- `outline_legal`: `1.1293`
- early `outline_soft_search`: `1.1263`
- congestion/corridor-aware search: `1.1236`
- latest richer exact-gated search: `1.1218`
- second exact-gated pass from that winner: `1.1211`
- third exact-gated pass from that winner: `1.1191`
- fourth exact-gated pass from that winner: no further gain, stayed `1.1191`
- tail local polish after the 3-pass winner: `1.1182`
- pair-separation local polish after that: `1.1180`
- a third local polish cycle improved it again to `1.1179`
- a coarse soft-region rebalance before local polish improved it again to `1.1176`
- a more aggressive multi-soft region rebalance was accepted and changed the basin, but the final exact result was `1.1177`, so it is a near-tie regression rather than a new best

This family also improved isolated outline baselines on:

- `ibm02`: `1.6132 -> 1.5165`
- `ibm03`: `1.4036 -> 1.3969`

### 2. Congestion-aware and corridor-aware soft candidates inside the search beam

These did not usually win as standalone one-shot candidates, but they did help inside the exact-gated search beam.

Examples on `ibm01`:

- standalone congestion refiner: around `1.1294`
- standalone `channel_then_cong`: around `1.1278`
- richer beam with corridor-aware candidates: best verified `1.1218`

So the report’s “channel shaping / congestion-first” guidance seems right, but only when embedded in a broader exact-scored search.

### 3. Real KaHyPar / Mt-KaHyPar integration

This matched the report’s seed-diversity recommendation and is now functioning correctly.

What it did well:

- gave real structural seed diversity
- made the intended report-style flow real
- removed uncertainty about whether partitioning was active

What it did not do:

- it did not by itself close the quality gap on `ibm01`

### 4. Macro-mode DREAMPlace as a real candidate family

This was a major engineering success relative to the report goals.

What worked:

- macro mode is now valid and integrated
- placements are real and exact-scoreable
- soft repair on top of macro mode mattered a lot

What did not happen:

- even after repair, macrobase families still did not beat the best outline-soft path on `ibm01`

## Report-Guided Ideas That Did Not Work Well

### 1. Native RePlAce-lite / local hard-macro repair on `ibm01`

The report ranked hotspot-driven hard repair highly, but in our implementation this was not the lever on `ibm01`.

What we saw:

- `outline_replite` became runnable
- it accepted zero hard-macro moves on `ibm01`
- discrete one-macro repair still accepted zero profitable moves
- final score stayed around the `outline_soft_local` basin instead of improving beyond it

Conclusion:

- on `ibm01`, local hard-macro repair is not currently paying for its complexity

### 2. PyTorch SoftOpt as the main answer, so far

The report strongly suggested a PyTorch SoftOpt path. We built it and tried multiple variants.

Verified `ibm01` outcomes:

- earlier torch SoftOpt: `1.1306`
- congestion-heavier retune: `1.1349`
- latest two-phase density-then-congestion tail-focused SoftOpt: `1.1388`

Conclusion:

- the SoftOpt direction is conceptually aligned with the report
- but our current implementation is still worse than the exact-gated heuristic soft-search family

### 3. Broad search expansion as a default strategy

The report advocates a strong experiment program, but online budget matters.

What we learned:

- wider `4 rounds / beam 4` soft search was too expensive for default use
- a leaner default pool hurt quality (`1.1218 -> 1.1233`)
- exact `ibm01` runs are now slow enough that runtime itself has become a tuning constraint

Conclusion:

- the richer search family helps quality
- but runtime must be managed carefully and broadening the search blindly is not free

### 4. One-shot whitespace shaping as a standalone winner

The report’s region-level whitespace redistribution idea is promising, and we started implementing it.

Current result on `ibm01` for standalone whitespace shaping:

- `1.1309`

What that means:

- it improved both density and congestion versus plain `outline_legal`
- but it is not a standalone winner yet
- the real question is whether it helps inside the full search beam

At the time of this note, that full-beam evaluation is still the open question.

### 5. Exact-evaluator speed work started, but the first wins are modest

We started following the “Incremental CD” style lesson that evaluation speed compounds.

What we added:

- exact-cost memoization in `runner.py`
- candidate runtime logging
- exact-cache hit/miss accounting
- a narrow `PARTCL_PROFILE_MODE=outline` path for faster profiling
- a lightweight adaptive operator selector inside `outline_soft_search`

What the first instrumented `ibm01` profile showed:

- winner still `outline_soft_search = 1.1218`
- outline-only runtime still about `538-559s`
- exact-cache hit rate only about `46.3%`
- exact compute time still around `372-390s`

We also tried:

- quantized placement keys for the cache
- beam-level candidate dedup before exact scoring

Result:

- neither materially changed hit rate or total exact compute time on the narrowed `ibm01` run

What did help:

- adaptive operator-family bias inside the soft-search beam
- a second exact-gated soft-search pass on top of the first-pass winner for small cases
- local exact soft polishing after the 3-pass winner:
  - soft coordinate descent
  - density-focused coordinate descent
  - a small hotspot-based pair-swap pass
  - a small hotspot-based pair-separation pass
  - repeating that local cycle one more time on small cases
  - a coarse soft-region rebalance step before local polishing

Measured on narrowed `ibm01` outline-only profile:

- before adaptive beam:
  - runtime about `538-559s`
  - exact compute about `372-390s`
- after adaptive beam:
  - runtime about `396s`
  - exact compute about `292s`
  - score still `1.1218`

Latest full narrowed `ibm01` check after the stronger rebalance:

- `outline_soft_search = 1.1177`
- runtime about `2224s`
- exact compute about `2030s`
- the aggressive rebalance was accepted after pass 3:
  - `1.1191 -> 1.1189`
- then the local tail polished it further:
  - `1.1189 -> 1.1177`

Conclusion:

- exact scoring is definitely the main runtime sink
- adaptive operator pruning is a real speed win
- simple memoization tweaks alone were not enough
- a more incremental evaluator may still matter later, but better beam control is already paying off now
- the newer, more aggressive global rebalance direction appears real, but the current version is still a tiny regression versus the `1.1176` best and also makes the narrowed `ibm01` path even more expensive

Cross-benchmark check of the adaptive beam on narrowed outline mode:

- `ibm01`: kept `1.1218`, runtime about `396s`
- `ibm02`: kept `1.5165`, runtime about `671s`
- `ibm03`: improved from earlier `1.3969` to `1.3902`, runtime about `904s`
- `ibm04`: improved from outline baseline `1.3734` to `1.3615`, runtime about `1259s`

This is the strongest recent sign that the adaptive beam is a real improvement path, not an `ibm01`-only trick.

Follow-up direct `ibm01` probe on top of that:

- first pass `outline_soft_search`: `1.1218`
- second exact-gated soft-search pass starting from that winner: `1.1211`
- third exact-gated soft-search pass starting from that winner: `1.1191`
- fourth exact-gated soft-search pass: no further gain, stayed `1.1191`
- tail local polish after the 3-pass winner improved it again to `1.1182`
- pair-separation on top of that improved it again to `1.1180`
- a third local cycle improved it again to `1.1179`
- a coarse soft-region rebalance plus the local tail improved it again to `1.1176`

This was a real improvement, not just search noise:

- wirelength improved slightly
- density worsened slightly
- congestion improved enough to win overall

So on small cases, the current best signal is that the search still has real headroom if it is reapplied from the best exact-scored basin, then globally rebalanced at the region level, and only then polished locally with exact soft moves. The first clear plateau in repeated full passes shows up by pass 4, but local polish still helped after pass 3, and the biggest recent change in direction was adding a coarse rebalance step before the local tail.

### 6. DREAMPlace was wasting runtime on GPU retries in this environment

The local DREAMPlace install was CPU-only, but the adapter was still trying GPU first.

What we changed:

- the adapter now auto-retries on CPU if it sees:
  - `CANNOT enable GPU without CUDA compiled`

What this means:

- we no longer waste repeated candidate attempts on the same avoidable GPU assertion failure
- this is a clean runtime fix, even though it does not improve placement quality by itself

## What We Implemented

### 1. KaHyPar / Mt-KaHyPar integration

We updated `runner.py` to:

- export the hard-macro hypergraph
- try `mtkahypar`, then `kahypar`, then CLI KaHyPar
- fall back only if those are unavailable

What we learned:

- the partitioning backend now works correctly
- `ibm01` logs consistently show:
  - `partitioner=k=4:mtkahypar, k=8:mtkahypar, k=16:mtkahypar`
- activating real KaHyPar helped correctness and diversity, but by itself did not close the proxy gap

### 2. DREAMPlace adapter and installation

We added `submissions/partcl/dreamplace_adapter.py` and wired `runner.py` to use it.

We also:

- cloned and built DREAMPlace locally
- fixed missing/generated config usage by pointing the runner at `dreamplace/install`
- patched DREAMPlace for NumPy compatibility
- patched DREAMPlace `PlaceDB.py` to classify multi-row movable nodes as macros for this challenge flow

What we learned:

- `dreamplace.configure` errors were caused by using the source tree instead of the built install tree
- once built, DREAMPlace ran, but the adapter semantics needed substantial work before outputs were usable

### 3. Adapter export fixes

We debugged the synthetic LEF/DEF bridge extensively.

Key fixes:

- corrected site/row/grid export so DEF rows match the benchmark geometry
- rejected sentinel DEF coordinates like `-2147483648`
- rejected obviously invalid or collapsed macro placements
- added automatic retry from broken macro mode into valid flat mode
- added config control for:
  - `allow_flat_fallback`
  - `macro_place_flag`
  - `use_bb`
  - `gift_init_flag`

What we learned:

- the giant bogus exact scores came from parsing invalid `.gp.def` outputs where macros were effectively unplaced
- once sentinel outputs were rejected, the pipeline stopped scoring nonsense placements
- flat-mode retry made DREAMPlace usable again, but flat mode was weaker than the outline baseline on the IBM tests we checked

### 4. Macro-mode DREAMPlace repair

This was the most important technical breakthrough.

We found macro mode was failing because:

- soft macros were being exported in a way that confused DREAMPlace macro placement
- DREAMPlace’s internal macro classifier was too strict for this benchmark representation

Fixes:

- changed macro-mode export so soft macros can be omitted as movable blocks and represented through virtual connectivity anchors
- patched DREAMPlace `PlaceDB.py` so multi-row movables are treated as macros in this flow
- narrowed to a short, stable macro-mode recipe:
  - `allow_flat_fallback=0`
  - `macro_place_flag=1`
  - `use_bb=1`
  - `omit_soft_macros=1`
  - `single_stage=1`
  - no fillers
  - no area-adjust passes
  - conservative density / gamma / learning rate settings

What we learned:

- macro mode is no longer fundamentally broken
- it now succeeds inside the real portfolio with:
  - `[partcl:dreamplace] success ... mode=macro`
- the blocker is now placement quality, not backend bring-up

### 5. Portfolio instrumentation

We added logs for:

- backend usage
- candidate start/end
- surrogate scores
- exact shortlist scores
- refinement scores

What we learned:

- this made it possible to tell whether a candidate really used:
  - DREAMPlace vs fallback
  - macro mode vs flat mode
  - KaHyPar vs fallback partitioning
- without this logging, we would have continued tuning broken paths by accident

## Candidate Families We Tried

### Outline-based family

Added and evaluated:

- `outline_legal`
- `outline_micro`
- `outline_soft`
- `outline_soft_local`
- `outline_soft_search`
- `outline_incre`
- `outline_replite`

What we learned:

- `outline_legal` is an extremely strong starting basin on `ibm01`
- small hard-macro moves away from it usually hurt density too much
- `outline_soft_local` is slightly worse than outline on `ibm01`, but did help on at least one other IBM case during spot checks
- `outline_soft_search` is the first verified improvement over `outline_legal` on `ibm01`
- the first native `outline_replite` path was runnable, but accepted no hard-macro moves on `ibm01` and did not improve beyond `outline_soft_local`

### Analytical / flat DREAMPlace family

Added and tested:

- `initial_anchor`
- `outline_anchor`
- partition-seeded DREAMPlace candidates
- `softspread` variants
- experimental route / halo variants

What we learned:

- flat DREAMPlace works, but remains weaker than the outline baseline on `ibm01`
- some aggressive experimental variants crashed DREAMPlace with allocator corruption
- those dangerous variants were gated behind `PARTCL_ENABLE_EXPERIMENTAL_DREAMPLACE=1`

### Report-style macrobase family

Added and tested:

- `macrobase_4_boundary_cut_e5`
- `macrobase_4_boundary_cut_e5_softlocal`
- `macrobase_4_boundary_cut_e5_softsearch`
- stable nearby variants:
  - `_spread`
  - `_gentle`
  - and their `_softsearch` versions

What we learned:

- this is the first truly report-aligned base family that now works end-to-end
- hard macro placements from macro mode are real and exact-scoreable
- but they still do not beat `outline_legal` on `ibm01`

## Best Macrobase Results Seen On IBM01

These are the most important verified numbers from the macro-mode path:

- `macrobase_4_boundary_cut_e5`: `1.6283`
- `macrobase_4_boundary_cut_e5_softlocal`: `1.2352`
- `macrobase_4_boundary_cut_e5_softsearch`: improved over time
  - earlier best: `1.2097`
  - later targeted soft subset search: `1.2024`
  - latest widened run: `1.2293`
- `macrobase_4_boundary_cut_e5_spread_softsearch`: `1.2815`
- `macrobase_4_boundary_cut_e5_gentle_softsearch`: `1.3439`

Important note:

- the latest widened run exact-scored the spread and gentle variants because their surrogates were best in that run
- the earlier narrower runs gave the clearest verified best macrobase value: `1.2024`

## Most Important Lessons

### 1. The main issue on `ibm01` is not wirelength

Again and again, our optimized candidates had:

- better wirelength than `outline_legal`
- much worse density
- sometimes slightly better congestion, but not enough

That means the proxy loss is mainly a density and congestion story, not a wirelength story.

### 2. `outline_legal` is strong because it preserves an already good density structure

On `ibm01`, the initial placement plus minimum-displacement legalization seems to preserve whitespace and low-density regions better than our analytical flows.

Small hard-macro moves often:

- reduce wirelength
- cluster structure more tightly
- create worse hotspot bins

### 3. The soft-macro stage matters a lot

This was one of the clearest findings.

For macro-mode candidates:

- raw macrobase was poor
- replacing generic soft follow with selective, exact-gated soft repair helped a lot

Examples:

- `1.6283 -> 1.2352 -> 1.2024`

So the post-macro soft distribution is a first-order contributor to final quality.

For the outline basin, the same lesson now holds too:

- `outline_legal = 1.1293`
- `outline_soft_search` evolved over time to `1.1218`
- a second exact-gated pass from that winner improved `ibm01` again to `1.1211`
- a third exact-gated pass improved it again to `1.1191`
- a tail local exact soft polish improved it again to `1.1182`
- a later pair-separation local polish improved it again to `1.1180`
- a later third local cycle improved it again to `1.1179`
- a later coarse soft-region rebalance improved it again to `1.1176`

And isolated soft-search probes also improved:

- `ibm02`: `1.6132 -> 1.5165`
- `ibm03`: `1.4036 -> 1.3969`

### 4. Macro-mode is now a quality problem, not an integration problem

Earlier, macro mode was failing semantically.

Now:

- it runs
- it produces valid placements
- it survives the full portfolio

So the remaining work is real optimization work, not plumbing.

### 5. More soft-search rounds gave diminishing returns

We turned the soft search into a two-round exact-gated beam.

What we learned:

- round 1 helped
- round 2 helped only a little
- this suggests the remaining gap is not simply “one more soft pass”

### 6. Nearby stable macrobase variants did not win

We added:

- a more spread-out stable macrobase
- a gentler macro move schedule

What we learned:

- both were stable
- neither beat the current best macrobase-softsearch trajectory
- neither beat `outline_legal`

### 7. The first native RePlAce-lite attempt did not unlock local hard-macro wins on `ibm01`

We built `outline_replite` as a native incremental analytical path in challenge coordinates.

Instrumentation showed:

- start from `outline_soft_local`: `1.1388`
- no accepted hard-macro rounds
- no improvement from final polish
- final remained `1.1388`

We then replaced the vector-field hard move with a discrete one-macro-at-a-time repair loop.

Result:

- still zero accepted hard-macro rounds
- still no improvement beyond `1.1388`

So for `ibm01`, the local hard-macro repair path is not currently the high-ROI direction.

## Current Best Understanding Of The Gap

On `ibm01`, the picture now looks like this:

- the best basin is still the outline-derived one
- stronger soft-only optimization on top of that basin is now the best verified direction
- the latest exact-gated soft search with corridor-aware candidates improved `ibm01` again from `1.1236` to `1.1218`
- a direct second-pass search from that winner improved it again to `1.1211`
- a real integrated third-pass run improved it again to `1.1191`
- a later integrated tail local polish improved it again to `1.1182`
- a later pair-separation local polish improved it again to `1.1180`
- a later third local cycle improved it again to `1.1179`
- a later coarse soft-region rebalance improved it again to `1.1176`
- that new best point had roughly:
  - `wl=0.0705`
  - `den=0.9180`
  - `cong=1.1761`
- macro-mode DREAMPlace is valid and useful for study, but still weaker than the best outline-soft path
- local hard-macro repair around the outline basin is not currently finding any exact-improving moves

Additional congestion experiments after that:

- one-shot `channel` and `channel_then_cong` candidates helped a little but did not beat the new best
- `ibm01`:
  - `channel_subset`: `1.1293`
  - `channel_then_cong`: `1.1278`
- standalone whitespace shaping: `1.1309`
- `ibm02` and `ibm03` showed these corridor candidates can be mildly useful, but they still did not beat the existing `outline_soft_search` results there either
- a wider `4 rounds / beam 4` soft search looked too expensive to keep as default, so it is now gated behind `PARTCL_WIDE_SOFT_SEARCH=1`
- a leaner default soft-search pool was tested, but it regressed `ibm01` from `1.1218` to `1.1233`, so the richer pool remains default and the leaner variant is now optional behind `PARTCL_LEAN_SOFT_POOL=1`
- the newer two-phase tail-focused PyTorch SoftOpt was also worse than the best heuristic search, landing at `1.1388`

So the remaining gap likely comes from one or more of:

- the soft-stage search still not being strong enough to close the last density/congestion gap on `ibm01`
- the outline basin may need more expressive soft-only optimization rather than local hard repair
- the synthetic DREAMPlace representation still not matching the challenge well enough for macro-generated bases to catch up

## What Did Not Work Well

- aggressive experimental DREAMPlace density/routability variants
- broad route-biased and halo-heavy configs in this synthetic adapter
- repeated tiny hard-macro edits around `outline_legal` on `ibm01`
- generic soft follow on top of macro-mode placements
- assuming a lower surrogate always meant a better exact proxy

## Recommended Next Steps

If we continue from here, the most promising next directions are:

1. Invest further in the outline-anchored soft-only exact-gated path.
This remains the highest-ROI family and the only one that has clearly beaten `outline_legal` on `ibm01`.

The newest signal is that multiple exact-gated passes from the first-pass winner are worth baking into the default small-benchmark path; three passes are now the production default for those cases because pass 4 plateaued on `ibm01`. After that, a small exact local polish on soft macros is the new highest-value final stage.

2. Keep density/congestion-oriented candidate families only if they improve the beam, not because they look good standalone.
This is the main lesson from congestion, corridor, and whitespace shaping so far.

3. Validate the stronger outline-soft family more broadly across the IBM suite.
We have good signals on `ibm01`, `ibm02`, and `ibm03`, but still need broader confirmation.

4. Keep macro-mode DREAMPlace and PyTorch SoftOpt as side branches, not the main path for now.
Both are now valuable experimental families, but neither is the quality leader today.

## Files Most Affected

- [runner.py](/home/robertliu/Documents/macro-place-challenge-2026/submissions/partcl/runner.py)
- [dreamplace_adapter.py](/home/robertliu/Documents/macro-place-challenge-2026/submissions/partcl/dreamplace_adapter.py)
- [PlaceDB.py](/home/robertliu/Documents/macro-place-challenge-2026/submissions/partcl/dreamplace/install/dreamplace/PlaceDB.py)
- [PlaceDB.py](/home/robertliu/Documents/macro-place-challenge-2026/submissions/partcl/dreamplace/dreamplace/PlaceDB.py)
- [deep-research-report.md](/home/robertliu/Documents/macro-place-challenge-2026/submissions/partcl/deep-research-report.md)

## Update: 2026-04-24

This update captures the later branch where we stopped treating `outline_soft_search`
as the only serious family, added a true hard-first basin-change path, then pushed
that path until it plateaued under an adaptive exact-gated outer-cycle controller.

### 1. We added a real `rebalance_first` family

The key change in `runner.py` was to stop only polishing the same outline-soft basin
and instead introduce a new direct-placement family:

- `rebalance_first_search`

Its structure is:

1. start from the exact-gated outline baseline
2. apply a hard-macro rebalance first
3. optionally try corridor/whitespace states and route-aware states
4. only then run the soft-search tail

This was the first branch in this phase that produced a real basin change rather than
just another tiny tail tweak.

### 2. Initial recovery: back below `1.15`, then into the `1.13x` range

The first important recovered path on `ibm01` was:

- outline baseline after early hard rebalance: about `1.1505`
- full narrowed run on that path: `1.1464`

Then the stronger hard-first family improved further:

- targeted `rebalance_first` check: `1.1351`
- later full `ibm01`-style check in the same branch: about `1.1361`

This confirmed that the new family was real, but it also showed that the remaining
gap was still overwhelmingly density and congestion, not wirelength.

### 3. Corridor/whitespace-first was useful, but weaker than hard-first on `ibm01`

We also added:

- `corridor_first_search`

This family explicitly starts from corridor / whitespace / route-space shaping before
the soft tail.

On `ibm01`, the exact comparison was:

- `OUTLINE = 1.1596`
- `REB = 1.1351`
- `COR = 1.1391`

So the corridor-first family was real and helpful, but still weaker than the
hard-first family on this benchmark. The most useful lesson was not “make corridor
the main path”; it was “keep corridor / whitespace states alive inside the winning
hard-first family.”

### 4. We promoted corridor / whitespace states inside the hard-first family

The next changes did not create a new top-level family. Instead, they improved the
working one:

- corridor / whitespace states are now tried inside `rebalance_first`
- shortlist selection now keeps density+congestion specialists, not only the best
  surrogate / proxy candidates
- the hard-first loop now considers:
  - `channel`
  - `whitespace`
  - `channel_then_route`
  - `whitespace_then_channel`
  - `channel_then_whitespace`
  - `corridor_then_rebalance`

This did improve `ibm01` modestly:

- from about `1.1351`
- to about `1.1344`

That was useful, but it also made the next lesson clearer: by this point, the branch
was no longer being held back by missing one obvious corridor operator.

### 5. Cross-benchmark checks showed the controller should be adaptive

A reduced exact comparison across `ibm01`, `ibm02`, and `ibm03` gave a much more
important result than any single `ibm01` tweak.

Observed exact results:

- `ibm01 OUT = 1.159563`
- `ibm01 REB = 1.134384`
- `ibm02 OUT = 1.538321`
- `ibm02 REB = 1.451757`
- `ibm03 OUT = 1.408870`
- `ibm03 REB = 1.338128`

Just as important as the final numbers was the path shape:

- `ibm01` liked hard rebalance plus whitespace / corridor shaping
- `ibm02` responded strongly to route-heavy follow-through
- `ibm03` liked the local density/congestion tail after rebalance, not route-heavy
  follow-through

So the correct lesson was:

- do not force all benchmarks through the same refinement policy
- the family controller should choose among route-heavy, whitespace-heavy, and
  local-tail behavior using early exact signals

### 6. We made the hard-first controller adaptive by benchmark behavior

Inside `_rebalance_first_refine(...)` we added explicit mode selection:

- `route-heavy`
- `whitespace-heavy`
- `local-tail`

The mode is chosen from early exact signals:

- if route relief gives a real exact gain, switch into `route-heavy`
- otherwise, if corridor / whitespace stages improve density+congestion enough,
  choose `whitespace-heavy`
- otherwise, use `local-tail`

This matched the cross-benchmark behavior we observed:

- `ibm01` classified as `whitespace-heavy`
- `ibm02` classified as `route-heavy`
- `ibm03` behaved like a `local-tail` / rebalance-driven case

We also added route-followup states when route relief was real:

- `route_then_channel`
- `route_then_whitespace`
- `route_then_channel_then_route`

On `ibm02`, this improved the early route-heavy basin itself:

- old route-heavy pre-tail basin: about `1.4988`
- improved route-heavy pre-tail basin: about `1.4951`

The final `ibm02` branch also nudged slightly lower in the late tail:

- from about `1.4518`
- to about `1.4517`

So the adaptive controller was doing the right thing, but `ibm01` still remained the
hardest case.

### 7. The first really meaningful `ibm01` jump came from repeated outer exact-gated cycles

The biggest breakthrough in this later phase came from one simple observation:

- repeatedly polishing the same `1.13x` basin was flattening
- but re-entering the hard-first pipeline from its own improved basin still
  produced new exact gains

We therefore changed `rebalance_first` so it could run multiple exact-gated outer
cycles:

1. hard rebalance / adaptive basin selection
2. soft-search tail
3. if still improving, re-enter from that new basin and repeat

This mattered much more than any single earlier local tweak.

Exact `ibm01` progression:

- one outer cycle: about `1.1344`
- two outer cycles: `1.1273`
- three outer cycles: `1.1206`
- four outer cycles: `1.1122`
- five outer cycles: `1.1119`

The detailed cycle progression looked like this:

- cycle 1 repolish: `1.1344`
- cycle 2 hard step: `1.1327`
- cycle 2 repolish: `1.1273`
- cycle 3 hard step: `1.1263`
- cycle 3 late hard-balance: `1.1217`
- cycle 3 repolish: `1.1206`
- cycle 4 hard step: `1.1184`
- cycle 4 late hard-balance: `1.1138`
- cycle 4 repolish: `1.1122`
- cycle 5 repolish: `1.1119`

Final exact `ibm01` number from this branch:

- `proxy = 1.111869`
- `wl = 0.0726`
- `den = 0.9202`
- `cong = 1.1583`

This is a real improvement over the earlier `1.13x` / `1.12x` basin, but still not
close to the desired `< 1.0` region.

### 8. We replaced fixed outer-cycle counts with adaptive stop logic

At first, the repeated outer-cycle experiments were being run manually by setting
`PARTCL_REBALANCE_OUTER_CYCLES` by hand.

We then made that loop adaptive:

- small cases now default to up to 6 outer cycles
- outer-cycle defaults now key off `num_hard_macros`, not total macro count
- the loop logs exact per-cycle gains
- it stops automatically when the exact gain falls below a threshold

Observed `ibm01` adaptive gain sequence:

- outer 2 gain: `0.0071`
- outer 3 gain: `0.0067`
- outer 4 gain: `0.0084`
- outer 5 gain: `0.0004`

The controller correctly stopped on cycle 5 because:

- `0.0004 < 0.0025`

This is important because it tells us the current repeated-cycle hard-first path is
not being cut off prematurely anymore. The plateau near `1.1119` is probably real
for this branch.

### 9. What this means

The most important updated conclusions are:

- the late hard-first / outer-cycle path is real and now much stronger than before
- the controller should remain adaptive across benchmarks
- `ibm01` still improves materially under repeated exact-gated cycles, but this
  branch now appears to flatten around `1.11x`
- that means more of the same local / outer polishing is no longer the highest-ROI
  direction

At this point, the next likely big gain is not “another outer cycle.” It is:

- improving the earlier basin-building stage
- especially the analytical / DREAMPlace-style starting placement quality
- then using the stronger later controller on top of that better basin

That is why the next engineering target should shift upstream:

- isolate and reproduce the stronger earlier DREAMPlace-style path the user
  mentioned
- compare its seed / export / parameter semantics against our current macro-mode
  / analytical candidate families
- pull the stronger early-placement choices into the main pipeline instead of
  only adding more late-stage refinement

## Update: 2026-04-25 - Full DREAMPlace basin broke below 1.0 on `ibm01`

We found the major upstream-basin issue the notes were pointing toward.

The old DREAMPlace adapter only returned hard-macro coordinates, so even when
DREAMPlace placed the full exported design, the runner discarded DREAMPlace's
soft-macro positions and rebuilt them with our local soft follower. That meant
we were not actually preserving the analytical/DREAMPlace soft-macro basin.

### 1. Adapter / runner changes

Implemented a full-placement DREAMPlace path:

- `dreamplace_adapter.py` can now return all macro coordinates when
  `return_all_macros=1`
- `runner.py` accepts `[num_macros, 2]` DREAMPlace results for that mode
- added `dreamplace_full_outline_*` candidate families
- added `PARTCL_PROFILE_MODE=dreamplace_basin` for fast raw-basin sweeps
- added `PARTCL_PROFILE_MODE=dreamplace_basin_rebalance` for slower downstream
  rebalance tests

We also installed missing DREAMPlace Python dependencies into the local
environment:

- `shapely`
- `cairocffi`
- `scipy`
- `torch_optimizer`
- `ncg_optimizer==0.2.2`

After those dependencies were present, DREAMPlace ran successfully through the
PartCL adapter in CPU mode.

### 2. Raw DREAMPlace full-placement results on `ibm01`

Focused command used:

```text
PARTCL_PROFILE_MODE=dreamplace_basin uv run evaluate submissions/partcl/runner.py -b ibm01
```

Important exact results:

- `outline_legal`: `1.1596`
- `dreamplace_full_outline_flat_rebalance` raw basin: `1.0392`
  - `wl = 0.074`
  - `den = 0.719`
  - `cong = 1.211`
- `dreamplace_full_outline_loose`: `1.0301`
  - `wl = 0.074`
  - `den = 0.615`
  - `cong = 1.298`
- `dreamplace_full_outline_dense_sharp`: `0.9795`
- `dreamplace_full_outline_balanced`: `0.9583`
  - `wl = 0.072`
  - `den = 0.601`
  - `cong = 1.171`

This is the first verified local PartCL result in this branch below `1.0` on
`ibm01`.

### 3. What failed or was demoted

Two experimental DREAMPlace variants were tested but removed from the active
default/profile sweep:

- `dreamplace_full_outline_route` failed inside DREAMPlace with
  `TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'`
- `dreamplace_full_outline_twostage` produced sentinel DEF coordinates for many
  macros and was rejected by the adapter

The heavy `dreamplace_basin_rebalance` path was also tried, including a
one-cycle cap with `PARTCL_REBALANCE_OUTER_CYCLES=1`, but it was too slow for
the debug loop. The immediate win came from improving the starting basin itself,
not from downstream rebalance.

### 4. Updated conclusion

The earlier concern was correct: we were relying too much on late-stage repair
and not enough on the analytical starting placement.

The new best known `ibm01` path is:

```text
full DREAMPlace macro placement
target_density = 0.84
density_weight = 2.8e-5
gamma = 5.2
lr = 0.0065
num_bins_scale = 0.75
stop_overflow = 0.08
proxy = 0.9583
```

Next best engineering targets:

- run the `dreamplace_full_outline_balanced` path on `ibm02` and `ibm03`
- generate a visualization for the new `ibm01` below-1.0 placement
- only then revisit downstream refinement, preferably with a much lighter
  exact-gated polish than the current full `rebalance_first` path

## Update: 2026-04-25 - Below-0.9 push from the DREAMPlace basin

We tried to push the new `ibm01` DREAMPlace basin from the `0.95x` region down
below `0.9`.

The important arithmetic:

- around `wl = 0.072`, reaching `proxy < 0.9` needs `density + congestion < 1.656`
- the best raw balanced result had about `density + congestion = 1.772`
- so the next target is roughly another `0.11` to `0.12` combined D+C reduction

### New best area from this round

The best new direction was a lightweight whitespace polish after the balanced
full-DREAMPlace placement.

Best observed post-polish band:

- historical first `dreamplace_full_outline_balanced_whitepolish`: `0.9406`
  - `wl = 0.072`
  - `den = 0.578`
  - `cong = 1.158`
- current reproducible local variants are around `0.9419` to `0.9421`
  - `dreamplace_full_outline_balanced_white_local_soft`: `0.9419`
  - `dreamplace_full_outline_balanced_white_local`: `0.9421`

This is a real improvement over raw `dreamplace_full_outline_balanced = 0.9583`,
but it still does not reach `< 0.9`.

### Variants tested that did not break through

Raw DREAMPlace parameter sweeps:

- `balanced_g6`: `1.0037`
- `mid_dense`: `0.9769`
- `mid_loose`: `0.9789`
- `zero_noise`: `0.9679`
- `balanced_bins`: `0.9533`
- `balanced_long`: `0.9618`
- `balanced_random`: `1.0069`
- `balanced_random_white`: `1.0084`

Soft polish variants:

- `balanced_routepolish`: `0.9579`
- `balanced_channelpolish`: `1.0022`
- `balanced_route_channel`: `0.9931`
- `balanced_white2`: `0.9641`
- `balanced_white3`: `1.0050`
- `balanced_white_route`: `0.9601`
- `balanced_white_channel`: `1.0165`
- `balanced_white_fine`: `0.9597`
- `balanced_white_strong`: `1.0078`
- `balanced_white_local_push`: `0.9523`
- `balanced_white_local_fine`: `1.0039`
- `balanced_white_local_soft`: `0.9419`
- `dense_sharp_white`: `0.9847`
- `loose_white`: `0.9783`
- `flat_white`: `1.0244`
- `balanced_white_exactsoft`: `0.9600`

Native DREAMPlace routability:

- route-aware DREAMPlace originally crashed because synthetic LEF/DEF designs did
  not populate per-layer routing capacity arrays
- patched installed DREAMPlace `PlaceDB.py` so synthetic designs create
  one-layer `unit_horizontal_capacities` and `unit_vertical_capacities`
- after the patch, route-aware DREAMPlace ran, but quality was poor:
  `dreamplace_full_outline_route_diag = 1.2471`

### What this means

The current local floor for this family appears to be around `0.94`.

The next likely route to `< 0.9` is probably not another small whitespace-pass
variant. Better next bets:

- inspect the `0.94` visualization and identify the remaining congestion shape
- add a more targeted hard+soft corridor opening step for the specific remaining
  hotspot, not generic channel/route polish
- try cross-benchmark validation before overfitting `ibm01` further
- consider exporting richer routing/pin data if we want DREAMPlace routability
  to be meaningful on this synthetic LEF/DEF representation

## Update: 2026-04-25 - Best-candidate-only full-suite attempt

We attempted a full IBM run using the current best reproducible candidate:

```text
PARTCL_PROFILE_MODE=dreamplace_basin_deep
PARTCL_ONLY_CANDIDATES=dreamplace_full_outline_balanced_white_local_soft
uv run evaluate submissions/partcl/runner.py --all
```

Completed results before stopping:

- `ibm01`: `0.9522`
  - `wl = 0.072`
  - `den = 0.602`
  - `cong = 1.158`
- `ibm02`: `1.4444`
  - `wl = 0.075`
  - `den = 0.686`
  - `cong = 2.053`
- `ibm03`: `1.1987`
  - `wl = 0.084`
  - `den = 0.542`
  - `cong = 1.688`
- `ibm04`: `1.2575`
  - `wl = 0.080`
  - `den = 0.596`
  - `cong = 1.758`
- `ibm06`: `1.7022`
  - `wl = 0.075`
  - `den = 0.539`
  - `cong = 2.714`
- `ibm07`: `1.2913`
  - `wl = 0.073`
  - `den = 0.527`
  - `cong = 1.909`
- `ibm08`: `1.4007`
  - `wl = 0.080`
  - `den = 0.597`
  - `cong = 2.045`
- `ibm09`: `0.9381`
  - `wl = 0.062`
  - `den = 0.524`
  - `cong = 1.227`

The run then moved to `ibm10`, but it did not emit a result after more than an
hour of total runtime and was stopped manually. The Python evaluator process was
still using about `99%` CPU, so this looked like an impractically expensive exact
loop rather than an idle hang.

Conclusion:

- this candidate is useful on `ibm01` and `ibm09`
- it is not a universal replacement for the existing selector
- the main weakness on the other benchmarks is high congestion despite good
  density
- future production logic should treat this as a benchmark-selective candidate,
  not the default path for every IBM case

## Update: 2026-04-26 - ibm10 hang diagnosis and fix

Focused ibm10 run:

```text
PYTHONUNBUFFERED=1
PARTCL_PROFILE_MODE=dreamplace_basin_deep
PARTCL_ONLY_CANDIDATES=dreamplace_full_outline_balanced_white_local_soft
uv run evaluate submissions/partcl/runner.py -b ibm10
```

Root causes found:

- `PARTCL_ONLY_CANDIDATES` was applied too late, after building expensive
  partition/portfolio candidates that would later be discarded.
- focused DREAMPlace runs still used the heavy outline pre-portfolio and the
  legacy random-start initializer, both unnecessary for the selected candidate.
- graph-cache construction expanded soft-neighbor links quadratically for
  high-degree nets; ibm10 has large enough nets that this burned CPU before
  candidate execution.

Fixes added in `runner.py`:

- parse `PARTCL_ONLY_CANDIDATES` early and skip the remaining portfolio build
  when all requested candidates are already present
- skip generic post-refine for focused candidate-only runs
- add a fast focused DREAMPlace init path for large benchmarks, using legalized
  reference coordinates instead of the heavy outline/random-start setup
- cap high-degree soft-neighbor expansion with `PARTCL_SOFT_NEIGHBOR_CAP`
  defaulting to `96`
- flush candidate-start/init logging so future hangs expose their phase

Result after the fix:

- `ibm10`: `1.1717`
  - `wl = 0.055`
  - `den = 0.566`
  - `cong = 1.667`
  - `VALID`
  - runtime: `76.44s`

Conclusion:

- the ibm10 execution problem is resolved
- the quality issue remains congestion-driven, but it is now measurable instead
  of hidden behind a setup-time hang

## Update: 2026-04-26 - Full-suite focused DREAMPlace run after ibm10 fix

Command:

```text
PYTHONUNBUFFERED=1
PARTCL_PROFILE_MODE=dreamplace_basin_deep
PARTCL_ONLY_CANDIDATES=dreamplace_full_outline_balanced_white_local_soft
uv run evaluate submissions/partcl/runner.py --all
```

Additional fixes before the final run:

- fast focused init is now gated by size, not benchmark name:
  `num_hard_macros >= PARTCL_FAST_FOCUSED_MIN_HARD`, default `380`
- this preserves the stronger outline-seeded path for smaller cases such as
  `ibm01`, while still avoiding the ibm10 setup hang
- added a conditional hard-overlap repair tail after hard legalization
- the repair only runs if a real hard overlap remains, so it fixes edge cases
  like `ibm12` without perturbing already-legal placements like `ibm01`

Final full-suite result:

- average proxy: `1.3034`
- total overlaps: `0`
- total runtime: `2850.10s`

Per-benchmark results:

- `ibm01`: `0.9522`
- `ibm02`: `1.4444`
- `ibm03`: `1.1987`
- `ibm04`: `1.2575`
- `ibm06`: `1.7022`
- `ibm07`: `1.2913`
- `ibm08`: `1.4007`
- `ibm09`: `0.9381`
- `ibm10`: `1.1717`
- `ibm11`: `0.9745`
- `ibm12`: `1.3449`
- `ibm13`: `1.0737`
- `ibm14`: `1.3774`
- `ibm15`: `1.3569`
- `ibm16`: `1.3367`
- `ibm17`: `1.6855`
- `ibm18`: `1.6509`

Takeaways:

- the focused DREAMPlace candidate is valid across the full suite now
- it beats RePlAce on most cases but still loses on `ibm06` and `ibm17`
- the main remaining weakness is congestion on several mid/large benchmarks
