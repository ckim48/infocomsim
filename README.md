# RECD Simulation — Road-Aware Encoder Caching and Dissemination

Trace-driven SUMO simulation for the INFOCOM paper *"Road-Aware Caching and
Dissemination for Multimodal Federated Learning in Vehicular Networks."*

## Layout

```
scenario/          SF downtown OSM extract + SUMO network + trips (3 densities)
sim/
  gen_traces.py    SUMO/TraCI mobility trace recorder (positions, segments,
                   per-segment traffic stats, segment transitions)
  road_graph.py    segment-level road graph, Markov mobility kernel (Eq. 8),
                   reachability (Eq. 16)
  gat_predictor.py GAT next-direction predictor (Sec. III-A)
  comm.py          V2V PHY: rate, contact prediction (Eqs. 11-13), LET,
                   ground-truth contact check
  fl.py            multimodal FL: 3 synthetic modalities over FashionMNIST,
                   CNN encoders + local fusion, Eq. (20) aggregation
  methods.py       RECD + benchmarks (DFL-Gossip, LRU-Random, Mobility-Greedy)
  engine.py        round loop, PHY simulation, caching, metrics
  run_experiment.py CLI for a single run
experiments/
  run_all.py       54-job campaign (main x3 seeds + 5 sweeps + ablation)
  plots.py         IEEE-style PDF figures from results/runs/*.json
  fig_motivation.py motivation figure (V2V contact concentration)
results/           traces, run JSONs, logs
figures/           output PDFs for the paper
performance_evaluation.tex  drop-in LaTeX section
```

## Reproduce

```bash
export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo
# 1. scenario (already generated): netconvert + randomTrips at 3 densities
# 2. traces
python3 sim/gen_traces.py --density med --n 60 --seed 1
# 3. one run
python3 sim/run_experiment.py --method RECD --rounds 55
# 4. everything
python3 experiments/run_all.py
python3 experiments/plots.py
python3 experiments/fig_motivation.py
```

## Notes

- Requires SUMO 1.20 (uses the traci bundled in `$SUMO_HOME/tools` —
  the pip traci 1.27 is incompatible with the 1.20 binary).
- Transmission success is evaluated against the *recorded* trace: a
  transfer succeeds only if the pair actually stays in range for the whole
  transmission time, so contact-prediction quality directly matters.
- Motivation figure panel (b) (nuScenes objects per drive, day vs. night)
  requires the nuScenes metadata and your own analysis; panels (a)(c)(d)
  are generated (a: published nuScenes statistics; c, d: SUMO contacts).
