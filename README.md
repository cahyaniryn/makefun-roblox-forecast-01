# Roblox Up-and-Coming Forecast

This offline forecaster estimates the next 72 hours of Roblox's Up-and-Coming
chart from the supplied snapshots. It fits rank-change moments by starting-rank
band, applies a Student-t diffusion with AR(1) rank momentum and a 28-day tenure
exit hazard, and assigns part of each place's probability mass to unnamed
newcomers. A restrained player-growth mixture accounts for breakout games.

The program uses only the Python standard library and the frozen input files.
It makes no network requests and emits deterministic JSON at
`forecast.json` in the output directory.
The output contains per-game probabilities for each of the top ten places and
an ordered `top10` estimate; its probability mass is averaged across forecast
horizons rather than assuming the latest chart order persists.

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

By default, the grader input and output locations are `/input` and `/output`.
