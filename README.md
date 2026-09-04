# Chowk

> Stub — the full README (results tables, plots, scores) is written at build
> step 12 and every number in it is produced by a script in this repo
> (`verify.py`). No figures are hand-written. See `AGENT_BRIEF.md` §11.

Chowk routes each payment to the downstream processor that maximises expected
net revenue, and — unlike a supervised model trained on confounded logs — spends
a small, budgeted slice of randomly-routed traffic so its estimates stay honest
in segments the old policy never explored.

**⚠️ All data in this project is synthetic**, produced by a simulator with known
latent ground truth. No real payment data is used.

Build in progress. Run `python verify.py` to reproduce everything.
