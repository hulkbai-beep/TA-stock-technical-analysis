# Stock Technical Analysis Dashboard

A standalone Streamlit dashboard for US stocks, with a Simplified Chinese UI,
conditional technical assessments with a weighted indicator score and market
context, a walk-forward backtest, and multi-method support/resistance:
confirmed swing clusters, Estimated Volume Profile, Anchored VWAP and Fibonacci
confluence.

See [setup and usage](technical_analysis/README.md),
[assessment rules](technical_analysis/docs/technical-assessment-rules.md),
[support/resistance methods](technical_analysis/docs/support-resistance.md), and
[performance measurements](technical_analysis/docs/performance.md).

Copy `technical_analysis/.env.example` to `technical_analysis/.env` and configure
your own FMP access locally. Real environment files, credentials, generated
artifacts, and virtual environments are excluded from Git.

The rules are technical heuristics; run `scripts/backtest.py` before relying on
them. No profitability or prediction accuracy is claimed.
