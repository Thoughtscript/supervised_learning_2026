import fit, leakage, drift, skew  # noqa: E401  (run as: python concepts/run_all.py)

for m in (leakage, fit, drift, skew):
    m.run()
