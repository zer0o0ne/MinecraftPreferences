---
name: read-paper
description: Extract exact equations, algorithms and implementation details from a paper (arXiv LaTeX source or HTML) and its official code. Use when a spec cites a paper.
---
# Read a paper

Work in the scratchpad; only what ends up in the spec enters the repo.

1. arXiv `<id>`: formulas from the LaTeX source — `curl -sL https://arxiv.org/e-print/<id> -o src` (tar.gz or single .gz: check with `file`); equation numbers from `https://arxiv.org/html/<id>` or `https://ar5iv.labs.arxiv.org/html/<id>`. Other papers: WebFetch the page or PDF.
2. Read the main text, algorithm boxes and appendix (hyperparameters, implementation details); they often disagree. Record each disagreement.
3. Official code, if any: record repo and commit; find the lines implementing each equation; record differences from the paper.
4. Cite in the spec as `[S1] Eq. (5)`, `[S1] Alg. 1 line 7`, `[S1-code] path:line`. Any disagreement between the user's text, the paper and the code is an open point for the user.
