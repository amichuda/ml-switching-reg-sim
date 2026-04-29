# Formatting Report

Rendered: 2026-04-29
PDF: `paper-writer/draft/paper.pdf`
Pages: 28
Engine: lualatex (TeX Live 2022)
Source: `paper-writer/draft/paper.qmd` (rebuilt for v3 with BCHS framing, Michuda voice, in-document tables via `\input`, and timing/convergence moved to Appendix A)
Bibliography: `paper-writer/draft/references.bib` (entries assembled from BCHS and Michuda JDE bibliographies; author should verify in reference manager before submission).

## JAE Compliance Status

| Check | Status | Notes |
|---|---|---|
| Page count <= 35 | OK | 28 pages |
| 12pt font | OK | YAML `classoption: [12pt, letterpaper]` |
| Double spacing | OK | YAML `linestretch: 2` |
| 1-inch margins | OK | YAML `geometry: margin=1in` |
| Abstract <= 100 words | Slightly long | ~145 words. JAE summary limit is 100; the author should trim before submission. |
| No citations in abstract | Marginal | Cites Battaglia et al. 2024 by reference. Acceptable for working-paper version; may need rewording for JAE submission. |
| H1 bold all-caps numbered | OK | "1. INTRODUCTION", etc. |
| H2 bold sentence case numbered | OK | "2.1. Identification", "5.2. Naive hard-classification baseline" |
| Tables embedded via \input | OK | All four appendix tables now use `\input{../results/tables/table_*.tex}` rather than file-path mentions. |
| Tables: caption above | OK | `captionsetup[table]{position=top}` |
| Figures: caption below | OK | `captionsetup[figure]{position=bottom}` |
| Equation numbering | N/A | Display equations are unnumbered in this version; consistent with typical economics methods-paper style. Switch to `\begin{equation}` if the journal requires numbered equations. |
| References: author-year inline prose | Working paper-acceptable | bib file shipped but not yet wired into natbib. See "Remaining Issues" below. |
| Page numbers: centered footer | OK | fancyhdr block in YAML |
| No severe Overfull hbox | OK | 0 overfull warnings in `paper.log` |
| Fatal errors | OK | 0 lines beginning with `!` in `paper.log` |

## Issues Fixed in This Pass

1. **Reframed contribution from "switching regression" to generated-regressor / causal-ML.** The introduction now positions the paper as extending [Battaglia et al., 2024] to settings where the latent variable indexes a regression slope rather than entering linearly. The literature placement paragraph adds Pagan (1984), Stock-Watson (2002), Bai-Ng (2006), Bernanke-Boivin-Eliasz (2005), Egami et al. (2023), Imbens (2000), Westreich-Lessler-Funk (2010), Lee-Lessler-Stuart (2010), Goller (2020), Hu (2008), Cameron-Gelbach-Miller (2008), Kline-Santos (2012), Athey-Imbens (2019), Mullainathan-Spiess (2017), Chernozhukov et al. (2018), Bandiera et al. (2020), Hoberg-Phillips (2016), Baker-Bloom-Davis (2016).

2. **Adopted Michuda voice.** All "we" rewritten to "I" (Michuda is sole author). Introduction now follows the JDE structure: numbered contributions, direct framing of methodological position. Section titles (e.g. "Naive hard-classification baseline", "RMSE, bias, and the IRLS-MLE near-identity") are descriptive rather than abstract.

3. **Adopted BCHS notation.** Latent variable is $\theta_i$ (categorical $r_i$), classifier output is $\mathbf{p}_i$, posterior weights are $\mathbf{w}_i$, parameter vector is $\psi$, ordinary covariates are $\mathbf{q}_{it}$, observed regression covariates are $\boldsymbol{\xi}_i = (\theta_i, \mathbf{q}_i)$ in spirit (with the categorical specialization noted). Outcome equation now includes a $\mathbf{q}_{it}^{\top}\delta$ term to keep BCHS structure even though it is not exercised in the simulation.

4. **Tables are now in-document, not path-references.** Each appendix table is embedded via `\input{../results/tables/<file>.tex}` inside a `\begin{table}` float with caption and note. Body prose no longer says "see `paper-writer/results/...`"; it just refers to "Table A2" etc.

5. **Timing and convergence moved to Appendix A.** Section 5.4 from v2 became Appendix A. Section 5 now ends with the RMSE/bias/IRLS-MLE near-identity discussion, which is the correct flow given the new contribution emphasis.

6. **References file (`references.bib`).** Created with verified entries for BCHS (verified via S2 API: arXiv:2402.15585, paper ID `af48ff2d576ce9c2349207ff18c7308911fea53f`) and Egami et al. 2023 (verified: arXiv:2306.04746, NeurIPS), plus carefully composed entries for the well-known classics (Pagan 1984, Quandt 1972, Heckman 1979, Imbens 2000, Hu 2008, etc.). The bib file ships with a header comment instructing the author to verify each entry in their reference manager.

7. **LuaLaTeX/unicode-math conflict resolved.** Initial render failed on `\boldsymbol{\delta}` and `\bm{\delta}`, both of which trip a `\mitdelta`-related `unicode-math` issue under Quarto's lualatex pipeline. Fix: drop boldface on $\delta$ and $\psi$ (boldness is not load-bearing for these symbols in this paper).

## Remaining Issues (require author attention)

1. **Bibliography integration.** `references.bib` ships but is not yet wired through natbib. Body citations are still inline name-year prose ("[Battaglia et al., 2024]"). To switch to formal citations:
   - Add to YAML: `bibliography: references.bib`, `biblio-style: jae`, `cite-method: natbib`, `natbiboptions: round`.
   - Download `jae.bst` (`cd draft && curl -O https://mirrors.ctan.org/biblio/bibtex/contrib/economic/jae.bst`).
   - Replace each `[Author, Year]` in the body with `\citep{key}` or `\citet{key}` as appropriate.
   - Replace the manual References section with `# References {.unnumbered}\n\n::: {#refs}\n:::`.
   - **Verify each `references.bib` entry against your reference manager before submitting**; the file was assembled from BCHS and Michuda JDE bibliographies plus standard sources, but I have not independently verified every page number and volume number.

2. **Abstract length.** The current abstract is ~145 words. Trim to 100 words for JAE submission. Suggested cut: the last sentence ("The exercise extends the generated-regressor framework...") can be folded into the introduction rather than the abstract.

3. **Equation numbering.** Display equations are currently unnumbered. If the journal requires numbered equations, replace `$$...$$` blocks with `\begin{equation}...\end{equation}`. The most-cited equation is the unit-level mixture likelihood in Section 2; numbering it would let the prose say "the likelihood in (2)" rather than "the likelihood above".

4. **`\bm`/`\boldsymbol` on Greek letters.** Removed $\bm{\delta}$ and $\bm{\psi}$ in favor of plain $\delta$ and $\psi$ to dodge a lualatex/unicode-math conflict. If the author prefers boldface for vectors and parameter blocks, switch the LaTeX engine to pdflatex (`pdf-engine: pdflatex` in YAML) and reinstate `\bm{\cdot}`. Pdflatex also avoids the "Improper alphabetic constant" issue.

5. **Date.** The PDF currently shows the render date via `date: today`. Pin to `date: "April 29, 2026"` (or similar) before circulating, so re-renders do not change the displayed date.

6. **Affiliation.** Set to "Department of Economics, Swarthmore College" with email `amichud1@swarthmore.edu`. Verify and correct.

## Submission Readiness

**Ready for coauthor / mentor circulation.** The PDF compiles cleanly, the contribution is reframed in the BCHS / generated-regressor language, the voice matches the Michuda JDE paper, tables are embedded via `\input`, and timing/convergence are moved to Appendix A. Two items must be completed before journal submission: (i) wire up natbib + `jae.bst` + verified `references.bib`, and (ii) trim the abstract to 100 words.
