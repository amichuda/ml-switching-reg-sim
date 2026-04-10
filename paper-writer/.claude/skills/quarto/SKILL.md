---
name: quarto
description: Render, debug, and format Quarto documents (.qmd) to PDF via LaTeX. Use this skill whenever working with .qmd files — rendering to PDF, diagnosing render failures, fixing YAML headers, resolving LaTeX errors, managing bibliography files, tuning figure/table placement, or applying journal style guides. Trigger on: "render the qmd", "quarto PDF", "latex error", "fix the formatting", "table overflow", "figure placement", "bibliography not rendering", or any task involving a .qmd manuscript file.
---

# Quarto Skill

Render and debug Quarto documents to PDF using the Quarto CLI and a LaTeX engine (TinyTeX or system TeX).

## Setup

### Check Quarto is available
```bash
quarto --version
```

### Install TinyTeX if no LaTeX engine present
```bash
quarto install tinytex
```

### Install required LaTeX packages (run once per environment)
```bash
# Core packages for academic PDFs
tlmgr install setspace booktabs caption geometry natbib lm lm-math \
  microtype hyperref xcolor fancyhdr titlesec \
  amsmath amssymb float placeins endfloat \
  biblatex biber || true
```

### Install Python dependencies for knitr/jupyter chunks if needed
```bash
pip install jupyter nbformat --break-system-packages -q
```

## Rendering

### Basic render to PDF
```bash
cd path/to/manuscript && quarto render paper.qmd --to pdf
```

### Render with verbose output (essential for debugging)
```bash
quarto render paper.qmd --to pdf --verbose 2>&1 | tee render.log
```

### Render keeping intermediate .tex file (use for diagnosing LaTeX issues)
```bash
quarto render paper.qmd --to pdf --keep-tex 2>&1 | tee render.log
```

### Check for LaTeX errors in log
```bash
grep -E "^!|Error|error|Warning|Overfull|Underfull" render.log | head -60
```

### Get only fatal errors (lines starting with !)
```bash
grep "^!" paper.log 2>/dev/null || grep "^!" render.log
```

## YAML Header Structure for Academic PDFs

Minimal working header:
```yaml
---
title: "Your Paper Title"
author:
  - name: "First Author"
    affiliation: "Institution"
  - name: "Second Author"
    affiliation: "Institution"
date: today
abstract: |
  Your abstract text here. Maximum 100 words for JAE.
keywords: ["keyword1", "keyword2", "keyword3"]
format:
  pdf:
    documentclass: article
    classoption: [12pt]
    geometry: margin=1in
    linestretch: 2
    fontfamily: lmodern
    number-sections: true
    keep-tex: true
    include-in-header:
      text: |
        \usepackage{booktabs}
        \usepackage{setspace}
        \usepackage{natbib}
        \usepackage{float}
        \usepackage{placeins}
        \usepackage{caption}
        \captionsetup[table]{position=top}
        \captionsetup[figure]{position=bottom}
bibliography: references.bib
biblio-style: jae
natbiboptions: round
---
```

## Common YAML Options

| Option | Values | Effect |
|--------|--------|--------|
| `linestretch` | `2` | Double spacing |
| `fontsize` | `12pt` | Font size |
| `geometry` | `margin=1in` | Page margins |
| `number-sections` | `true/false` | Numbered headings |
| `keep-tex` | `true` | Retain .tex for debugging |
| `latex-engine` | `pdflatex`, `xelatex`, `lualatex` | LaTeX engine |
| `biblio-style` | `jae`, `apa`, `chicago` | Reference style |

## Figure Placement

### Force figure to stay in section (use FloatBarrier)
Add to header: `\usepackage{placeins}`

In document body, add after section where figure should appear:
```markdown
\FloatBarrier
```

### Figure sizing in chunks (R)
````markdown
```{r fig-name, fig.cap="Caption", fig.width=6.5, fig.height=4, fig.pos="htbp"}
# plot code
```
````

### Figure sizing (Python/matplotlib)
````markdown
```{python}
#| label: fig-name
#| fig-cap: "Caption"
#| fig-width: 6.5
#| fig-height: 4
#| fig-pos: "htbp"
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4))
```
````

### Recommended figure widths
- Single column: `fig.width=3.15` (80mm)
- Full width: `fig.width=7.09` (180mm)
- Standard US letter with 1" margins: `fig.width=6.5` (safe default)

### Include static figures (PDF/PNG from analysis repo)
```markdown
![Caption text.](../analysis/output/figures/fig1.pdf){#fig-name width=90%}
```
Or for precise control:
```markdown
\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{../analysis/output/figures/fig1.pdf}
\caption{Caption text.}
\label{fig:name}
\end{figure}
```

## Table Placement and Formatting

### Kable/kableExtra in R (recommended for JAE-style tables)
````markdown
```{r tbl-name}
#| label: tbl-name
#| tbl-cap: "Table caption"
#| echo: false
library(kableExtra)
kbl(df, booktabs = TRUE, digits = 3,
    format = "latex", linesep = "") |>
  kable_styling(latex_options = c("hold_position", "scale_down")) |>
  add_footnote("Notes: ...", notation = "none")
```
````

### Include static LaTeX tables from analysis repo
```markdown
\input{../analysis/output/tables/table1.tex}
```
Wrap in a float environment for caption/placement control:
```latex
\begin{table}[htbp]
\centering
\caption{Caption text.}
\input{../analysis/output/tables/table1.tex}
\begin{flushleft}
\small\textit{Note.} Notes text here.
\end{flushleft}
\end{table}
```

### Force table to stay in place
Use `[H]` placement specifier (requires `\usepackage{float}`):
```latex
\begin{table}[H]
```
Or add to table chunk options: `#| tbl-pos: "H"`

### Scale oversized tables
```latex
\begin{table}[htbp]
\centering
\caption{...}
\resizebox{\textwidth}{!}{%
\input{../analysis/output/tables/widetable.tex}
}
\end{table}
```

## Diagnosing Common Errors

### "Overfull \hbox" warnings
Table or figure wider than text width. Fix: use `scale_down` in kableExtra or `\resizebox`.

### "Float too large for page"
Figure height exceeds page. Reduce `fig.height` or use `[p]` placement for figure-only page.

### "! Package natbib Error: Bibliography not compatible"
Wrong biblio-style. Ensure `jae.bst` is in the working directory or on the TeX path.

### "undefined control sequence"
Missing LaTeX package. Add to `include-in-header`.

### Tables appearing at end of document
Quarto default holds floats. Add `\FloatBarrier` at section ends or use `[H]` placement.

### Bibliography not rendering
Check: (1) `.bib` file path is correct relative to the `.qmd` file; (2) `bibliography: references.bib` in YAML; (3) at least one citation in the document body.

### Page numbers overlapping footer content
Add to header:
```latex
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}
```

## Post-Render Checks

```bash
# Check PDF was created
ls -lh *.pdf

# Count pages
pdfinfo paper.pdf | grep Pages

# Check PDF is valid (not corrupted)
pdftotext paper.pdf /dev/null && echo "PDF OK" || echo "PDF corrupted"

# Open for visual inspection (if display available)
# open paper.pdf  # macOS
# xdg-open paper.pdf  # Linux
```

## Workflow Summary

1. Render with `--keep-tex --verbose`, pipe to `render.log`
2. Check log for errors: `grep "^!" paper.log`
3. Fix YAML / LaTeX issues in `.qmd`
4. Re-render and verify page count: `pdfinfo paper.pdf | grep Pages`
5. Check overfull boxes: `grep "Overfull" render.log`
6. Visual spot-check: table/figure placement, page number collisions, reference list format
