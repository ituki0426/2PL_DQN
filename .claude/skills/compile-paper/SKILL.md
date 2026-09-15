---
name: compile-paper
description: Compile paper/paper.tex to PDF using the pLaTeX + dvipdfmx toolchain (Japanese jsbook class). Use when the user asks to build, compile, or generate paper/paper.pdf. Runs preflight checks for missing image and bib entries, then runs platex → pbibtex → platex → platex → dvipdfmx from paper/.
---

# compile-paper

`paper/paper.tex` is a Japanese `jsbook` LaTeX document using the `dvipdfmx` graphics driver. Build it with pLaTeX (not pdfLaTeX / lualatex).

## Preflight

Do these checks first. If anything is missing, list what and stop — do not attempt to compile.

1. **Toolchain**. Run `eval "$(/usr/libexec/path_helper)" && which platex pbibtex dvipdfmx`. If any is missing, tell the user to `brew install --cask mactex-no-gui` (interactive `sudo`; must be run by the user).
2. **Images**. Every `\includegraphics{...}` target must exist relative to `paper/`. Check with:
   ```
   grep -oE 'includegraphics(\[[^]]*\])?\{[^}]+\}' paper/paper.tex \
     | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u \
     | while read f; do [ -e "paper/$f" ] || [ -e "paper/$f.pdf" ] || [ -e "paper/$f.png" ] || [ -e "paper/$f.jpg" ] || echo "MISSING: $f"; done
   ```
3. **Citations**. Every `\cite{...}` key must be defined in `paper/myrefs.bib`:
   ```
   comm -23 <(grep -oE '\\cite[pt]?\{[^}]+\}' paper/paper.tex | sed -E 's/\\cite[pt]?\{([^}]+)\}/\1/' | tr ',' '\n' | sort -u) \
            <(grep -oE '^@[a-zA-Z]+\{[^,]+,' paper/myrefs.bib | sed -E 's/@[a-zA-Z]+\{([^,]+),/\1/' | sort -u)
   ```
   Non-empty output = undefined keys.

Both preflights are known to fail on the current `paper/paper.tex` (missing `Pic/` folder and most bib entries). Do not try to work around this by generating dummy assets — surface the gap.

## Compile

From the repo root, once preflight is clean:

```
eval "$(/usr/libexec/path_helper)"
cd paper
platex -interaction=nonstopmode -halt-on-error paper.tex
pbibtex paper
platex -interaction=nonstopmode -halt-on-error paper.tex
platex -interaction=nonstopmode -halt-on-error paper.tex
dvipdfmx paper.dvi
```

Result: `paper/paper.pdf`. Confirm with `ls -la paper/paper.pdf` and open with `open paper/paper.pdf`.

If any `platex` step exits non-zero, print the last ~40 lines of `paper/paper.log` and stop. Do not rerun with `-interaction=batchmode` to mask errors.

## Cleanup

Intermediate files after a successful build: `paper.aux paper.dvi paper.log paper.toc paper.bbl paper.blg paper.out`. Delete only if the user explicitly asks.

## Not this skill's job

- `LaTex/main.tex` (a different, current IEICE paper in this repo) — has its own asset layout (`ieicej.cls`, `references.bib`, `fig_selection.pdf`) and is compiled the same way but from `LaTex/`. Not covered here.
- Generating or restoring missing figures / bib entries.
