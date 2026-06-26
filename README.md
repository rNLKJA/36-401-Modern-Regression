<div align="center">

# 36-401 Modern Regression

Personal study notes and worked derivations for Carnegie Mellon's **36-401 Modern Regression** course (Fall 2025), written and knitted in R Markdown.

![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)
![R Markdown](https://img.shields.io/badge/R%20Markdown-198CE7?logo=rstudio&logoColor=white)
![Topic](https://img.shields.io/badge/topic-linear%20regression-4c1)
![Type](https://img.shields.io/badge/type-study%20notes-blue)

</div>

## Overview

This repository follows the CMU 36-401 Modern Regression syllabus and works through the theory of linear regression from first principles. Each chapter pairs the mathematics (estimation, inference, prediction) with R code that fits and checks the models on real data sets, so the notes double as both a derivation reference and a runnable worked example.

The aim is to genuinely understand how linear models are learned and interpreted rather than just to apply them: where the estimators come from, why the assumptions matter, and how to read the output.

## Topics covered

- Review of random variables, expectation, variance and covariance as the groundwork for regression.
- The simple linear regression model and the least squares estimators of the slope and intercept.
- Properties and inference for the estimated coefficients (the $\beta$), including estimating the error variance.
- Prediction and prediction intervals in simple linear regression.
- A short companion note on partial derivatives and "partial integrals" to support the derivations.

## What's inside

| Path                                                                    | What it holds                                                                                                                                           |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `notebook/`                                                             | The source R Markdown notes, one `.Rmd` per chapter plus the partial-derivatives explainer.                                                             |
| `Chapter-1---Review-of-Random_files/` … `Chapter-4---Prediction_files/` | Knitted GitHub-flavoured Markdown output of each chapter, with rendered figures under `figure-gfm/`. Browse these to read the notes directly on GitHub. |
| `formula.Rmd`                                                           | A complete formula reference for the simple linear regression model: assumptions, estimators and key results in one place.                              |
| `handout/`                                                              | Course handouts, homework briefs and the syllabus (third party course material, see note below).                                                        |

## Highlights

- Notes are written to be **read on GitHub**: each chapter's knitted `README.md` renders the maths and the figures inline.
- Every result is tied back to its derivation, so the estimators and intervals are explained, not just stated.
- R code is embedded throughout, so the same files serve as a reproducible analysis when re-knitted.

## Getting started

These are R Markdown documents. To re-render a chapter from source:

```r
# from R, in the repository root
install.packages("rmarkdown")   # if not already installed
rmarkdown::render("notebook/Chapter 1 - Review of Random.Rmd")
```

Or open any `.Rmd` in RStudio and click **Knit**. The knitted Markdown and figures are also committed under the `*_files/` folders, so you can read everything without running R.

## A note on course material

The PDFs under `handout/` (syllabus, lecture handouts and homework briefs) are official 36-401 course material and remain the copyright of Carnegie Mellon University and the course staff. They are kept here only as a personal study reference and are not for redistribution. The notes themselves are my own working through of the material.
