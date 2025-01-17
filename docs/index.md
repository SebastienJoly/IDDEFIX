---
sd_hide_title: true
---
# 🔎 Overview

## Welcome to `IDDEFIX` documentation

> **IDDEFIX**: **I**mpedance **D**etermination through **D**ifferential **E**volution **FI**tting and e**X**trapolation

`IDDEFIX` is a package for **fitting resonators by a Differential Evolution algorithm to Impedance data** developed by Sébastian Joly. It computes the shunt impedance, Q-factor and resonant frequecny of the resonators present in the impedance data for both partially and fully decayed wakes. Extrapolation to a desired wakelength is then possible to quickly reconstruct the fully decayed wake.


🚀 `IDDEFIX` features:

* Resonators formulas
    * Longitudinal and transverse impedance (Fully/ partially decayed)
    * Longitudinal and transverse wake
    * Longitudinal and transverse wake potentials

* Differential Evolution algorihm for fitting resonsators to impedance
    * SciPy
    * pyfde ClassicDE
    * pyfde JADE


The source code is available in the `IDDEFIX` [GitHub repository](https://github.com/ImpedanCEI/IDDEFIX).

```{toctree} 
:caption: Table of Contents
:maxdepth: 3

index.md
theory.md
iddefix.rst
```