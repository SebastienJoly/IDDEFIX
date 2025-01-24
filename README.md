<img src="https://raw.githubusercontent.com/SebastienJoly/GARFIELD/main/logo.png"  width="100px"/>

# IDDEFIX

**I**mpedance **D**etermination through **D**ifferential **E**volution **FI**tting and e**X**trapolation.
![PyPI - Version](https://img.shields.io/pypi/v/IDDEFIX?style=flat-square&color=green)
![PyPI - License](https://img.shields.io/pypi/l/IDDEFIX?style=flat-square&color=pink)
![Tokei - LOC](https://tokei.rs/b1/github/ImpedanCEI/IDDEFIX?category=code?/style=square&color=green)


`IDDEFIX` is a tool for fitting resonators on impedance data using the Differential Evolution algorithm developed by Sébastian Joly. 
It computes the shunt impedance, Q-factor and resonant frequecny of the resonators present in the impedance data for both partially and fully decayed wakes. By incorporating the partially decayed resonator fitting, `IDDEFIX`  enables the extrapolation of non-converged impedances.


## About

🚀 `IDDEFIX` features:

* Resonators formulas
    * Longitudinal and transverse impedance (Fully/ partially decayed)
    * Longitudinal and transverse wake
    * Longitudinal and transverse wake potentials

* Differential Evolution algorithm for fitting resonsators to impedance
    * **SciPy**
    * pyfde ClassicDE
    * pyfde JADE

* Smart Bound Determination for precise and easy boundary setting

## How to use / Examples

Check :file_folder: `examples/` for different DE resonator fitting cases
* Analytical resonator initialization and fitting
* Resonator fitting on accelerator cavity simulation and extrapolation
* Resonator fitting on beam wire scanner simulation
* Resonator fitting on SPS transistion device and extrapolation




<img src="https://mattermost.web.cern.ch/files/4si7ipbezfyjdmd1zzr567hswh/public?h=2dcugjRruq3p9yEYea-9f1mXPfUbuujKRNh8dTA77a4"/>

Author : Sébastien Joly (sebastien.joly@helmholtz-berlin.de)
