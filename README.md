<img src="https://raw.githubusercontent.com/SebastienJoly/GARFIELD/main/logo.png"  width="100px"/>

# IDDEFIX
> Originally forked from https://github.com/SebastienJoly/IDDEFIX

**I**mpedance **D**etermination through **D**ifferential **E**volution **FI**tting and e**X**trapolation.

![PyPI - Version](https://img.shields.io/pypi/v/IDDEFIX?style=flat-square&color=green)
![PyPI - License](https://img.shields.io/pypi/l/IDDEFIX?style=flat-square&color=pink)
[![Documentation Status](https://readthedocs.org/projects/iddefix/badge/?version=latest)](https://iddefix.readthedocs.io/en/latest/?badge=latest)

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
## How to install
IDDEFIX is deployed to the [Python Package Index (pyPI)](https://pypi.org/project/iddefix/). To install it in a conda environment do:
```
pip install iddefix
```
It can also be installed directly from the Github source to get the latest changes:
```
pip install git+https://github.com/ImpedanCEI/IDDEFIX
```

## How to use / Examples

IDDEFIX is documented using `Sphinx` and `ReadTheDocs`. Documentation is available at: http://iddefix.readthedocs.io/ 

Check :file_folder: `examples/` for different DE resonator fitting cases
* Analytical resonator initialization and fitting
* Resonator fitting on accelerator cavity simulation and extrapolation
* Resonator fitting on beam wire scanner simulation
* Resonator fitting on SPS transistion device and extrapolation




<img src="https://mattermost.web.cern.ch/files/4si7ipbezfyjdmd1zzr567hswh/public?h=2dcugjRruq3p9yEYea-9f1mXPfUbuujKRNh8dTA77a4"/>

## Contributors :woman_technologist: :man_technologist:
* Author : Sébastien Joly (sebastien.joly@helmholtz-berlin.de)
* Collaborator : Malthe Raschke (malthe@raschke.dk)
  * Refactored code and PYPI deployment
  * Smart Bound Determination
  * Example notebooks for extrapolation of analytical and simulated devices
* Maintainer: Elena de la Fuente (elena.de.la.fuente.garcia@cern.ch)
