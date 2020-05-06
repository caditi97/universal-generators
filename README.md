# universal-generators

## References
- Identifying the Quantum Color Representation of New Particles with Machine Learning (CERN Talk)
  https://indico.cern.ch/event/782953/contributions/3460035/
- Identifying the Higgs Boson with Convolutional Neural Networks (Models)
  http://cs231n.stanford.edu/reports/2016/pdfs/300_Report.pdf
- Parton Shower Uncertainties in Jet Substructure Analyses with Deep Neural Networks (Samples)
  https://arxiv.org/abs/1609.00607


## Problem Statement
- Show universality of the quark/gluon parton shower generator using Deep Learning Jet Image technique.

## Installation
Currently these packages can only be installed for Linux and MacOs.
Packages should be installed in a specific order

1. [Boost](https://www.boost.org/)
2. [Pythia](http://home.thep.lu.se/Pythia/) (ver.8.2.135 - most recent version should work too)
3. [Fastjet](http://fastjet.fr/) (ver.3.3.2) along with the [fj-contrib](https://fastjet.hepforge.org/contrib/downloads/) (ver.1.039)
4. [Madgraph](https://launchpad.net/mg5amcnlo) (ver.2.5.5)
    - *pythia-pkgs from this [tutorial](https://twiki.cern.ch/twiki/bin/view/CMSPublic/MadgraphTutorial).*
    - *Do not install Delphes however, as it might mess things up.*
5. [ROOT](https://root.cern.ch/downloading-root) (ver 6.10 - some others might work). 
    - *Do not build it from source* 
6. [Rootpy](http://www.rootpy.org/install.html) (can use Conda to install too)
