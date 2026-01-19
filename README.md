# Juno Haze Classification System (HCS)

*A reproducible pipeline for detecting and analyzing detached haze layers in Jupiter’s atmosphere using JunoCam imagery, NAIF SPICE geometry, and ISIS3.*

---

## Overview

The **Juno Haze Classification System (HCS)** is an end-to-end analysis pipeline developed to identify and characterize detached atmospheric haze layers along Jupiter’s limb using data from the **JunoCam** instrument. The system integrates photometric brightness-gradient analysis with SPICE-based geometric constraints to distinguish primary cloud decks from secondary, optically detached haze features.

All processing is performed on a **per-IMG basis**, ensuring that each JunoCam image is analyzed independently with fully traceable intermediate products. This design prevents output collisions, supports reproducibility, and allows statistical aggregation across images to be performed as a distinct analytical step.

The pipeline is implemented as a sequence of modular Python stages:

- Framelet ingestion and SPICE initialization (ISIS3)
- Sub-pixel limb tracing
- Limb rectification and perpendicular sampling
- Brightness-gradient peak classification
- Latitude-binned statistical analysis of haze occurrence

Scientific conclusions are derived conservatively from **aggregate statistical behavior**, rather than individual limb profiles, in order to respect intrinsic geometric limitations at the planetary limb.

This repository accompanies the manuscript  
**“Detached Haze Classification Along the Jovian Limb Using JunoCam”**  
and contains the complete software and workflow required to reproduce the published results.

---

## Repository Scope

- Fully reproducible for the JunoCam IMG(s) analyzed in the paper  
- Explicit handling of SPICE geometry limitations  
- Designed for extension to additional images and future studies  

---

## Citation

If you use this software or methodology in your work, please cite the accompanying manuscript and this repository.  
A machine-readable citation file is provided in `CITATION.cff`.

---

