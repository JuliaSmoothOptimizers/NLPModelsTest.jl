# NLPModelsTest

This package provides testing facilities for developers of models implementing the [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API.

## How to Cite

If you use NLPModelsTest.jl in your work, please cite using the format given in [CITATION.bib](https://github.com/JuliaSmoothOptimizers/NLPModelsTest.jl/blob/main/CITATION.bib).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4603933.svg)](https://doi.org/10.5281/zenodo.4603933)
[![GitHub release](https://img.shields.io/github/release/JuliaSmoothOptimizers/NLPModelsTest.jl.svg)](https://github.com/JuliaSmoothOptimizers/NLPModelsTest.jl/releases/latest)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/NLPModelsTest.jl/stable)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/NLPModelsTest.jl/dev)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/NLPModelsTest.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/NLPModelsTest.jl)

![CI](https://github.com/JuliaSmoothOptimizers/NLPModelsTest.jl/workflows/CI/badge.svg?branch=main)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/NLPModelsTest.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/NLPModelsTest.jl)

## Usage

During the development of your model, you might find the need for more robust tests.
This package provides problems and functions to that end.

The main usage of this package are the consistency checks, which runs a comparison of two or models on all API functions.

Check the [docs](https://JuliaSmoothOptimizers.github.io/NLPModelsTest.jl/dev) for the complete usage.