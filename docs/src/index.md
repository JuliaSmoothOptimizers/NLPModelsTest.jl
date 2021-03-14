# [NLPModelsTest.jl documentation](@id Home)

This package provides testing functions for packages implementing optimization models using the [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API.

## Usage

This packages export commonly used problems and functions to test optimization models using the NLPModels API.
There are currently the following tests in this package:

- **Consistency**: Given 2 or more models of the same problem, do they behave the same way?
- **Multiple precision**: Given a model in a floating point type, do the API functions output have the same type?
- **Input dimension check**: Do the functions in this model correctly check the input dimensions, and throw the correct error otherwise?
- **View subarray support**: Check that your model accepts `@view` subarrays.
- **Coord memory**: (incomplete) Check that in place version of coord functions don't use too much memory.

The [TL;DR](@ref) section shows an example using these functions.

### [Consistency](@id consistency)

Two functions are given, one for NLP problems and another for NLS problems:

```@docs
consistent_nlps
consistent_nlss
```

To use them, implement a few or all of these [Problems](@ref), and call these functions on an array with both the model you created and the model we have here.

### Multiple precision

Two functions are given, one for NLP problems and another for NLS problems:

```@docs
multiple_precision_nlp
multiple_precision_nls
```

To use this function simply call it on your model.

### Check dimensions

Two functions are given, one for NLP problems and another for NLS problems:

```@docs
check_nlp_dimensions
check_nls_dimensions
```

To use this function simply call it on your model.

### View subarray support

Two functions are given, one for NLP problems and another for NLS problems:

```@docs
view_subarray_nlp
view_subarray_nls
```

To use this function simply call it on your model.

### Coordinate functions memory usage

Disclaimer: This function is incomplete.

```@docs
coord_memory_nlp
```

### Derivative Checker

Inside the consistency check, the following functions are used to check whether the derivatives are correct.
You can also use the manually.

```@docs
gradient_check
jacobian_check
hessian_check_from_grad
hessian_check
```

## TL;DR

```
TODO after CUTEst.jl and NLPModelsJuMP are updated.
```

## License

This content is released under the [MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/) License.

## Contents

```@contents
```
