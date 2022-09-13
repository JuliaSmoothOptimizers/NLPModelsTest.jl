# Test allocations of NLPModels

NLPModels has features to test allocations of any model following the NLPModel API defined in [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

Given a model `nlp` whose type is a subtype of `AbstractNLPModel` or `AbstractNLSModel`, the function [`test_allocs_nlpmodels`](@ref) returns a `Dict` containing the allocations of each of the following:
- `obj(nlp, x)` or `obj(nlp, x, Fx)` for `AbstractNLSModel`;
- `grad!(nlp, x, gx)` or `grad!(nlp, x, gx, Fx)` for `AbstractNLSModel`;
- `hess_structure!(nlp, rows, cols)`;
- `hess_coord!(nlp, x, vals)`;
- `hprod!(nlp, x, v, Hv)`;
- `mul!(Hv, H, v)` for a `H` a LinearOperator, see [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl), obtained by `hess_op!(nlp, x, Hv)`.

If the problem has constraints, we also check:
- `cons!(nlp, x, c)`;
- `jac_structure!(nlp, rows, cols)`;
- `jac_coord!(nlp, x, vals)`;
- `jprod!(nlp, x, v, Jv)`;
- `jtprod!(nlp, x, v, Jtv)`;
- `mul!(Jv, J, v)` for a `J` a LinearOperator, see [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl), obtained by `jac_op!(nlp, x, Jv, Jtv)`;
- `hess_coord!(nlp, x, y, vals)`;
- `hprod!(nlp, x, y, v, Hv)`;
- `mul!(Hv, H, v)` for a `H` a LinearOperator, see [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl), obtained by `hess_op!(nlp, x, y, Hv)`.

It is also possible to test the functions for the linear and nonlinear constraints by setting the keyword `linear_api = true` in the call to [`test_allocs_nlpmodels`](@ref).

For nonlinear least-squares, i.e. `AbstractNLSModel`, we can also test allocations of the functions related to the residual evaluation with the function [`test_allocs_nlsmodels`](@ref):
- `residual!(nlp, x, Fx)`
- `jac_structure_residual!(nlp, rows, cols)`
- `jac_coord_residual!(nlp, x, vals)`
- `jprod_residual!(nlp, x, v, Jv)`
- `jtprod_residual!(nlp, x, w, Jtv)`
- `mul!(Jv, J, v)` for a `J` a LinearOperator, see [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl), obtained by `jac_op_residual!(nlp, x, Jv, Jtv)`;
- `hess_structure_residual!(nlp, rows, cols)`
- `hess_coord_residual!(nlp, x, v, vals)`
- `hprod_residual!(nlp, x, i, v, Hv)`
- `mul!(Hv, H, v)` for a `H` a LinearOperator, see [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl), obtained by `hess_op_residual!(nlp, x, i, Hv)`.

The function [`print_nlp_allocations`](@ref) allows a better rending of the result.

## Examples

### Examples with an NLPModels

```@example nlp
using NLPModelsTest
list_of_problems = NLPModelsTest.nlp_problems
```

```@example nlp
nlp = eval(Symbol(list_of_problems[1]))()
```

```@example nlp
print_nlp_allocations(nlp, linear_api = true);
```

```@example nlp
nlp = eval(Symbol(list_of_problems[7]))()
print_nlp_allocations(nlp, linear_api = true);
```

### Examples with an NLSModels

```@example nlp
list_of_problems = NLPModelsTest.nls_problems
```

```@example nlp
nls = eval(Symbol(list_of_problems[4]))()
```

```@example nlp
print_nlp_allocations(nls, linear_api = true);
```

### Examples with a testing environment

The function [`test_zero_allocations`](@ref) combines [`test_allocs_nlpmodels`](@ref) and [`test_allocs_nlsmodels`](@ref) in a testing environment.
Note that all the problems manually implemented in this package are allocations free using Julia ≥ 1.7.

```@example nlp
list_of_nlps = map(x -> eval(Symbol(x))(), NLPModelsTest.nlp_problems) # load a list of nlpmodels
map(
    nlp -> test_zero_allocations(nlp, linear_api = true),
    list_of_nlps,
)
```
