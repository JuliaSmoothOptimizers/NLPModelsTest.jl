export BNDROSENBROCK

"""
    nls = BNDROSENBROCK()

## Rosenbrock function in nonlinear least squares format with bound constraints.

```math
\\begin{aligned}
\\min \\quad & \\tfrac{1}{2}\\| F(x) \\|^2
\\text{s. to} \\quad & -1 \\leq x_1 \\leq 0.8 \\\\
& -2 \\leq x_2 \\leq 2
\\end{aligned}
```
where
```math
F(x) = \\begin{bmatrix}
1 - x_1 \\\\
10 (x_2 - x_1^2)
\\end{bmatrix}.
```

Starting point: `[-1.2; 1]`.
"""
mutable struct BNDROSENBROCK{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
end

function BNDROSENBROCK(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    x0 = T[-1.2; 1],
    lvar = T[-1; -2],
    uvar = T[0.8; 2],
    name = "BNDROSENBROCK_manual",
  )
  nls_meta = NLSMeta{T, Vector{T}}(2, 2, nnzj = 3, nnzh = 1)

  return BNDROSENBROCK(meta, nls_meta, NLSCounters())
end
BNDROSENBROCK() = BNDROSENBROCK(Float64)

function NLPModels.residual!(nls::BNDROSENBROCK, x::AbstractVector, Fx::AbstractVector)
  @lencheck 2 x Fx
  increment!(nls, :neval_residual)
  Fx[1] = 1 - x[1]
  Fx[2] = 10 * (x[2] - x[1]^2)
  return Fx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure_residual!(
  nls::BNDROSENBROCK,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 3 rows cols
  rows[1] = 1
  cols[1] = 1
  rows[2] = 2
  cols[2] = 1
  rows[3] = 2
  cols[3] = 2
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::BNDROSENBROCK, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac_residual)
  vals[1] = -1
  vals[2] = -20x[1]
  vals[3] = 10
  return vals
end

function NLPModels.jprod_residual!(
  nls::BNDROSENBROCK,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v Jv
  increment!(nls, :neval_jprod_residual)
  Jv[1] = -v[1]
  Jv[2] = -20 * x[1] * v[1] + 10 * v[2]
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::BNDROSENBROCK,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv[1] = -v[1] - 20 * x[1] * v[2]
  Jtv[2] = 10 * v[2]
  return Jtv
end

function NLPModels.hess_structure_residual!(
  nls::BNDROSENBROCK,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord_residual!(
  nls::BNDROSENBROCK,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 1 vals
  increment!(nls, :neval_hess_residual)
  vals[1] = -20v[2]
  return vals
end

function NLPModels.hprod_residual!(
  nls::BNDROSENBROCK,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck 2 x v Hiv
  increment!(nls, :neval_hprod_residual)
  if i == 2
    Hiv[1] = -20v[1]
    Hiv[2] = zero(eltype(x))
  else
    Hiv .= zero(eltype(x))
  end
  return Hiv
end

function NLPModels.hess_structure!(
  nls::BNDROSENBROCK,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 3 rows cols
  n = nls.meta.nvar
  k = 0
  for j = 1:n, i = j:n
    k += 1
    rows[k] = i
    cols[k] = j
  end
  return rows, cols
end

function NLPModels.hess_coord!(
  nls::BNDROSENBROCK,
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  @lencheck 2 x
  @lencheck 3 vals
  vals[1] = T(1) - 200 * x[2] + 600 * x[1]^2
  vals[2] = -200 * x[1]
  vals[3] = T(100)
  vals .*= obj_weight
  return vals
end

function NLPModels.hprod!(
  nls::BNDROSENBROCK,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nls, :neval_hprod)
  Hv[1] = obj_weight * ((T(1) - 200 * x[2] + 600 * x[1]^2) * v[1] - 200 * x[1] * v[2])
  Hv[2] = obj_weight * (-200 * x[1] * v[1] + T(100) * v[2])
  return Hv
end
