export NLSHS20

"""
    nls = NLSH20()

## Problem 20 in the Hock-Schittkowski suite in nonlinear least squares format

```math
\\begin{aligned}
\\min \\quad & \\tfrac{1}{2}\\| F(x) \\|^2 \\\\
\\text{s. to} \\quad & x_1 + x_2^2 \\geq 0 \\\\
& x_1^2 + x_2 \\geq 0 \\\\
& x_1^2 + x_2^2 -1 \\geq 0 \\\\
& -0.5 \\leq x_1 \\leq 0.5
\\end{aligned}
```
where
```math
F(x) = \\begin{bmatrix}
1 - x_1 \\\\
10 (x_2 - x_1^2)
\\end{bmatrix}.
```

Starting point: `[-2; 1]`.
"""
mutable struct NLSHS20{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
end

function NLSHS20(::Type{S}) where {S}
  T = eltype(S)
  meta = NLPModelMeta{T, S}(
    2,
    x0 = S([-2; 1]),
    name = "NLSHS20_manual",
    lvar = S([-1 // 2; -T(Inf)]),
    uvar = S([1 // 2; T(Inf)]),
    ncon = 3,
    lcon = fill!(S(undef, 3), 0),
    ucon = fill!(S(undef, 3), T(Inf)),
    nnzj = 6,
  )
  nls_meta = NLSMeta{T, S}(2, 2, nnzj = 3, nnzh = 1)

  return NLSHS20(meta, nls_meta, NLSCounters())
end
NLSHS20() = NLSHS20(Float64)
NLSHS20(::Type{T}) where {T <: Number} = NLSHS20(Vector{T})

function NLPModels.residual!(nls::NLSHS20, x::AbstractVector, Fx::AbstractVector)
  @lencheck 2 x Fx
  increment!(nls, :neval_residual)
  Fx[1] = 1 - x[1]
  Fx[2] = 10 * (x[2] - x[1]^2)
  return Fx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure_residual!(
  nls::NLSHS20,
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

function NLPModels.jac_coord_residual!(nls::NLSHS20, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac_residual)
  vals[1] = -1
  vals[2] = -20x[1]
  vals[3] = 10
  return vals
end

function NLPModels.jprod_residual!(
  nls::NLSHS20,
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
  nls::NLSHS20,
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
  nls::NLSHS20,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord_residual!(
  nls::NLSHS20,
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
  nls::NLSHS20,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck 2 x v Hiv
  increment!(nls, :neval_hprod_residual)
  if i == 2
    Hiv[1] = -20v[1]
    Hiv[2] = 0
  else
    Hiv .= zero(eltype(x))
  end
  return Hiv
end

function NLPModels.cons_nln!(nls::NLSHS20, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 3 cx
  increment!(nls, :neval_cons_nln)
  cx[1] = x[1] + x[2]^2
  cx[2] = x[1]^2 + x[2]
  cx[3] = x[1]^2 + x[2]^2 - 1
  return cx
end

function NLPModels.jac_nln_structure!(
  nls::NLSHS20,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 6 rows cols
  rows[1] = 1
  cols[1] = 1
  rows[2] = 1
  cols[2] = 2
  rows[3] = 2
  cols[3] = 1
  rows[4] = 2
  cols[4] = 2
  rows[5] = 3
  cols[5] = 1
  rows[6] = 3
  cols[6] = 2
  return rows, cols
end

function NLPModels.jac_nln_coord!(nls::NLSHS20, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 6 vals
  increment!(nls, :neval_jac_nln)
  vals[1] = 1
  vals[2] = 2x[2]
  vals[3] = 2x[1]
  vals[4] = 1
  vals[5] = 2x[1]
  vals[6] = 2x[2]
  return vals
end

function NLPModels.jprod_nln!(
  nls::NLSHS20,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 3 Jv
  increment!(nls, :neval_jprod_nln)
  Jv[1] = v[1] + 2x[2] * v[2]
  Jv[2] = 2x[1] * v[1] + v[2]
  Jv[3] = 2x[1] * v[1] + 2x[2] * v[2]
  return Jv
end

function NLPModels.jtprod_nln!(
  nls::NLSHS20,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 3 v
  increment!(nls, :neval_jtprod_nln)
  Jtv[1] = v[1] + 2x[1] * (v[2] + v[3])
  Jtv[2] = v[2] + 2x[2] * (v[1] + v[3])
  return Jtv
end

function NLPModels.hess_structure!(
  nls::NLSHS20,
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
  nls::NLSHS20,
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_hess)
  vals[1] = T(1) - 200 * x[2] + 600 * x[1]^2
  vals[2] = -200 * x[1]
  vals[3] = T(100)
  vals .*= obj_weight
  return vals
end

function NLPModels.hess_coord!(
  nls::NLSHS20,
  x::AbstractVector{T},
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  @lencheck 2 x
  @lencheck 3 y
  @lencheck 3 vals
  increment!(nls, :neval_hess)
  vals[1] = obj_weight * (T(1) - 200 * x[2] + 600 * x[1]^2) + 2 * y[2] + 2 * y[3]
  vals[2] = -obj_weight * 200 * x[1]
  vals[3] = obj_weight * T(100) + 2 * y[1] + 2 * y[3]
  return vals
end

function NLPModels.hprod!(
  nls::NLSHS20,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nls, :neval_hprod)
  Hv[1] = (obj_weight * (T(1) - 200 * x[2] + 600 * x[1]^2)) * v[1] - obj_weight * 200 * x[1] * v[2]
  Hv[2] = -obj_weight * 200 * x[1] * v[1] + (obj_weight * T(100)) * v[2]
  return Hv
end

function NLPModels.hprod!(
  nls::NLSHS20,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nls, :neval_hprod)
  Hv[1] =
    (obj_weight * (T(1) - 200 * x[2] + 600 * x[1]^2) + 2 * y[2] + 2 * y[3]) * v[1] -
    obj_weight * 200 * x[1] * v[2]
  Hv[2] = -obj_weight * 200 * x[1] * v[1] + (obj_weight * T(100) + 2 * y[1] + 2 * y[3]) * v[2]
  return Hv
end

function NLPModels.jth_hprod!(
  nls::NLSHS20,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 3 j
  NLPModels.increment!(nls, :neval_jhprod)
  Hv .= zero(T)
  if j == 1
    Hv[2] = 2v[2]
  elseif j == 2
    Hv[1] = 2v[1]
  elseif j == 3
    Hv[1] = 2v[1]
    Hv[2] = 2v[2]
  end
  return Hv
end

function NLPModels.jth_hess_coord!(
  nls::NLSHS20,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 3 vals
  @lencheck 2 x
  @rangecheck 1 3 j
  NLPModels.increment!(nls, :neval_jhess)
  vals .= zero(T)
  if j == 1
    vals[3] = T(2)
  elseif j == 2
    vals[1] = T(2)
  elseif j == 3
    vals[1] = T(2)
    vals[3] = T(2)
  end
  return vals
end

function NLPModels.ghjvprod!(
  nls::NLSHS20,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nls.meta.nvar x g v
  @lencheck nls.meta.ncon gHv
  increment!(nls, :neval_hprod)
  gHv[1] = g[2] * 2v[2]
  gHv[2] = g[1] * 2v[1]
  gHv[3] = g[1] * 2v[1] + g[2] * 2v[2]
  return gHv
end
