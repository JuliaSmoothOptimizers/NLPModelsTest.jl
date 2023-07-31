export LLS

"""
    nls = LLS()

## Linear least squares

```math
\\begin{aligned}
\\min \\quad & \\tfrac{1}{2}\\| F(x) \\|^2 \\\\
\\text{s. to} \\quad & x_1 + x_2 \\geq 0
\\end{aligned}
```
where
```math
F(x) = \\begin{bmatrix}
x_1 - x_2 \\\\
x_1 + x_2 - 2 \\\\
x_2 - 2
\\end{bmatrix}.
```

Starting point: `[0; 0]`.
"""
mutable struct LLS{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
end

function LLS(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    x0 = zeros(T, 2),
    name = "LLS_manual",
    ncon = 1,
    lcon = T[0.0],
    ucon = T[Inf],
    nnzj = 2,
    nnzh = 2,
    lin = 1:1,
    lin_nnzj = 2,
    nln_nnzj = 0,
  )
  nls_meta = NLSMeta{T, Vector{T}}(3, 2, nnzj = 5, nnzh = 0)

  return LLS(meta, nls_meta, NLSCounters())
end
LLS() = LLS(Float64)

function NLPModels.residual!(nls::LLS, x::AbstractVector, Fx::AbstractVector)
  @lencheck 2 x
  @lencheck 3 Fx
  increment!(nls, :neval_residual)
  Fx[1] = x[1] - x[2]
  Fx[2] = x[1] + x[2] - 2
  Fx[3] = x[2] - 2
  return Fx
end

function NLPModels.jac_structure_residual!(
  nls::LLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 5 rows cols
  rows[1] = 1
  rows[2] = 1
  rows[3] = 2
  rows[4] = 2
  rows[5] = 3
  cols[1] = 1
  cols[2] = 2
  cols[3] = 1
  cols[4] = 2
  cols[5] = 2
  return rows, cols
end

function NLPModels.jac_coord_residual!(
  nls::LLS,
  x::AbstractVector{T},
  vals::AbstractVector,
) where {T}
  @lencheck 2 x
  @lencheck 5 vals
  increment!(nls, :neval_jac_residual)
  vals[1] = T(1)
  vals[2] = T(-1)
  vals[3] = T(1)
  vals[4] = T(1)
  vals[5] = T(1)
  return vals
end

function NLPModels.jprod_residual!(
  nls::LLS,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v
  @lencheck 3 Jv
  increment!(nls, :neval_jprod_residual)
  Jv[1] = v[1] - v[2]
  Jv[2] = v[1] + v[2]
  Jv[3] = v[2]
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::LLS,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 3 v
  increment!(nls, :neval_jtprod_residual)
  Jtv[1] = v[1] + v[2]
  Jtv[2] = -v[1] + v[2] + v[3]
  return Jtv
end

function NLPModels.hess_structure_residual!(
  nls::LLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 0 rows cols
  return rows, cols
end

function NLPModels.hess_coord_residual!(
  nls::LLS,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck 2 x
  @lencheck 3 v
  @lencheck 0 vals
  increment!(nls, :neval_hess_residual)
  return vals
end

function NLPModels.hprod_residual!(
  nls::LLS,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck 2 x v Hiv
  increment!(nls, :neval_hprod_residual)
  Hiv .= zero(eltype(x))
  return Hiv
end

function NLPModels.cons_lin!(nls::LLS, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nls, :neval_cons_lin)
  cx[1] = x[1] + x[2]
  return cx
end

function NLPModels.jac_lin_structure!(
  nls::LLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 1
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.jac_lin_coord!(nls::LLS, x::AbstractVector{T}, vals::AbstractVector) where {T}
  @lencheck 2 x vals
  increment!(nls, :neval_jac_lin)
  vals .= T(1)
  return vals
end

function NLPModels.jprod_lin!(nls::LLS, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nls, :neval_jprod_lin)
  Jv[1] = v[1] + v[2]
  return Jv
end

function NLPModels.jtprod_lin!(nls::LLS, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nls, :neval_jtprod_lin)
  Jtv .= v
  return Jtv
end

function NLPModels.hess_structure!(nls::LLS, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 2
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nls::LLS{T, S},
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight::T = one(T),
) where {T, S}
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nls, :neval_hess)
  vals[1] = 2obj_weight
  vals[2] = 3obj_weight
  return vals
end

function NLPModels.hprod!(
  nls::LLS,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nls, :neval_hprod)
  Hv[1] = 2 * obj_weight * v[1]
  Hv[2] = 3 * obj_weight * v[2]
  return Hv
end

function NLPModels.hprod!(
  nls::LLS,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nls, :neval_hprod)
  Hv[1] = 2 * obj_weight * v[1]
  Hv[2] = 3 * obj_weight * v[2]
  return Hv
end

function NLPModels.jth_hprod!(
  nls::LLS,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 1 j
  NLPModels.increment!(nls, :neval_jhprod)
  Hv .= zero(T)
  return Hv
end

function NLPModels.jth_hess_coord!(
  nls::LLS,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 2 vals
  @lencheck 2 x
  @rangecheck 1 1 j
  NLPModels.increment!(nls, :neval_jhess)
  vals .= zero(T)
  return vals
end

function NLPModels.ghjvprod!(
  nls::LLS,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nls.meta.nvar x g v
  @lencheck nls.meta.ncon gHv
  increment!(nls, :neval_hprod)
  gHv .= zero(T)
  return gHv
end
