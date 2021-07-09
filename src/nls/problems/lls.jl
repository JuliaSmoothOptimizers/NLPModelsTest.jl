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
  )
  nls_meta = NLSMeta{T, Vector{T}}(3, 2, nnzj = 5, nnzh = 0)

  return LLS(meta, nls_meta, NLSCounters())
end
LLS() = LLS(Float64)

function NLPModels.residual!(nls::LLS, x::AbstractVector, Fx::AbstractVector)
  @lencheck 2 x
  @lencheck 3 Fx
  increment!(nls, :neval_residual)
  Fx .= [x[1] - x[2]; x[1] + x[2] - 2; x[2] - 2]
  return Fx
end

function NLPModels.jac_structure_residual!(
  nls::LLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 5 rows cols
  rows .= [1, 1, 2, 2, 3]
  cols .= [1, 2, 1, 2, 2]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::LLS, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 5 vals
  increment!(nls, :neval_jac_residual)
  T = eltype(x)
  vals .= T[1, -1, 1, 1, 1]
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
  Jv .= [v[1] - v[2]; v[1] + v[2]; v[2]]
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
  Jtv .= [v[1] + v[2]; -v[1] + v[2] + v[3]]
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

function NLPModels.cons!(nls::LLS, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nls, :neval_cons)
  cx[1] = x[1] + x[2]
  return cx
end

function NLPModels.jac_structure!(
  nls::LLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nls::LLS, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x vals
  increment!(nls, :neval_jac)
  T = eltype(x)
  vals .= T[1, 1]
  return vals
end

function NLPModels.jprod!(nls::LLS, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nls, :neval_jprod)
  Jv[1] = v[1] + v[2]
  return Jv
end

function NLPModels.jtprod!(nls::LLS, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nls, :neval_jtprod)
  Jtv .= v
  return Jtv
end

function NLPModels.hess_structure!(nls::LLS, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows .= [1, 2]
  cols .= [1, 2]
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
  vals .= [2obj_weight; 3obj_weight]
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
  gHv .= zeros(T, nls.meta.ncon)
  return gHv
end
