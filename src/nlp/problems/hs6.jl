export HS6

"""
    nlp = HS6()

## Problem 6 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & (1 - x_1)^2 \\\\
\\text{s. to} \\quad & 10 (x_2 - x_1^2) = 0
\\end{aligned}
```

Starting point: `[-1.2; 1.0]`.
"""
mutable struct HS6{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function HS6(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    ncon = 1,
    nnzh = 1,
    nnzj = 2,
    x0 = T[-1.2; 1],
    lcon = T[0],
    ucon = T[0],
    name = "HS6_manual",
  )

  return HS6(meta, Counters())
end
HS6() = HS6(Float64)

function NLPModels.obj(nlp::HS6, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return (1 - x[1])^2
end

function NLPModels.grad!(nlp::HS6, x::AbstractVector{T}, gx::AbstractVector) where {T}
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx[1] = 2 * (x[1] - 1)
  gx[2] = zero(T)
  return gx
end

function NLPModels.hess_structure!(nlp::HS6, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::HS6,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x
  @lencheck 1 vals
  increment!(nlp, :neval_hess)
  vals[1] = 2obj_weight
  return vals
end

function NLPModels.hprod!(
  nlp::HS6,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nlp, :neval_hprod)
  Hv[1] = 2obj_weight * v[1]
  Hv[2] = zero(T)
  return Hv
end

function NLPModels.hess_coord!(
  nlp::HS6,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x
  @lencheck 1 y vals
  increment!(nlp, :neval_hess)
  vals[1] = 2obj_weight - 20y[1]
  return vals
end

function NLPModels.hprod!(
  nlp::HS6,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  @lencheck 1 y
  increment!(nlp, :neval_hprod)
  Hv[1] = (2obj_weight - 20y[1]) * v[1]
  Hv[2] = zero(T)
  return Hv
end

function NLPModels.cons_nln!(nlp::HS6, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons_nln)
  cx[1] = 10 * (x[2] - x[1]^2)
  return cx
end

function NLPModels.jac_nln_structure!(
  nlp::HS6,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 1
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.jac_nln_coord!(nlp::HS6, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac_nln)
  vals[1] = -20 * x[1]
  vals[2] = 10
  return vals
end

function NLPModels.jprod_nln!(nlp::HS6, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod_nln)
  Jv[1] = -20 * x[1] * v[1] + 10 * v[2]
  return Jv
end

function NLPModels.jtprod_nln!(nlp::HS6, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_nln)
  Jtv[1] = -20 * x[1] * v[1]
  Jtv[2] = 10 * v[1]
  return Jtv
end

function NLPModels.jth_hprod!(
  nlp::HS6,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 1 j
  NLPModels.increment!(nlp, :neval_jhprod)
  Hv[1] = -20v[1]
  Hv[2] = zero(T)
  return Hv
end

function NLPModels.jth_hess_coord!(
  nlp::HS6,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 1 vals
  @lencheck 2 x
  @rangecheck 1 1 j
  NLPModels.increment!(nlp, :neval_jhess)
  vals[1] = T(-20)
  return vals
end

function NLPModels.ghjvprod!(
  nlp::HS6,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
)
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv[1] = -20 * g[1] * v[1]
  return gHv
end
