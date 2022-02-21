export HS13

"""
    nlp = HS13()

## Problem 13 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & (x_1 - 2)^2 + x_2^2 \\\\
\\text{s. to} \\quad & (1 - x_1)^3 - x_2 \\geq 0
\\end{aligned}
```

Starting point: `[-2; -2]`.
"""
mutable struct HS13{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function HS13(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    ncon = 1,
    x0 = T[-2; -2],
    lcon = T[0],
    ucon = T[Inf],
    name = "HS13_manual",
    nnzh = 2,
  )

  return HS13(meta, Counters())
end
HS13() = HS13(Float64)

function NLPModels.obj(nlp::HS13, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return (x[1] - 2)^2 + x[2]^2
end

function NLPModels.grad!(nlp::HS13, x::AbstractVector{T}, gx::AbstractVector{T}) where {T}
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= T[2 * (x[1] - 2); 2 * x[2]]
  return gx
end

function NLPModels.hess_structure!(nlp::HS13, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 2
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::HS13,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = 1.0,
) where {T}
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nlp, :neval_hess)
  vals .= obj_weight .* T[2; 2]
  return vals
end

function NLPModels.hess_coord!(
  nlp::HS13,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = 1.0,
) where {T}
  @lencheck 2 x
  @lencheck 1 y
  @lencheck 2 vals
  increment!(nlp, :neval_hess)
  vals .= obj_weight .* T[2; 2] + T[6 * (1 - x[1]); 0] * y[1]
  return vals
end

function NLPModels.hprod!(
  nlp::HS13,
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = 1.0,
) where {T}
  @lencheck 2 x v Hv
  increment!(nlp, :neval_hprod)
  Hv .= obj_weight .* T[2 * v[1]; 2 * v[2]]
  return Hv
end

function NLPModels.hprod!(
  nlp::HS13,
  x::AbstractVector{T},
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = 1.0,
) where {T}
  @lencheck 2 x v Hv
  @lencheck 1 y
  increment!(nlp, :neval_hprod)
  Hv .= obj_weight .* T[2 * v[1]; 2 * v[2]] + T[6 * (1 - x[1]) * y[1] * v[1]; 0]
  return Hv
end

function NLPModels.cons!(nlp::HS13, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons)
  cx .= [(1 - x[1])^3 - x[2]]
  return cx
end

function NLPModels.jac_structure!(nlp::HS13, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp::HS13, x::AbstractVector{T}, vals::AbstractVector) where {T}
  @lencheck 2 x vals
  increment!(nlp, :neval_jac)
  vals .= T[-3 * (1 - x[1])^2; -1]
  return vals
end

function NLPModels.jprod!(nlp::HS13, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod)
  Jv .= (-3 * (1 - x[1])^2) * v[1] - v[2]
  return Jv
end

function NLPModels.jtprod!(
  nlp::HS13,
  x::AbstractVector{T},
  v::AbstractVector,
  Jtv::AbstractVector,
) where {T}
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod)
  Jtv .= T[-3 * (1 - x[1])^2; -1] * v[1]
  return Jtv
end

function NLPModels.jth_hprod!(
  nlp::HS13,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 1 j
  NLPModels.increment!(nlp, :neval_jhprod)
  Hv .= [6 * (1 - x[1]) * v[1]; 0]
  return Hv
end

function NLPModels.jth_hess_coord!(
  nlp::HS13,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 2 vals
  @lencheck 2 x
  @rangecheck 1 1 j
  NLPModels.increment!(nlp, :neval_jhess)
  vals .= T[6 * (1 - x[1]); 0]
  return vals
end

function NLPModels.ghjvprod!(
  nlp::HS13,
  x::AbstractVector{T},
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
) where {T}
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv .= T[g[1] * 6 * (1 - x[1]) * v[1]]
  return gHv
end
