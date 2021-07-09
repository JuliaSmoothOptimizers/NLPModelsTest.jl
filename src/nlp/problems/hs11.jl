export HS11

"""
    nlp = HS11()

## Problem 11 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & (x_1 - 5)^2 + x_2^2 - 25 \\\\
\\text{s. to} \\quad & -x_1^2 + x_2 \\leq 0
\\end{aligned}
```

Starting point: `[-4.9; 0.1]`.
"""
mutable struct HS11{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function HS11(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    ncon = 1,
    nnzh = 2,
    nnzj = 2,
    x0 = T[4.9; 0.1],
    lcon = T[-Inf],
    ucon = T[0],
    name = "HS11_manual",
  )

  return HS11(meta, Counters())
end
HS11() = HS11(Float64)

function NLPModels.obj(nlp::HS11, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return (x[1] - 5)^2 + x[2]^2 - 25
end

function NLPModels.grad!(nlp::HS11, x::AbstractVector, gx::AbstractVector)
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 5); 2 * x[2]]
  return gx
end

function NLPModels.hess_structure!(nlp::HS11, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 2
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::HS11,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x vals
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  return vals
end

function NLPModels.hess_coord!(
  nlp::HS11,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x vals
  @lencheck 1 y
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  vals[1] -= 2y[1]
  return vals
end

function NLPModels.hprod!(
  nlp::HS11,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  @lencheck 1 y
  increment!(nlp, :neval_hprod)
  Hv .= 2obj_weight * v
  Hv[1] -= 2y[1] * v[1]
  return Hv
end

function NLPModels.cons!(nlp::HS11, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons)
  cx .= [-x[1]^2 + x[2]]
  return cx
end

function NLPModels.jac_structure!(nlp::HS11, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp::HS11, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac)
  vals .= [-2 * x[1], 1]
  return vals
end

function NLPModels.jprod!(nlp::HS11, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod)
  Jv .= [-2 * x[1] * v[1] + v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp::HS11, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod)
  Jtv .= [-2 * x[1]; 1] * v[1]
  return Jtv
end

function NLPModels.jth_hprod!(
  nlp::HS11,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 1 j
  NLPModels.increment!(nlp, :neval_jhprod)
  Hv .= [-2v[1]; zero(T)]
  return Hv
end

function NLPModels.jth_hess_coord!(
  nlp::HS11,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 2 vals
  @lencheck 2 x
  @rangecheck 1 1 j
  NLPModels.increment!(nlp, :neval_jhess)
  vals[1] = T(-2)
  vals[2] = zero(T)
  return vals
end

function NLPModels.ghjvprod!(
  nlp::HS11,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
)
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv .= [-2 * g[1] * v[1]]
  return gHv
end
