export HS10

"""
    nlp = HS10()

## Problem 10 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & x_1 - x_2 \\\\
\\text{s. to} \\quad & -3x_1^2 + 2x_1 x_2 - x_2^2 + 1 \\geq 0
\\end{aligned}
```

Starting point: `[-10; 10]`.
"""
mutable struct HS10{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function HS10(::Type{S}) where {S}
  T = eltype(S)
  meta = NLPModelMeta{T, S}(
    2,
    ncon = 1,
    x0 = S([-10; 10]),
    lcon = fill!(S(undef, 1), 0),
    ucon = fill!(S(undef, 1), T(Inf)),
    name = "HS10_manual",
  )

  return HS10(meta, Counters())
end
HS10() = HS10(Float64)
HS10(::Type{T}) where {T <: Number} = HS10(Vector{T})

function NLPModels.obj(nlp::HS10, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return x[1] - x[2]
end

function NLPModels.grad!(nlp::HS10, x::AbstractVector{T}, gx::AbstractVector{T}) where {T}
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx[1] = one(T)
  gx[2] = -one(T)
  return gx
end

function NLPModels.hess_structure!(nlp::HS10, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 3 rows cols
  rows[1] = 1
  rows[2] = 2
  rows[3] = 2
  cols[1] = 1
  cols[2] = 1
  cols[3] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::HS10,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = 1.0,
) where {T}
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nlp, :neval_hess)
  vals .= zero(T)
  return vals
end

function NLPModels.hprod!(
  nlp::HS10,
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = 1.0,
) where {T}
  @lencheck 2 x v Hv
  increment!(nlp, :neval_hprod)
  Hv .= zero(T)
  return Hv
end

function NLPModels.hess_coord!(
  nlp::HS10,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = 1.0,
) where {T}
  @lencheck 2 x
  @lencheck 1 y
  @lencheck 3 vals
  increment!(nlp, :neval_hess)
  vals[1] = -6 * y[1]
  vals[2] = 2 * y[1]
  vals[3] = -2 * y[1]
  return vals
end

function NLPModels.hprod!(
  nlp::HS10,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = 1.0,
)
  @lencheck 2 x v Hv
  @lencheck 1 y
  increment!(nlp, :neval_hprod)
  Hv[1] = y[1] * (-6 * v[1] + 2 * v[2])
  Hv[2] = y[1] * (2 * v[1] - 2 * v[2])
  return Hv
end

function NLPModels.cons_nln!(nlp::HS10, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x
  @lencheck 1 cx
  increment!(nlp, :neval_cons_nln)
  cx[1] = -3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1
  return cx
end

function NLPModels.jac_nln_structure!(
  nlp::HS10,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  rows[1] = 1
  cols[1] = 1
  rows[2] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.jac_nln_coord!(nlp::HS10, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac_nln)
  vals[1] = -6 * x[1] + 2 * x[2]
  vals[2] = 2 * x[1] - 2 * x[2]
  return vals
end

function NLPModels.jprod_nln!(nlp::HS10, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nlp, :neval_jprod_nln)
  Jv[1] = (-6 * x[1] + 2 * x[2]) * v[1] + (2 * x[1] - 2 * x[2]) * v[2]
  return Jv
end

function NLPModels.jtprod_nln!(nlp::HS10, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_nln)
  Jtv[1] = (-6 * x[1] + 2 * x[2]) * v[1]
  Jtv[2] = (2 * x[1] - 2 * x[2]) * v[1]
  return Jtv
end

function NLPModels.jth_hprod!(
  nlp::HS10,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 1 j
  NLPModels.increment!(nlp, :neval_jhprod)
  Hv[1] = -6 * v[1] + 2 * v[2]
  Hv[2] = 2 * v[1] - 2 * v[2]
  return Hv
end

function NLPModels.jth_hess_coord!(
  nlp::HS10,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 3 vals
  @lencheck 2 x
  @rangecheck 1 1 j
  NLPModels.increment!(nlp, :neval_jhess)
  vals[1] = T(-6)
  vals[2] = T(2)
  vals[3] = T(-2)
  return vals
end

function NLPModels.ghjvprod!(
  nlp::HS10,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
)
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv[1] = g[1] * (-6 * v[1] + 2 * v[2]) + g[2] * (2 * v[1] - 2 * v[2])
  return gHv
end
