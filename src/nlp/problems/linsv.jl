export LINSV

"""
    nlp = LINSV()

## Linear problem

```math
\\begin{aligned}
\\min \\quad & x_1 \\\\
\\text{s. to} \\quad & x_1 + x_2 \\geq 3 \\\\
& x_2 \\geq 1
\\end{aligned}
```

Starting point: `[0; 0]`.
"""
mutable struct LINSV{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function LINSV(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    nnzh = 0,
    nnzj = 3,
    ncon = 2,
    x0 = zeros(T, 2),
    lcon = T[3; 1],
    ucon = T[Inf; Inf],
    name = "LINSV_manual",
  )

  return LINSV(meta, Counters())
end
LINSV() = LINSV(Float64)

function NLPModels.obj(nlp::LINSV, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return x[1]
end

function NLPModels.grad!(nlp::LINSV, x::AbstractVector, gx::AbstractVector)
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx[1] = 1
  gx[2] = 0
  return gx
end

function NLPModels.hess_structure!(nlp::LINSV, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 0 rows cols
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::LINSV,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x
  @lencheck 0 vals
  increment!(nlp, :neval_hess)
  return vals
end

function NLPModels.hess_coord!(
  nlp::LINSV,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y
  @lencheck 0 vals
  increment!(nlp, :neval_hess)
  return vals
end

function NLPModels.hprod!(
  nlp::LINSV,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y v Hv
  increment!(nlp, :neval_hprod)
  Hv .= 0
  return Hv
end

function NLPModels.cons!(nlp::LINSV, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x cx
  increment!(nlp, :neval_cons)
  cx .= [x[1] + x[2]; x[2]]
  return cx
end

function NLPModels.jac_structure!(nlp::LINSV, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 3 rows cols
  rows .= [1, 1, 2]
  cols .= [1, 2, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp::LINSV, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nlp, :neval_jac)
  vals .= eltype(x).([1, 1, 1])
  return vals
end

function NLPModels.jprod!(nlp::LINSV, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v Jv
  increment!(nlp, :neval_jprod)
  Jv[1] = v[1] + v[2]
  Jv[2] = v[2]
  return Jv
end

function NLPModels.jtprod!(nlp::LINSV, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x v Jtv
  increment!(nlp, :neval_jtprod)
  Jtv[1] = v[1]
  Jtv[2] = v[1] + v[2]
  return Jtv
end

function NLPModels.jth_hprod!(
  nlp::LINSV,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 2 j
  NLPModels.increment!(nlp, :neval_jhprod)
  Hv .= zero(T)
  return Hv
end

function NLPModels.jth_hess_coord!(
  nlp::LINSV,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck nlp.meta.nnzh vals
  @lencheck 2 x
  @rangecheck 1 2 j
  NLPModels.increment!(nlp, :neval_jhess)
  vals .= zero(T)
  return vals
end

function NLPModels.ghjvprod!(
  nlp::LINSV,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv .= zeros(T, nlp.meta.ncon)
  return gHv
end
