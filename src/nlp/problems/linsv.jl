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

function LINSV(::Type{S}) where {S}
  T = eltype(S)
  meta = NLPModelMeta{T, S}(
    2,
    nnzh = 0,
    nnzj = 3,
    ncon = 2,
    x0 = fill!(S(undef, 2), 0),
    lcon = S([3; 1]),
    ucon = S([T(Inf); T(Inf)]),
    name = "LINSV_manual",
    lin = 1:2,
    lin_nnzj = 3,
    nln_nnzj = 0,
  )

  return LINSV(meta, Counters())
end
LINSV() = LINSV(Float64)
LINSV(::Type{T}) where {T <: Number} = LINSV(Vector{T})

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

function NLPModels.hprod!(
  nlp::LINSV,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nlp, :neval_hprod)
  Hv .= 0
  return Hv
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

function NLPModels.cons_lin!(nlp::LINSV, x::AbstractVector, cx::AbstractVector)
  @lencheck 2 x cx
  increment!(nlp, :neval_cons_lin)
  cx[1] = x[1] + x[2]
  cx[2] = x[2]
  return cx
end

function NLPModels.jac_lin_structure!(
  nlp::LINSV,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 3 rows cols
  rows[1] = 1
  cols[1] = 1
  rows[2] = 1
  cols[2] = 2
  rows[3] = 2
  cols[3] = 2
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::LINSV, x::AbstractVector{T}, vals::AbstractVector) where {T}
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nlp, :neval_jac_lin)
  vals[1] = T(1)
  vals[2] = T(1)
  vals[3] = T(1)
  return vals
end

function NLPModels.jprod_lin!(nlp::LINSV, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v Jv
  increment!(nlp, :neval_jprod_lin)
  Jv[1] = v[1] + v[2]
  Jv[2] = v[2]
  return Jv
end

function NLPModels.jtprod_lin!(
  nlp::LINSV,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x v Jtv
  increment!(nlp, :neval_jtprod_lin)
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
  gHv .= zero(T)
  return gHv
end
