export HS14

"""
    nlp = HS14()

## Problem 14 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & (x_1 - 2)^2 + (x_2 - 1)^2 \\\\
\\text{s. to} \\quad & x_1 - 2x_2 = -1 \\\\
& -\\tfrac{1}{4} x_1^2 - x_2^2 + 1 \\geq 0
\\end{aligned}
```

Starting point: `[2; 2]`.
"""
mutable struct HS14{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function HS14(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    nnzh = 2,
    ncon = 2,
    x0 = T[2; 2],
    lcon = T[-1; 0],
    ucon = T[-1; Inf],
    name = "HS14_manual",
    lin = 1:1,
    lin_nnzj = 2,
    nln_nnzj = 2,
  )

  return HS14(meta, Counters())
end
HS14() = HS14(Float64)

function NLPModels.obj(nlp::HS14, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return (x[1] - 2)^2 + (x[2] - 1)^2
end

function NLPModels.grad!(nlp::HS14, x::AbstractVector, gx::AbstractVector)
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 2); 2 * (x[2] - 1)]
  return gx
end

function NLPModels.hess_structure!(nlp::HS14, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 2
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::HS14,
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
  nlp::HS14,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y vals
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  vals[1] -= y[2] / 2
  vals[2] -= 2y[2]
  return vals
end

function NLPModels.hprod!(
  nlp::HS14,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y v Hv
  increment!(nlp, :neval_hprod)
  Hv .= 2obj_weight * v
  Hv[1] -= y[2] * v[1] / 2
  Hv[2] -= 2y[2] * v[2]
  return Hv
end

function NLPModels.cons_lin!(nlp::HS14, x::AbstractVector, cx::AbstractVector)
  @lencheck 1 cx
  @lencheck 2 x
  increment!(nlp, :neval_cons_lin)
  cx .= [x[1] - 2 * x[2]]
  return cx
end

function NLPModels.cons_nln!(nlp::HS14, x::AbstractVector, cx::AbstractVector)
  @lencheck 1 cx
  @lencheck 2 x
  increment!(nlp, :neval_cons_nln)
  cx .= [-x[1]^2 / 4 - x[2]^2 + 1]
  return cx
end

function NLPModels.jac_lin_structure!(
  nlp::HS14,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nlp::HS14,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::HS14, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nlp, :neval_jac_lin)
  vals .= [1, -2]
  return vals
end

function NLPModels.jac_nln_coord!(nlp::HS14, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nlp, :neval_jac_nln)
  vals .= [-x[1] / 2, -2 * x[2]]
  return vals
end

function NLPModels.jprod_lin!(nlp::HS14, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 1 Jv
  @lencheck 2 x v
  increment!(nlp, :neval_jprod_lin)
  Jv .= [v[1] - 2 * v[2]]
  return Jv
end

function NLPModels.jprod_nln!(nlp::HS14, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 1 Jv
  @lencheck 2 x v
  increment!(nlp, :neval_jprod_nln)
  Jv .= [-x[1] * v[1] / 2 - 2 * x[2] * v[2]]
  return Jv
end

function NLPModels.jtprod_lin!(nlp::HS14, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_lin)
  Jtv .= [v[1]; -2 * v[1]]
  return Jtv
end

function NLPModels.jtprod_nln!(nlp::HS14, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_nln)
  Jtv .= [-x[1] * v[1] / 2; -2 * x[2] * v[1]]
  return Jtv
end

function NLPModels.jth_hprod!(
  nlp::HS14,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 2 j
  NLPModels.increment!(nlp, :neval_jhprod)
  if j == 1
    Hv .= zero(T)
  elseif j == 2
    Hv[1] = -v[1] / 2
    Hv[2] = -2v[2]
  end
  return Hv
end

function NLPModels.jth_hess_coord!(
  nlp::HS14,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 2 vals
  @lencheck 2 x
  @rangecheck 1 2 j
  NLPModels.increment!(nlp, :neval_jhess)
  if j == 1
    vals .= zero(T)
  elseif j == 2
    vals[1] = T(-1 / 2)
    vals[2] = T(-2)
  end
  return vals
end

function NLPModels.ghjvprod!(
  nlp::HS14,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv .= [T(0); -g[1] * v[1] / 2 - 2 * g[2] * v[2]]
  return gHv
end
