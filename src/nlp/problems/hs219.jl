export HS219

"""
    nlp = HS219()

## Problem 219 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & -x_1 \\\\
\\text{s. to} \\quad & x_1^2 - x_2 - x_4^2 = 0 \\\\
& x_2 - x_1^3 - x_3^2 = 0
\\end{aligned}
```

Starting point: `[10; 10; 10; 10]`.
"""
mutable struct HS219{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

# function hs219(args...; kwargs...)
#   nlp = Model()
#   x0 = [10, 10, 10, 10]
#   @variable(nlp, x[i = 1:4], start = x0[i])

#   @constraint(nlp, x[1]^2 - x[2] - x[4]^2 == 0)
#   @NLconstraint(nlp, x[2] - x[1]^3 - x[3]^2 == 0)

#   @NLobjective(
#     nlp,
#     Min,
#     -x[1]
#   )

#   return nlp
# end

function HS219(::Type{S}) where {S}
  T = eltype(S)
  meta = NLPModelMeta{T, S}(
    2,
    nnzh = 2,
    ncon = 2,
    x0 = S([10; 10; 10; 10]),
    lcon = S([0; 0]),
    ucon = S([0; 0]),
    name = "HS219_manual",
  )

  return HS219(meta, Counters())
end
HS219() = HS219(Float64)
HS219(::Type{T}) where {T <: Number} = HS219(Vector{T})

function NLPModels.obj(nlp::HS219, x::AbstractVector)
  @lencheck 4 x
  increment!(nlp, :neval_obj)
  return -x[1]
end

function NLPModels.grad!(nlp::HS219, x::AbstractVector, gx::AbstractVector)
  @lencheck 4 x gx
  increment!(nlp, :neval_grad)
  gx[1] = -1
  gx[2] = 0
  gx[3] = 0
  gx[4] = 0
  return gx
end

function NLPModels.hess_structure!(nlp::HS219, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 2
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::HS219,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x vals
  increment!(nlp, :neval_hess)
  vals .= 2obj_weight
  return vals
end

function NLPModels.hprod!(
  nlp::HS219,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nlp, :neval_hprod)
  Hv .= 2obj_weight .* v
  return Hv
end

function NLPModels.hess_coord!(
  nlp::HS219,
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
  nlp::HS219,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y v Hv
  increment!(nlp, :neval_hprod)
  Hv .= 2obj_weight .* v
  Hv[1] -= y[2] * v[1] / 2
  Hv[2] -= 2y[2] * v[2]
  return Hv
end

function NLPModels.cons_lin!(nlp::HS219, x::AbstractVector, cx::AbstractVector)
  @lencheck 1 cx
  @lencheck 2 x
  increment!(nlp, :neval_cons_lin)
  cx[1] = x[1] - 2 * x[2]
  return cx
end

function NLPModels.cons_nln!(nlp::HS219, x::AbstractVector, cx::AbstractVector)
  @lencheck 1 cx
  @lencheck 2 x
  increment!(nlp, :neval_cons_nln)
  cx[1] = -x[1]^2 / 4 - x[2]^2 + 1
  return cx
end

function NLPModels.jac_lin_structure!(
  nlp::HS219,
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

function NLPModels.jac_nln_structure!(
  nlp::HS219,
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

function NLPModels.jac_lin_coord!(nlp::HS219, x::AbstractVector{T}, vals::AbstractVector) where {T}
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nlp, :neval_jac_lin)
  vals[1] = T(1)
  vals[2] = T(-2)
  return vals
end

function NLPModels.jac_nln_coord!(nlp::HS219, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nlp, :neval_jac_nln)
  vals[1] = -x[1] / 2
  vals[2] = -2 * x[2]
  return vals
end

function NLPModels.jprod_lin!(nlp::HS219, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 1 Jv
  @lencheck 2 x v
  increment!(nlp, :neval_jprod_lin)
  Jv[1] = v[1] - 2 * v[2]
  return Jv
end

function NLPModels.jprod_nln!(nlp::HS219, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 1 Jv
  @lencheck 2 x v
  increment!(nlp, :neval_jprod_nln)
  Jv[1] = -x[1] * v[1] / 2 - 2 * x[2] * v[2]
  return Jv
end

function NLPModels.jtprod!(nlp::HS219, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 2 v
  increment!(nlp, :neval_jtprod)
  Jtv[1] = v[1] - x[1] * v[2] / 2
  Jtv[2] = -2 * v[1] - 2 * x[2] * v[2]
  return Jtv
end

function NLPModels.jtprod_lin!(nlp::HS219, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_lin)
  Jtv[1] = v[1]
  Jtv[2] = -2 * v[1]
  return Jtv
end

function NLPModels.jtprod_nln!(nlp::HS219, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_nln)
  Jtv[1] = -x[1] * v[1] / 2
  Jtv[2] = -2 * x[2] * v[1]
  return Jtv
end

function NLPModels.jth_hprod!(
  nlp::HS219,
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
  nlp::HS219,
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
  nlp::HS219,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv[1] = T(0)
  gHv[2] = -g[1] * v[1] / 2 - 2 * g[2] * v[2]
  return gHv
end
