export HS100

"""
    nlp = HS100()

## Problem 100 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & (x_1 - 10)^2 + 5 (x_2 - 12)^2 + x_3^4 + 3 (x_4 - 11)^2 + 10 x_5^6 + 7 x_6^2 + x_7^4 - 4 x_6 x_7 - 10 x_6 - 8 x_7 \\\\
\\text{s. to} \\quad & 127 - 2x_1^2 - 3x_2^4 - x_3 - 4x_4^2 - 5x_5 \ge 0 \\\\
& 282 - 7 x_1 - 3 x_2 - 10 x_3^2 - x_4 + x_5 \ge 0 \\\\
& 196 - 23 x_1 - x_2^2 - 6 x_6^2 + 8 x_7 \ge 0 \\\\
& -4 x_1^2 - x_2^2 + 3 x_1 x_2 - 2 x_3^2 - 5 x_6 + 11 * x_7 \ge 0
\\end{aligned}
````

Starting point: `[1; 2; 0; 4; 0; 1; 1]`.
"""
mutable struct HS100{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

# function hs100(args...; kwargs...)
#   nlp = Model()
#   x0 = [1, 2, 0, 4, 0, 1, 1]
#   @variable(nlp, x[i = 1:7], start = x0[i])

#   @NLconstraint(nlp, 127 - 2 * x[1]^2 - 3 * x[2]^4 - x[3] - 4 * x[4]^2 - 5 * x[5] ≥ 0)
#   @constraint(nlp, 282 - 7 * x[1] - 3 * x[2] - 10 * x[3]^2 - x[4] + x[5] ≥ 0)
#   @constraint(nlp, 196 - 23 * x[1] - x[2]^2 - 6 * x[6]^2 + 8 * x[7] ≥ 0)
#   @constraint(nlp, -4 * x[1]^2 - x[2]^2 + 3 * x[1] * x[2] - 2 * x[3]^2 - 5 * x[6] + 11 * x[7] ≥ 0)

#   @NLobjective(
#     nlp,
#     Min,
#     (x[1] - 10)^2 +
#     5 * (x[2] - 12)^2 +
#     x[3]^4 +
#     3 * (x[4] - 11)^2 +
#     10 * x[5]^6 +
#     7 * x[6]^2 +
#     x[7]^4 - 4 * x[6] * x[7] - 10 * x[6] - 8 * x[7]
#   )

#   return nlp
# end

function HS100(::Type{S}) where {S}
  T = eltype(S)
  meta = NLPModelMeta{T, S}(
    7,
    nnzh = 2,
    ncon = 2,
    x0 = S([1; 2; 0; 4; 0; 1; 1]),
    lcon = S([0; 0]),
    ucon = S([0; 0]),
    name = "HS100_manual",
    lin = ....,
    lin_nnzj = 2,
    nln_nnzj = 2,
  )

  return HS100(meta, Counters())
end
HS100() = HS100(Float64)
HS100(::Type{T}) where {T <: Number} = HS100(Vector{T})

function NLPModels.obj(nlp::HS100, x::AbstractVector)
  @lencheck 4 x
  increment!(nlp, :neval_obj)
  return -x[1]
end

function NLPModels.grad!(nlp::HS100, x::AbstractVector, gx::AbstractVector)
  @lencheck 4 x gx
  increment!(nlp, :neval_grad)
  gx[1] = -1
  gx[2] = 0
  gx[3] = 0
  gx[4] = 0
  return gx
end

function NLPModels.hess_structure!(nlp::HS100, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows[1] = 1
  rows[2] = 2
  cols[1] = 1
  cols[2] = 2
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::HS100,
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
  nlp::HS100,
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
  nlp::HS100,
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
  nlp::HS100,
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

function NLPModels.cons_lin!(nlp::HS100, x::AbstractVector, cx::AbstractVector)
  @lencheck 1 cx
  @lencheck 2 x
  increment!(nlp, :neval_cons_lin)
  cx[1] = x[1] - 2 * x[2]
  return cx
end

function NLPModels.cons_nln!(nlp::HS100, x::AbstractVector, cx::AbstractVector)
  @lencheck 1 cx
  @lencheck 2 x
  increment!(nlp, :neval_cons_nln)
  cx[1] = -x[1]^2 / 4 - x[2]^2 + 1
  return cx
end

function NLPModels.jac_lin_structure!(
  nlp::HS100,
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
  nlp::HS100,
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

function NLPModels.jac_lin_coord!(nlp::HS100, x::AbstractVector{T}, vals::AbstractVector) where {T}
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nlp, :neval_jac_lin)
  vals[1] = T(1)
  vals[2] = T(-2)
  return vals
end

function NLPModels.jac_nln_coord!(nlp::HS100, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nlp, :neval_jac_nln)
  vals[1] = -x[1] / 2
  vals[2] = -2 * x[2]
  return vals
end

function NLPModels.jprod_lin!(nlp::HS100, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 1 Jv
  @lencheck 2 x v
  increment!(nlp, :neval_jprod_lin)
  Jv[1] = v[1] - 2 * v[2]
  return Jv
end

function NLPModels.jprod_nln!(nlp::HS100, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 1 Jv
  @lencheck 2 x v
  increment!(nlp, :neval_jprod_nln)
  Jv[1] = -x[1] * v[1] / 2 - 2 * x[2] * v[2]
  return Jv
end

function NLPModels.jtprod!(nlp::HS100, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 2 v
  increment!(nlp, :neval_jtprod)
  Jtv[1] = v[1] - x[1] * v[2] / 2
  Jtv[2] = -2 * v[1] - 2 * x[2] * v[2]
  return Jtv
end

function NLPModels.jtprod_lin!(nlp::HS100, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_lin)
  Jtv[1] = v[1]
  Jtv[2] = -2 * v[1]
  return Jtv
end

function NLPModels.jtprod_nln!(nlp::HS100, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nlp, :neval_jtprod_nln)
  Jtv[1] = -x[1] * v[1] / 2
  Jtv[2] = -2 * x[2] * v[1]
  return Jtv
end

function NLPModels.jth_hprod!(
  nlp::HS100,
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
  nlp::HS100,
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
  nlp::HS100,
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
