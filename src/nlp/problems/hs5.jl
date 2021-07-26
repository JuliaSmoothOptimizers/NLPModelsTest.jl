export HS5

"""
    nlp = HS5()

## Problem 5 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & \\sin(x_1 + x_2) + (x_1 - x_2)^2 - \\tfrac{3}{2}x_1 + \\tfrac{5}{2}x_2 + 1 \\\\
\\text{s. to} \\quad & -1.5 \\leq x_1 \\leq 4 \\\\
& -3 \\leq x_2 \\leq 3
\\end{aligned}
```

Starting point: `[0.0; 0.0]`.
"""
mutable struct HS5{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function HS5(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    x0 = zeros(T, 2),
    lvar = T[-1.5; -3],
    uvar = T[4; 3],
    name = "HS5_manual",
  )

  return HS5(meta, Counters())
end
HS5() = HS5(Float64)

function NLPModels.obj(nlp::HS5, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1
end

function NLPModels.grad!(nlp::HS5, x::AbstractVector{T}, gx::AbstractVector{T}) where {T}
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= cos(x[1] + x[2]) * ones(T, 2) + 2 * (x[1] - x[2]) * T[1; -1] + T[-1.5; 2.5]
  return gx
end

function NLPModels.hess_structure!(nlp::HS5, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 3 rows cols
  rows .= [1; 2; 2]
  cols .= [1; 1; 2]
  return rows, cols
end

function NLPModels.hess_coord!(nlp::HS5, x::AbstractVector, vals::AbstractVector; obj_weight = 1.0)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nlp, :neval_hess)
  vals[1] = vals[3] = -sin(x[1] + x[2]) + 2
  vals[2] = -sin(x[1] + x[2]) - 2
  vals .*= obj_weight
  return vals
end

function NLPModels.hprod!(
  nlp::HS5,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  increment!(nlp, :neval_hprod)
  Hv .=
    (-sin(x[1] + x[2]) * (v[1] + v[2]) * ones(T, 2) + 2 * [v[1] - v[2]; v[2] - v[1]]) * obj_weight
  return Hv
end

function NLPModels.cons!(nlp::HS5, x::AbstractVector{T}, c) where {T}
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  return c
end
function NLPModels.jac_structure!(
  nlp::HS5,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T}
  @lencheck nlp.meta.nnzj rows cols
  return rows, cols
end
function NLPModels.jac_coord!(nlp::HS5, x::AbstractVector{T}, vals) where {T}
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  return vals
end
function NLPModels.jprod!(nlp::HS5, x::AbstractVector{T}, v, Jv) where {T}
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  return Jv
end
function NLPModels.jtprod!(nlp::HS5, x::AbstractVector{T}, v, Jtv) where {T}
  @lencheck nlp.meta.nvar Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  fill!(Jtv, zero(T))
  return Jtv
end
function NLPModels.hess_coord!(nlp::HS5, x, y, vals; obj_weight=1.0)
  return hess_coord!(nlp, x, vals; obj_weight=obj_weight)
end
NLPModels.hprod!(nlp::HS5, x, y, v, Hv; obj_weight=1.0) = hprod!(nlp, x, v, Hv; obj_weight=obj_weight)
