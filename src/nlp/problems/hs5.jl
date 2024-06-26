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

function HS5(::Type{S}) where {S}
  meta = NLPModelMeta{eltype(S), S}(
    2,
    x0 = fill!(S(undef, 2), 0),
    lvar = S([-15 // 10; -3]),
    uvar = S([4; 3]),
    name = "HS5_manual",
  )
  return HS5(meta, Counters())
end
HS5() = HS5(Float64)
HS5(::Type{T}) where {T <: Number} = HS5(Vector{T})

function NLPModels.obj(nlp::HS5, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1
end

function NLPModels.grad!(nlp::HS5, x::AbstractVector{T}, gx::AbstractVector{T}) where {T}
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx[1] = cos(x[1] + x[2]) + 2 * (x[1] - x[2]) + T(-1.5)
  gx[2] = cos(x[1] + x[2]) - 2 * (x[1] - x[2]) + T(2.5)
  return gx
end

function NLPModels.hess_structure!(nlp::HS5, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 3 rows cols
  k = 0
  for j = 1:2, i = j:2
    k += 1
    rows[k] = i
    cols[k] = j
  end
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
  Hv[1] = (-sin(x[1] + x[2]) * (v[1] + v[2]) + 2 * (v[1] - v[2])) * obj_weight
  Hv[2] = (-sin(x[1] + x[2]) * (v[1] + v[2]) + 2 * (v[2] - v[1])) * obj_weight
  return Hv
end
