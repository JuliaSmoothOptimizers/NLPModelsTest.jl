export BROWNDEN

using LinearAlgebra

"""
    nlp = BROWNDEN()

## Brown and Dennis function.

    Source: Problem 16 in
    J.J. Moré, B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981

    classification SUR2-AN-4-0

```math
\\min_x \\ \\sum_{i=1}^{20} \\left(\\left(x_1 + \\tfrac{i}{5} x_2 - e^{i / 5}\\right)^2
+ \\left(x_3 + \\sin(\\tfrac{i}{5}) x_4 - \\cos(\\tfrac{i}{5})\\right)^2\\right)^2
```

Starting point: `[25.0; 5.0; -5.0; -1.0]`
"""
mutable struct BROWNDEN{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function BROWNDEN(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(4, x0 = T[25; 5; -5; -1], name = "BROWNDEN_manual", nnzh = 10)

  return BROWNDEN(meta, Counters())
end
BROWNDEN() = BROWNDEN(Float64)

function NLPModels.obj(nlp::BROWNDEN, x::AbstractVector{T}) where {T}
  @lencheck 4 x
  increment!(nlp, :neval_obj)
  return sum(
    (
      (x[1] + x[2] * T(i) / 5 - exp(T(i) / 5))^2 + (x[3] + x[4] * sin(T(i) / 5) - cos(T(i) / 5))^2
    )^2 for i = 1:20
  )
end

function NLPModels.grad!(nlp::BROWNDEN, x::AbstractVector, gx::AbstractVector)
  @lencheck 4 x gx
  increment!(nlp, :neval_grad)
  α(x, i) = x[1] + x[2] * i / 5 - exp(i / 5)
  β(x, i) = x[3] + x[4] * sin(i / 5) - cos(i / 5)
  θ(x, i) = α(x, i)^2 + β(x, i)^2
  gx[1] = sum(4 * θ(x, i) * (α(x, i)) for i = 1:20)
  gx[2] = sum(4 * θ(x, i) * (α(x, i) * i / 5) for i = 1:20)
  gx[3] = sum(4 * θ(x, i) * (β(x, i)) for i = 1:20)
  gx[4] = sum(4 * θ(x, i) * (β(x, i) * sin(i / 5)) for i = 1:20)
  return gx
end

function NLPModels.hess(nlp::BROWNDEN, x::AbstractVector{T}; obj_weight = 1.0) where {T}
  @lencheck 4 x
  increment!(nlp, :neval_hess)
  α(x, i) = x[1] + x[2] * T(i) / 5 - exp(T(i) / 5)
  β(x, i) = x[3] + x[4] * sin(T(i) / 5) - cos(T(i) / 5)
  Hx = zeros(T, 4, 4)
  if obj_weight == 0
    return Hx
  end
  for i = 1:20
    αi, βi = α(x, i), β(x, i)
    vi, wi = T[1; i / 5; 0; 0], T[0; 0; 1; sin(i / 5)]
    zi = αi * vi + βi * wi
    θi = αi^2 + βi^2
    Hx += (4vi * vi' + 4wi * wi') * θi + 8zi * zi'
  end
  return Symmetric(T(obj_weight) * Hx, :L)
end

function NLPModels.hess_structure!(
  nlp::BROWNDEN,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 10 rows cols
  n = nlp.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::BROWNDEN,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight = 1.0,
)
  @lencheck 4 x
  @lencheck 10 vals
  Hx = hess(nlp, x, obj_weight = obj_weight)
  k = 1
  for j = 1:4
    for i = j:4
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.hprod!(
  nlp::BROWNDEN,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 4 x v Hv
  increment!(nlp, :neval_hprod)
  α(x, i) = x[1] + x[2] * i / 5 - exp(i / 5)
  β(x, i) = x[3] + x[4] * sin(i / 5) - cos(i / 5)
  Hv .= 0
  if obj_weight == 0
    return Hv
  end
  for i = 1:20
    αi, βi = α(x, i), β(x, i)
    vi, wi = [1; i / 5; 0; 0], [0; 0; 1; sin(i / 5)]
    zi = αi * vi + βi * wi
    θi = αi^2 + βi^2
    Hv .+= obj_weight * ((4 * dot(vi, v) * vi + 4 * dot(wi, v) * wi) * θi + 8 * dot(zi, v) * zi)
  end
  return Hv
end
