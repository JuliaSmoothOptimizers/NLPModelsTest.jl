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

function NLPModels.hess_structure!(
  nlp::BROWNDEN,
  rows::AbstractVector{Int},
  cols::AbstractVector{Int},
)
  @lencheck 10 rows cols
  n = nlp.meta.nvar
  k = 0
  for j = 1:n, i = j:n
    k += 1
    rows[k] = i
    cols[k] = j
  end
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::BROWNDEN,
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight = 1.0,
) where {T}
  @lencheck 4 x
  @lencheck 10 vals
  increment!(nlp, :neval_hess)
  α(x, i) = x[1] + x[2] * i / 5 - exp(i / 5)
  dα(x, i) = i / 5
  β(x, i) = x[3] + x[4] * sin(i / 5) - cos(i / 5)
  dβ(x, i) = sin(i / 5)
  θ(x, i) = α(x, i)^2 + β(x, i)^2
  dθ1(x, i) = 2 * α(x, i)
  dθ2(x, i) = 2 * α(x, i) * dα(x, i)
  dθ3(x, i) = 2 * β(x, i)
  dθ4(x, i) = 2 * β(x, i) * dβ(x, i)
  vals[1] = 4 * sum( θ(x, i) + α(x, i) * dθ1(x, i) for i=1:20)
  vals[2] = 4 * sum( i/5 * (θ(x, i) + α(x, i) * dθ1(x, i)) for i=1:20)
  vals[3] = 4 * sum( β(x, i) * dθ1(x, i) for i=1:20)
  vals[4] = 4 * sum( sin(i/5) * β(x, i) * dθ1(x, i) for i=1:20)
  vals[5] = 4 * sum( i/5 * (θ(x, i) * dα(x, i) + α(x, i) * dθ2(x, i)) for i=1:20)
  vals[6] = 4 * sum( β(x, i) * dθ2(x, i) for i=1:20)
  vals[7] = 4 * sum( sin(i/5) * β(x, i) * dθ2(x, i) for i=1:20)
  vals[8] = 4 * sum( θ(x, i) + β(x, i) * dθ3(x, i) for i=1:20)
  vals[9] = 4 * sum( sin(i/5) * (θ(x, i) + β(x, i) * dθ3(x, i)) for i=1:20)
  vals[10] = 4 * sum( sin(i/5) * (θ(x, i) * dβ(x, i) + β(x, i) * dθ4(x, i)) for i=1:20)
  vals .*= T(obj_weight)
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
    viv = v[1] + v[2] * i / 5 # dot([1; i / 5; 0; 0], v)
    wiv = v[3] + sin(i / 5) * v[4] # dot([0; 0; 1; sin(i / 5)], v)
    ziv = αi * viv + βi * wiv # dot(αi * [1; i / 5; 0; 0] + βi * [0; 0; 1; sin(i / 5)], v)
    θi = αi^2 + βi^2
    Hv[1] += obj_weight * ((4 * viv) * θi + 8 * ziv * αi)
    Hv[2] += obj_weight * ((4 * viv * i / 5) * θi + 8 * ziv * αi * i / 5)
    Hv[3] += obj_weight * ((4 * wiv) * θi + 8 * ziv * βi)
    Hv[4] += obj_weight * ((4 * wiv * sin(i / 5)) * θi + 8 * ziv * βi * sin(i / 5))
  end
  return Hv
end
