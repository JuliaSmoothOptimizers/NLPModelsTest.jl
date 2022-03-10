export MGH01Feas

"""
    nlp = MGH01Feas()

## Rosenbrock function in feasibility format

    Source: Problem 1 in
    J.J. Moré, B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981

```math
\\begin{aligned}
\\min \\quad & 0 \\\\
\\text{s. to} \\quad & 1 - x_1 = 0 \\\\
& 10 (x_2 - x_1^2) = 0.
\\end{aligned}
```

Starting point: `[-1.2; 1]`.
"""
mutable struct MGH01Feas{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function MGH01Feas(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    2,
    x0 = T[-1.2; 1.0],
    name = "MGH01Feas_manual",
    ncon = 2,
    lcon = zeros(T, 2),
    ucon = zeros(T, 2),
    nnzj = 3,
    nnzh = 1,
    lin = 1:1,
    lin_nnzj = 1,
    nln_nnzj = 2,
  )

  return MGH01Feas(meta, Counters())
end
MGH01Feas() = MGH01Feas(Float64)

function NLPModels.obj(nlp::MGH01Feas, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  return zero(eltype(x))
end

function NLPModels.grad!(nlp::MGH01Feas, x::AbstractVector{T}, gx::AbstractVector{T}) where {T}
  @lencheck 2 x gx
  increment!(nlp, :neval_grad)
  gx .= 0
  return gx
end

function NLPModels.cons_lin!(nls::MGH01Feas, x::AbstractVector, cx::AbstractVector)
  @lencheck 1 cx
  @lencheck 2 x
  increment!(nls, :neval_cons_lin)
  cx .= [1 - x[1]]
  return cx
end

function NLPModels.cons_nln!(nls::MGH01Feas, x::AbstractVector, cx::AbstractVector)
  @lencheck 1 cx
  @lencheck 2 x
  increment!(nls, :neval_cons_nln)
  cx .= [10 * (x[2] - x[1]^2)]
  return cx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_lin_structure!(
  nls::MGH01Feas,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 1 rows cols
  rows .= [1]
  cols .= [1]
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nls::MGH01Feas,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_lin_coord!(nls::MGH01Feas, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 1 vals
  increment!(nls, :neval_jac_lin)
  vals .= [-1]
  return vals
end

function NLPModels.jac_nln_coord!(nls::MGH01Feas, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 2 vals
  increment!(nls, :neval_jac_nln)
  vals .= [-20x[1], 10]
  return vals
end

function NLPModels.jprod_lin!(nls::MGH01Feas, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nls, :neval_jprod_lin)
  Jv .= [-v[1]]
  return Jv
end

function NLPModels.jprod_nln!(nls::MGH01Feas, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  increment!(nls, :neval_jprod_nln)
  Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod!(
  nls::MGH01Feas,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod)
  Jtv .= [-v[1] - 20 * x[1] * v[2]; 10 * v[2]]
  return Jtv
end

function NLPModels.jtprod_lin!(
  nls::MGH01Feas,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nls, :neval_jtprod_lin)
  Jtv .= [-v[1]; 0]
  return Jtv
end

function NLPModels.jtprod_nln!(
  nls::MGH01Feas,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x Jtv
  @lencheck 1 v
  increment!(nls, :neval_jtprod_nln)
  Jtv .= [-20 * x[1] * v[1]; 10 * v[1]]
  return Jtv
end

function NLPModels.hess_structure!(
  nls::MGH01Feas,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function NLPModels.hess_coord!(
  nls::MGH01Feas,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck 2 x y
  @lencheck 1 vals
  increment!(nls, :neval_hess)
  vals[1] = -20y[2]
  return vals
end

function NLPModels.hprod!(
  nls::MGH01Feas,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck 2 x y v Hv
  increment!(nls, :neval_hprod)
  Hv .= [-20y[2] * v[1]; 0]
  return Hv
end

function NLPModels.jth_hprod!(
  nls::MGH01Feas,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 2 x v Hv
  @rangecheck 1 2 j
  NLPModels.increment!(nls, :neval_jhprod)
  if j == 1
    Hv .= zero(T)
  elseif j == 2
    Hv .= [-20v[1]; zero(T)]
  end
  return Hv
end

function NLPModels.jth_hess_coord!(
  nls::MGH01Feas,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 1 vals
  @lencheck 2 x
  @rangecheck 1 2 j
  NLPModels.increment!(nls, :neval_jhess)
  if j == 1
    vals .= zero(T)
  elseif j == 2
    vals .= [T(-20)]
  end
  return vals
end

function NLPModels.ghjvprod!(
  nls::MGH01Feas,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nls.meta.nvar x g v
  @lencheck nls.meta.ncon gHv
  increment!(nls, :neval_hprod)
  gHv .= [T(0); -g[1] * 20v[1]]
  return gHv
end
