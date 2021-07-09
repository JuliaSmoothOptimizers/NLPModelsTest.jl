export NLSLC

"""
    nls = NLSLC()

## Linearly constrained nonlinear least squares problem

```math
\\begin{aligned}
\\min \\quad & \\tfrac{1}{2}\\| F(x) \\|^2 \\\\
\\text{s. to} \\quad & x_{15} = 0 \\\\
& x_{10} + 2x_{11} + 3x_{12} \\geq 1 \\\\
& x_{13} - x_{14} \\leq 16 \\\\
& -11 \\leq 5x_8 - 6x_9 \\leq 9 \\\\
& -2x_7 = -1 \\\\
& 4x_6 = 1 \\\\
& x_1 + 2x_2 \\geq -5 \\\\
& 3x_1 + 4x_2 \\geq -6 \\\\
& 9x_3 \\leq 1 \\\\
& 12x_4 \\leq 2 \\\\
& 15x_5 \\leq 3
\\end{aligned}
```
where
```math
F(x) = \\begin{bmatrix}
x_1^2 - 1 \\\\
x_2^2 - 2^2 \\\\
\\vdots \\\\
x_{15}^2 - 15^2
\\end{bmatrix}
```

Starting point: `zeros(15)`.
"""
mutable struct NLSLC{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
end

function NLSLC(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(
    15,
    nnzj = 17,
    ncon = 11,
    x0 = zeros(T, 15),
    lcon = T[22.0; 1.0; -Inf; -11.0; -1.0; 1.0; -5.0; -6.0; -Inf * ones(3)],
    ucon = T[22.0; Inf; 16.0; 9.0; -1.0; 1.0; Inf * ones(2); 1.0; 2.0; 3.0],
    name = "NLSLINCON",
  )
  nls_meta = NLSMeta{T, Vector{T}}(15, 15, nnzj = 15, nnzh = 15)

  return NLSLC(meta, nls_meta, NLSCounters())
end
NLSLC() = NLSLC(Float64)

function NLPModels.residual!(nls::NLSLC, x::AbstractVector, Fx::AbstractVector)
  @lencheck 15 x Fx
  increment!(nls, :neval_residual)
  Fx .= [x[i]^2 - i^2 for i = 1:(nls.nls_meta.nequ)]
  return Fx
end

function NLPModels.jac_structure_residual!(
  nls::NLSLC,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 15 rows cols
  for i = 1:(nls.nls_meta.nnzj)
    rows[i] = i
    cols[i] = i
  end
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::NLSLC, x::AbstractVector, vals::AbstractVector)
  @lencheck 15 x vals
  increment!(nls, :neval_jac_residual)
  vals .= [2 * x[i] for i = 1:(nls.nls_meta.nnzj)]
  return vals
end

function NLPModels.jprod_residual!(
  nls::NLSLC,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 15 x v Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= [2 * x[i] * v[i] for i = 1:(nls.nls_meta.nnzj)]
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::NLSLC,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 15 x v Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv .= [2 * x[i] * v[i] for i = 1:(nls.nls_meta.nnzj)]
  return Jtv
end

function NLPModels.hess_structure_residual!(
  nls::NLSLC,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 15 rows cols
  for i = 1:(nls.nls_meta.nnzh)
    rows[i] = i
    cols[i] = i
  end
  return rows, cols
end

function NLPModels.hess_coord_residual!(
  nls::NLSLC,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck 15 x v vals
  increment!(nls, :neval_hess_residual)
  vals .= [2 * v[i] for i = 1:(nls.nls_meta.nnzh)]
  return vals
end

function NLPModels.hprod_residual!(
  nls::NLSLC,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck 15 x v Hiv
  increment!(nls, :neval_hprod_residual)
  Hiv .= zero(eltype(x))
  Hiv[i] = 2 * v[i]
  return Hiv
end

function NLPModels.cons!(nls::NLSLC, x::AbstractVector, cx::AbstractVector)
  @lencheck 15 x
  @lencheck 11 cx
  increment!(nls, :neval_cons)
  cx .= [
    15 * x[15]
    [1; 2; 3]' * x[10:12]
    [1; -1]' * x[13:14]
    [5; 6]' * x[8:9]
    [0 -2; 4 0] * x[6:7]
    [1 2; 3 4] * x[1:2]
    diagm([3 * i for i = 3:5]) * x[3:5]
  ]
  return cx
end

function NLPModels.jac_structure!(
  nls::NLSLC,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 17 rows cols
  rows .= [1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9, 10, 11]
  cols .= [15, 10, 11, 12, 13, 14, 8, 9, 7, 6, 1, 2, 1, 2, 3, 4, 5]
  return rows, cols
end

function NLPModels.jac_coord!(nls::NLSLC, x::AbstractVector, vals::AbstractVector)
  @lencheck 15 x
  @lencheck 17 vals
  increment!(nls, :neval_jac)
  vals .= eltype(x).([15, 1, 2, 3, 1, -1, 5, 6, -2, 4, 1, 2, 3, 4, 9, 12, 15])
  return vals
end

function NLPModels.jprod!(nls::NLSLC, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 15 x v
  @lencheck 11 Jv
  increment!(nls, :neval_jprod)
  Jv[1] = 15 * v[15]
  Jv[2] = [1; 2; 3]' * v[10:12]
  Jv[3] = [1; -1]' * v[13:14]
  Jv[4] = [5; 6]' * v[8:9]
  Jv[5:6] = [0 -2; 4 0] * v[6:7]
  Jv[7:8] = [1.0 2; 3 4] * v[1:2]
  Jv[9:11] = diagm([3 * i for i = 3:5]) * v[3:5]
  return Jv
end

function NLPModels.jtprod!(nls::NLSLC, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 15 x Jtv
  @lencheck 11 v
  increment!(nls, :neval_jtprod)
  Jtv[1] = 1 * v[7] + 3 * v[8]
  Jtv[2] = 2 * v[7] + 4 * v[8]
  Jtv[3] = 9 * v[9]
  Jtv[4] = 12 * v[10]
  Jtv[5] = 15 * v[11]
  Jtv[6] = 4 * v[6]
  Jtv[7] = -2 * v[5]
  Jtv[8] = 5 * v[4]
  Jtv[9] = 6 * v[4]
  Jtv[10] = 1 * v[2]
  Jtv[11] = 2 * v[2]
  Jtv[12] = 3 * v[2]
  Jtv[13] = 1 * v[3]
  Jtv[14] = -1 * v[3]
  Jtv[15] = 15 * v[1]
  return Jtv
end

function NLPModels.hess(nls::NLSLC, x::AbstractVector{T}; obj_weight = 1.0) where {T}
  @lencheck 15 x
  increment!(nls, :neval_hess)
  return Symmetric(obj_weight * diagm(0 => [6 * x[i]^2 - 2 * i^2 for i = 1:15]), :L)
end

function NLPModels.hess(
  nls::NLSLC,
  x::AbstractVector{T},
  y::AbstractVector{T};
  obj_weight = 1.0,
) where {T}
  @lencheck 15 x
  @lencheck 11 y
  increment!(nls, :neval_hess)
  return Symmetric(hess(nls, x, obj_weight = obj_weight), :L)
end

function NLPModels.hess_structure!(nls::NLSLC, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 120 rows cols
  n = nls.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function NLPModels.hess_coord!(
  nls::NLSLC,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight = 1.0,
)
  @lencheck 15 x
  @lencheck 120 vals
  Hx = hess(nls, x, obj_weight = obj_weight)
  k = 1
  for j = 1:15
    for i = j:15
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.hess_coord!(
  nls::NLSLC,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = 1.0,
)
  @lencheck 15 x
  @lencheck 11 y
  @lencheck 120 vals
  Hx = hess(nls, x, y, obj_weight = obj_weight)
  k = 1
  for j = 1:15
    for i = j:15
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.hprod!(
  nls::NLSLC,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 15 x v Hv
  increment!(nls, :neval_hprod)
  Hv .= obj_weight * [(6 * x[i]^2 - 2 * i^2) * v[i] for i = 1:15]
  return Hv
end

function NLPModels.hprod!(
  nls::NLSLC,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 15 x v Hv
  @lencheck 11 y
  increment!(nls, :neval_hprod)
  return hprod!(nls, x, v, Hv, obj_weight = obj_weight)
end

function NLPModels.jth_hprod!(
  nls::NLSLC,
  x::AbstractVector{T},
  v::AbstractVector{T},
  j::Integer,
  Hv::AbstractVector{T},
) where {T}
  @lencheck 15 x v Hv
  @rangecheck 1 11 j
  NLPModels.increment!(nls, :neval_jhprod)
  Hv .= zero(T)
  return Hv
end

function NLPModels.jth_hess_coord!(
  nls::NLSLC,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  @lencheck 120 vals
  @lencheck 15 x
  @rangecheck 1 11 j
  NLPModels.increment!(nls, :neval_jhess)
  vals .= zero(T)
  return vals
end

function NLPModels.ghjvprod!(
  nls::NLSLC,
  x::AbstractVector{T},
  g::AbstractVector{T},
  v::AbstractVector{T},
  gHv::AbstractVector{T},
) where {T}
  @lencheck nls.meta.nvar x g v
  @lencheck nls.meta.ncon gHv
  increment!(nls, :neval_hprod)
  gHv .= zeros(T, nls.meta.ncon)
  return gHv
end
