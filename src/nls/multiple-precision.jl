export multiple_precision_nls, multiple_precision_nls_array

function multiple_precision_nls(problem::String; kwargs...)
  Base.depwarn(
    "This function signature will be deprecated, see the help for the new signature",
    :multiple_precision_nls,
  )
  nls_from_T = eval(Symbol(problem))
  multiple_precision_nls(nls_from_T; kwargs...)
end

"""
    multiple_precision_nls_array(nlp_from_T, ::Type{S}; precisions=[Float16, Float32, Float64])

Check that the NLS API functions output type are the same as the input.
It calls [`multiple_precision_nls`](@ref) on problem type `T -> nlp_from_T(S(T))`.

The array `precisions` are the tested floating point types.
Note that `BigFloat` is not tested by default, because it is not supported by `CuArray`.
"""
function multiple_precision_nls_array(
  nlp_from_T,
  ::Type{S};
  precisions::Array = [Float16, Float32, Float64],
  kwargs...,
) where {S}
  return multiple_precision_nls(T -> nlp_from_T(S{T}), precisions = precisions; kwargs...)
end

"""
    multiple_precision_nls(nls_from_T; precisions=[...], exclude = [])

Check that the NLS API functions output type are the same as the input.
In other words, make sure that the model handles multiple precisions.

The input `nls_from_T` is a function that returns an `nls` from a type `T`.
The array `precisions` are the tested floating point types.
Defaults to `[Float16, Float32, Float64, BigFloat]`.
"""
function multiple_precision_nls(
  nls_from_T;
  linear_api = false,
  precisions::Array = [Float16, Float32, Float64, BigFloat],
  exclude = [],
)
  for T in precisions
    nls = nls_from_T(T)
    S = typeof(nls.meta.x0)
    x = fill!(S(undef, nls.meta.nvar), 1)
    v = fill!(S(undef, nls.meta.nvar), 2)
    y = fill!(S(undef, nls.meta.ncon), 2)
    w = fill!(S(undef, nls.nls_meta.nequ), 2)
    @test residual ∈ exclude || typeof(residual(nls, x)) == S
    @test jac_residual ∈ exclude || eltype(jac_residual(nls, x)) == T
    @test jac_op_residual ∈ exclude || eltype(jac_op_residual(nls, x)) == T
    @test jprod_residual ∈ exclude || typeof(jprod_residual(nls, x, v)) == S
    @test jtprod_residual ∈ exclude || typeof(jtprod_residual(nls, x, w)) == S
    if jac_coord_residual ∉ exclude && jac_op_residual ∉ exclude
      rows, cols = jac_structure_residual(nls)
      vals = jac_coord_residual(nls, x)
      @test typeof(vals) == S
      Av = fill!(S(undef, nls.nls_meta.nequ), 0)
      Atv = fill!(S(undef, nls.meta.nvar), 0)
      @test eltype(jac_op_residual!(nls, rows, cols, vals, Av, Atv)) == T
    end
    @test hess_residual ∈ exclude || eltype(hess_residual(nls, x, ones(T, nls.nls_meta.nequ))) == T
    if hess_op_residual ∉ exclude
      for i = 1:(nls.nls_meta.nequ)
        @test eltype(hess_op_residual(nls, x, i)) == T
      end
    end
    if hprod_residual ∉ exclude
      for i = 1:(nls.nls_meta.nequ)
        @test typeof(hprod_residual(nls, x, i, v)) == S
      end
    end
    @test obj ∈ exclude || typeof(obj(nls, x)) == T
    @test grad ∈ exclude || typeof(grad(nls, x)) == S
    if nls.meta.ncon > 0
      @test cons ∈ exclude || typeof(cons(nls, x)) == S
      @test jac ∈ exclude || eltype(jac(nls, x)) == T
      @test jac_op ∈ exclude || eltype(jac_op(nls, x)) == T
      if linear_api && nls.meta.nnln > 0
        @test cons ∈ exclude || typeof(cons_nln(nls, x)) == S
        @test jac ∈ exclude || eltype(jac_nln(nls, x)) == T
        @test jac_op ∈ exclude || eltype(jac_nln_op(nls, x)) == T
      end
      if linear_api && nls.meta.nlin > 0
        @test cons ∈ exclude || typeof(cons_lin(nls, x)) == S
        @test jac ∈ exclude || eltype(jac_lin(nls, x)) == T
        @test jac_op ∈ exclude || eltype(jac_lin_op(nls, x)) == T
      end
      if jac_coord ∉ exclude && jac_op ∉ exclude
        rows, cols = jac_structure(nls)
        vals = jac_coord(nls, x)
        @test typeof(vals) == S
        Av = fill!(S(undef, nls.meta.ncon), 0)
        Atv = fill!(S(undef, nls.meta.nvar), 0)
        @test eltype(jac_op!(nls, rows, cols, vals, Av, Atv)) == T
        @test jprod ∈ exclude || typeof(jprod(nls, x, v)) == S
        @test jtprod ∈ exclude || typeof(jtprod(nls, x, y)) == S
        if linear_api && nls.meta.nnln > 0
          rows, cols = jac_nln_structure(nls)
          vals = jac_nln_coord(nls, x)
          @test typeof(vals) == S
          Av = fill!(S(undef, nls.meta.nnln), 0)
          Atv = fill!(S(undef, nls.meta.nvar), 0)
          @test eltype(jac_nln_op!(nls, rows, cols, vals, Av, Atv)) == T
        end
        if linear_api && nls.meta.nlin > 0
          rows, cols = jac_lin_structure(nls)
          vals = jac_lin_coord(nls, x)
          @test typeof(vals) == S
          Av = fill!(S(undef, nls.meta.nlin), 0)
          Atv = fill!(S(undef, nls.meta.nvar), 0)
          @test eltype(jac_lin_op!(nls, rows, cols, vals, Av, Atv)) == T
        end
      end
    end
  end
end
