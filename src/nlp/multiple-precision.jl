export multiple_precision_nlp, multiple_precision_nlp_array

function multiple_precision_nlp(problem::String; kwargs...)
  Base.depwarn(
    "This function signature will be deprecated, see the help for the new signature",
    :multiple_precision_nlp,
  )
  nlp_from_T = eval(Symbol(problem))
  multiple_precision_nlp(nlp_from_T; kwargs...)
end

"""
    multiple_precision_nlp_array(nlp_from_T, ::Type{S}; precisions=[Float16, Float32, Float64])

Check that the NLP API functions output type are the same as the input.
It calls [`multiple_precision_nlp`](@ref) on problem type `T -> nlp_from_T(S(T))`.

The array `precisions` are the tested floating point types.
Note that `BigFloat` is not tested by default, because it is not supported by `CuArray`.
"""
function multiple_precision_nlp_array(
  nlp_from_T,
  ::Type{S};
  precisions::Array = [Float16, Float32, Float64],
  kwargs...,
) where {S}
  return multiple_precision_nlp(T -> nlp_from_T(S{T}), precisions = precisions; kwargs...)
end

"""
    multiple_precision_nlp(nlp_from_T; precisions=[...], exclude = [ghjvprod])

Check that the NLP API functions output type are the same as the input.
In other words, make sure that the model handles multiple precisions.

The input `nlp_from_T` is a function that returns an `nlp` from a type `T`.
The array `precisions` are the tested floating point types.
Defaults to `[Float16, Float32, Float64, BigFloat]`.
"""
function multiple_precision_nlp(
  nlp_from_T;
  linear_api = false,
  precisions::Array = [Float16, Float32, Float64, BigFloat],
  exclude = [jth_hess, jth_hess_coord, jth_hprod, ghjvprod],
)
  for T in precisions
    nlp = nlp_from_T(T)
    S = typeof(nlp.meta.x0)
    x = fill!(S(undef, nlp.meta.nvar), T(1))
    v = fill!(S(undef, nlp.meta.nvar), T(2))
    @test obj ∈ exclude || typeof(obj(nlp, x)) == T
    @test grad ∈ exclude || typeof(grad(nlp, x)) == S
    @test hess ∈ exclude || eltype(hess(nlp, x)) == T
    @test hess_op ∈ exclude || eltype(hess_op(nlp, x)) == T
    @test hprod ∈ exclude || typeof(hprod(nlp, x, v)) == S
    if hess_coord ∉ exclude && hess_op ∉ exclude
      rows, cols = hess_structure(nlp)
      vals = hess_coord(nlp, x)
      @test typeof(vals) == S
      Hv = fill!(S(undef, nlp.meta.nvar), T(1))
      @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
    end
    if nlp.meta.ncon > 0
      y = fill!(S(undef, nlp.meta.ncon), T(1))
      @test cons ∈ exclude || typeof(cons(nlp, x)) == S
      @test jac ∈ exclude || eltype(jac(nlp, x)) == T
      @test jac_op ∈ exclude || eltype(jac_op(nlp, x)) == T
      @test jprod ∈ exclude || typeof(jprod(nlp, x, v)) == S
      @test jtprod ∈ exclude || typeof(jtprod(nlp, x, y)) == S
      if linear_api && nlp.meta.nnln > 0
        @test cons ∈ exclude || typeof(cons_nln(nlp, x)) == S
        @test jac ∈ exclude || eltype(jac_nln(nlp, x)) == T
        @test jac_op ∈ exclude || eltype(jac_nln_op(nlp, x)) == T
      end
      if linear_api && nlp.meta.nlin > 0
        @test cons ∈ exclude || typeof(cons_lin(nlp, x)) == S
        @test jac ∈ exclude || eltype(jac_lin(nlp, x)) == T
        @test jac_op ∈ exclude || eltype(jac_lin_op(nlp, x)) == T
      end
      if jac_coord ∉ exclude && jac_op ∉ exclude
        rows, cols = jac_structure(nlp)
        vals = jac_coord(nlp, x)
        @test typeof(vals) == S
        Av = fill!(S(undef, nlp.meta.ncon), T(0))
        Atv = fill!(S(undef, nlp.meta.nvar), T(0))
        @test eltype(jac_op!(nlp, rows, cols, vals, Av, Atv)) == T
        if linear_api && nlp.meta.nnln > 0
          rows, cols = jac_nln_structure(nlp)
          vals = jac_nln_coord(nlp, x)
          @test typeof(vals) == S
          Av = fill!(S(undef, nlp.meta.nnln), T(0))
          Atv = fill!(S(undef, nlp.meta.nvar), T(0))
          @test eltype(jac_nln_op!(nlp, rows, cols, vals, Av, Atv)) == T
        end
        if linear_api && nlp.meta.nlin > 0
          rows, cols = jac_lin_structure(nlp)
          vals = jac_lin_coord(nlp, x)
          @test typeof(vals) == S
          Av = fill!(S(undef, nlp.meta.nlin), T(0))
          Atv = fill!(S(undef, nlp.meta.nvar), T(0))
          @test eltype(jac_lin_op!(nlp, rows, cols, vals, Av, Atv)) == T
        end
      end
      @test hess ∈ exclude || eltype(hess(nlp, x, y)) == T
      @test hess ∈ exclude || eltype(hess(nlp, x, y, obj_weight = one(T))) == T
      @test hess_op ∈ exclude || eltype(hess_op(nlp, x, y)) == T
      @test hprod ∈ exclude || typeof(hprod(nlp, x, y, v)) == S
      if hess_coord ∉ exclude && hess_op ∉ exclude
        rows, cols = hess_structure(nlp)
        vals = hess_coord(nlp, x, y)
        @test typeof(vals) == S
        Hv = fill!(S(undef, nlp.meta.nvar), T(0))
        @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
      end
      @test jth_hess ∈ exclude || eltype(jth_hess(nlp, x, 1)) == T
      @test jth_hess_coord ∈ exclude || eltype(jth_hess_coord(nlp, x, 1)) == T
      @test jth_hprod ∈ exclude || eltype(jth_hprod!(nlp, x, x, 1, Hv)) == T
      @test jth_hprod ∈ exclude || eltype(jth_hprod(nlp, x, x, 1)) == T
      @test ghjvprod ∈ exclude || eltype(ghjvprod(nlp, x, x, x)) == T
    end
  end
end
