export multiple_precision_nlp

function multiple_precision_nlp(
  problem :: String;
  kwargs...
)
  Base.depwarn("This function signature will be deprecated, see the help for the new signature", :multiple_precision_nlp)
  nlp_from_T = eval(Symbol(problem))
  multiple_precision_nlp(nlp_from_T; kwargs...)
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
  precisions::Array = [Float16, Float32, Float64, BigFloat],
  exclude = [jth_hess, jth_hess_coord, jth_hprod, ghjvprod],
)
  for T in precisions
    nlp = nlp_from_T(T)
    x = ones(T, nlp.meta.nvar)
    @test obj ∈ exclude || typeof(obj(nlp, x)) == T
    @test grad ∈ exclude || eltype(grad(nlp, x)) == T
    @test hess ∈ exclude || eltype(hess(nlp, x)) == T
    @test hess_op ∈ exclude || eltype(hess_op(nlp, x)) == T
    if hess_coord ∉ exclude && hess_op ∉ exclude
      rows, cols = hess_structure(nlp)
      vals = hess_coord(nlp, x)
      @test eltype(vals) == T
      Hv = zeros(T, nlp.meta.nvar)
      @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
    end
    if nlp.meta.ncon > 0
      y = ones(T, nlp.meta.ncon)
      @test cons ∈ exclude || eltype(cons(nlp, x)) == T
      @test jac ∈ exclude || eltype(jac(nlp, x)) == T
      @test jac_op ∈ exclude || eltype(jac_op(nlp, x)) == T
      if jac_coord ∉ exclude && jac_op ∉ exclude
        rows, cols = jac_structure(nlp)
        vals = jac_coord(nlp, x)
        @test eltype(vals) == T
        Av = zeros(T, nlp.meta.ncon)
        Atv = zeros(T, nlp.meta.nvar)
        @test eltype(jac_op!(nlp, rows, cols, vals, Av, Atv)) == T
      end
      @test hess ∈ exclude || eltype(hess(nlp, x, y)) == T
      @test hess ∈ exclude || eltype(hess(nlp, x, y, obj_weight = one(T))) == T
      @test hess_op ∈ exclude || eltype(hess_op(nlp, x, y)) == T
      if hess_coord ∉ exclude && hess_op ∉ exclude
        rows, cols = hess_structure(nlp)
        vals = hess_coord(nlp, x, y)
        @test eltype(vals) == T
        Hv = zeros(T, nlp.meta.nvar)
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
