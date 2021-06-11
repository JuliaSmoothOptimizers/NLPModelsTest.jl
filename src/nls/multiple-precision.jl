export multiple_precision_nls

"""
    multiple_precision_nls(nls; precisions=[...], exclude = [])

Check that the NLS API functions output type are the same as the input.
In other words, make sure that the model handles multiple precisions.

The array `precisions` are the tested floating point types.
Defaults to `[Float16, Float32, Float64, BigFloat]`.
"""
function multiple_precision_nls(
  nls::AbstractNLSModel;
  precisions::Array = [Float16, Float32, Float64, BigFloat],
  exclude = [],
)
  for T in precisions
    x = ones(T, nls.meta.nvar)
    @test residual ∈ exclude || eltype(residual(nls, x)) == T
    @test jac_residual ∈ exclude || eltype(jac_residual(nls, x)) == T
    @test jac_op_residual ∈ exclude || eltype(jac_op_residual(nls, x)) == T
    if jac_coord_residual ∉ exclude && jac_op_residual ∉ exclude
      rows, cols = jac_structure_residual(nls)
      vals = jac_coord_residual(nls, x)
      @test eltype(vals) == T
      Av = zeros(T, nls.nls_meta.nequ)
      Atv = zeros(T, nls.meta.nvar)
      @test eltype(jac_op_residual!(nls, rows, cols, vals, Av, Atv)) == T
    end
    @test hess_residual ∈ exclude || eltype(hess_residual(nls, x, ones(T, nls.nls_meta.nequ))) == T
    if hess_op_residual ∉ exclude
      for i = 1:(nls.nls_meta.nequ)
        @test eltype(hess_op_residual(nls, x, i)) == T
      end
    end
    @test obj ∈ exclude || typeof(obj(nls, x)) == T
    @test grad ∈ exclude || eltype(grad(nls, x)) == T
    if nls.meta.ncon > 0
      @test cons ∈ exclude || eltype(cons(nls, x)) == T
      @test jac ∈ exclude || eltype(jac(nls, x)) == T
      @test jac_op ∈ exclude || eltype(jac_op(nls, x)) == T
      if jac_coord ∉ exclude && jac_op ∉ exclude
        rows, cols = jac_structure(nls)
        vals = jac_coord(nls, x)
        @test eltype(vals) == T
        Av = zeros(T, nls.meta.ncon)
        Atv = zeros(T, nls.meta.nvar)
        @test eltype(jac_op!(nls, rows, cols, vals, Av, Atv)) == T
      end
    end
  end
end
