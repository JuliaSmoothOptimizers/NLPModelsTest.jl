export coord_memory_nlp

"""
    coord_memory_nlp(nlp; exclude = [])

Check that the allocated memory for in place coord methods is
sufficiently smaller than their allocating counter parts.
"""
function coord_memory_nlp(nlp::AbstractNLPModel; exclude = [jth_hess_coord])
  n = nlp.meta.nvar
  m = nlp.meta.ncon

  x = 10 * [-(-1.0)^i for i = 1:n]
  y = [-(-1.0)^i for i = 1:m]

  # Hessian unconstrained test
  if hess_coord ∉ exclude
    vals = hess_coord(nlp, x)
    al1 = @allocated hess_coord(nlp, x)
    V = zeros(nlp.meta.nnzh)
    hess_coord!(nlp, x, V)
    al2 = @allocated hess_coord!(nlp, x, V)
    @test al2 < al1 - 50
  end

  if m > 0
    if hess_coord ∉ exclude
      vals = hess_coord(nlp, x, y)
      al1 = @allocated vals = hess_coord(nlp, x, y)
      hess_coord!(nlp, x, y, V)
      al2 = @allocated hess_coord!(nlp, x, y, V)
      @test al2 < al1 - 50
    end

    if jth_hess_coord ∉ exclude
      vals = jth_hess_coord(nlp, x, m)
      al1 = @allocated vals = jth_hess_coord(nlp, x, m)
      jth_hess_coord!(nlp, x, m, V)
      al2 = @allocated jth_hess_coord!(nlp, x, m, V)
      @test al2 < al1 - 50
    end

    if jac_coord ∉ exclude
      vals = jac_coord(nlp, x)
      al1 = @allocated vals = jac_coord(nlp, x)
      V = zeros(nlp.meta.nnzj)
      jac_coord!(nlp, x, vals)
      al2 = @allocated jac_coord!(nlp, x, vals)
      @test al2 < al1 - 50
    end
  end
end
