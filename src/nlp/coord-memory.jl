export coord_memory_nlp

"""
    coord_memory_nlp(nlp; linear_api = false, exclude = [])

Check that the allocated memory for in place coord methods is
sufficiently smaller than their allocating counter parts.
"""
function coord_memory_nlp(nlp::AbstractNLPModel; linear_api = false, exclude = [jth_hess_coord])
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
    @test (al2 < al1) | (al2 == 0)
  end

  if m > 0
    if hess_coord ∉ exclude
      vals = hess_coord(nlp, x, y)
      al1 = @allocated vals = hess_coord(nlp, x, y)
      hess_coord!(nlp, x, y, V)
      al2 = @allocated hess_coord!(nlp, x, y, V)
      @test (al2 < al1) | (al2 == 0)
    end

    if jth_hess_coord ∉ exclude
      vals = jth_hess_coord(nlp, x, m)
      al1 = @allocated vals = jth_hess_coord(nlp, x, m)
      jth_hess_coord!(nlp, x, m, V)
      al2 = @allocated jth_hess_coord!(nlp, x, m, V)
      @test (al2 < al1) | (al2 == 0)
    end

    if jac_coord ∉ exclude
      vals = jac_coord(nlp, x)
      al1 = @allocated vals = jac_coord(nlp, x)
      V = zeros(nlp.meta.nnzj)
      jac_coord!(nlp, x, vals)
      al2 = @allocated jac_coord!(nlp, x, vals)
      @test (al2 < al1) | (al2 == 0)
      if linear_api && nlp.meta.nlin > 0
        vals = jac_lin_coord(nlp, x)
        al1 = @allocated vals = jac_lin_coord(nlp, x)
        V = zeros(nlp.meta.lin_nnzj)
        jac_lin_coord!(nlp, x, vals)
        al2 = @allocated jac_lin_coord!(nlp, x, vals)
        @test (al2 < al1) | (al2 == 0)
      end
      if linear_api && nlp.meta.nnln > 0
        vals = jac_nln_coord(nlp, x)
        al1 = @allocated vals = jac_nln_coord(nlp, x)
        V = zeros(nlp.meta.nln_nnzj)
        jac_nln_coord!(nlp, x, vals)
        al2 = @allocated jac_nln_coord!(nlp, x, vals)
        @test (al2 < al1) | (al2 == 0)
      end
    end
  end
end
