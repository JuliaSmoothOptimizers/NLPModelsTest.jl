export check_nlp_dimensions

"""
    check_nlp_dimensions(nlp; exclude = [ghjvprod])

Make sure NLP API functions will throw DimensionError if the inputs are not the correct dimension.
To make this assertion in your code use

    @lencheck size input [more inputs separated by spaces]
"""
function check_nlp_dimensions(nlp; linear_api = false, exclude = [jth_hess, jth_hess_coord, jth_hprod, ghjvprod])
  n, m = nlp.meta.nvar, nlp.meta.ncon
  nnzh, nnzj = nlp.meta.nnzh, nlp.meta.nnzj

  x, badx = nlp.meta.x0, zeros(n + 1)
  v, badv = ones(n), zeros(n + 1)
  Hv, badHv = zeros(n), zeros(n + 1)
  hrows, badhrows = zeros(Int, nnzh), zeros(Int, nnzh + 1)
  hcols, badhcols = zeros(Int, nnzh), zeros(Int, nnzh + 1)
  hvals, badhvals = zeros(nnzh), zeros(nnzh + 1)
  if obj ∉ exclude
    @test_throws DimensionError obj(nlp, badx)
  end
  if grad ∉ exclude
    @test_throws DimensionError grad(nlp, badx)
    @test_throws DimensionError grad!(nlp, badx, v)
    @test_throws DimensionError grad!(nlp, x, badv)
  end
  if hprod ∉ exclude
    @test_throws DimensionError hprod(nlp, badx, v)
    @test_throws DimensionError hprod(nlp, x, badv)
    @test_throws DimensionError hprod!(nlp, badx, v, Hv)
    @test_throws DimensionError hprod!(nlp, x, badv, Hv)
    @test_throws DimensionError hprod!(nlp, x, v, badHv)
  end
  if hess_op ∉ exclude
    @test_throws DimensionError hess_op(nlp, badx)
    @test_throws DimensionError hess_op!(nlp, badx, Hv)
    @test_throws DimensionError hess_op!(nlp, x, badHv)
    @test_throws DimensionError hess_op!(nlp, badhrows, hcols, hvals, Hv)
    @test_throws DimensionError hess_op!(nlp, hrows, badhcols, hvals, Hv)
    @test_throws DimensionError hess_op!(nlp, hrows, hcols, badhvals, Hv)
    @test_throws DimensionError hess_op!(nlp, hrows, hcols, hvals, badHv)
  end
  if hess ∉ exclude
    @test_throws DimensionError hess(nlp, badx)
  end
  if hess_coord ∉ exclude
    @test_throws DimensionError hess_structure!(nlp, badhrows, hcols)
    @test_throws DimensionError hess_structure!(nlp, hrows, badhcols)
    @test_throws DimensionError hess_coord!(nlp, badx, hvals)
    @test_throws DimensionError hess_coord!(nlp, x, badhvals)
  end

  if m > 0
    lin_nnzj, nln_nnzj = nlp.meta.lin_nnzj, nlp.meta.nln_nnzj
    y, bady = nlp.meta.y0, zeros(m + 1)
    w, badw = ones(m), zeros(m + 1)
    w_lin, badw_lin = ones(nlp.meta.nlin), zeros(nlp.meta.nlin + 1)
    w_nln, badw_nln = ones(nlp.meta.nnln), zeros(nlp.meta.nnln + 1)
    Jv, badJv = zeros(m), zeros(m + 1)
    Jv_lin, badJv_lin = zeros(nlp.meta.nlin), zeros(nlp.meta.nlin + 1)
    Jv_nln, badJv_nln = zeros(nlp.meta.nnln), zeros(nlp.meta.nnln + 1)
    Jtw, badJtw = zeros(n), zeros(n + 1)
    jrows, badjrows = zeros(Int, nnzj), zeros(Int, nnzj + 1)
    jcols, badjcols = zeros(Int, nnzj), zeros(Int, nnzj + 1)
    jvals, badjvals = zeros(nnzj), zeros(nnzj + 1)
    jrows_lin, badjrows_lin = zeros(Int, lin_nnzj), zeros(Int, lin_nnzj + 1)
    jcols_lin, badjcols_lin = zeros(Int, lin_nnzj), zeros(Int, lin_nnzj + 1)
    jvals_lin, badjvals_lin = zeros(lin_nnzj), zeros(lin_nnzj + 1)
    jrows_nln, badjrows_nln = zeros(Int, nln_nnzj), zeros(Int, nln_nnzj + 1)
    jcols_nln, badjcols_nln = zeros(Int, nln_nnzj), zeros(Int, nln_nnzj + 1)
    jvals_nln, badjvals_nln = zeros(nln_nnzj), zeros(nln_nnzj + 1)
    if hprod ∉ exclude
      @test_throws DimensionError hprod(nlp, badx, y, v)
      @test_throws DimensionError hprod(nlp, x, bady, v)
      @test_throws DimensionError hprod(nlp, x, y, badv)
      @test_throws DimensionError hprod!(nlp, badx, y, v, Hv)
      @test_throws DimensionError hprod!(nlp, x, bady, v, Hv)
      @test_throws DimensionError hprod!(nlp, x, y, badv, Hv)
      @test_throws DimensionError hprod!(nlp, x, y, v, badHv)
    end
    if hess ∉ exclude
      @test_throws DimensionError hess(nlp, badx, y)
      @test_throws DimensionError hess(nlp, x, bady)
    end
    if hess_op ∉ exclude
      @test_throws DimensionError hess_op(nlp, badx, y)
      @test_throws DimensionError hess_op(nlp, x, bady)
      @test_throws DimensionError hess_op!(nlp, badx, y, Hv)
      @test_throws DimensionError hess_op!(nlp, x, bady, Hv)
      @test_throws DimensionError hess_op!(nlp, x, y, badHv)
    end
    if hess_coord ∉ exclude
      @test_throws DimensionError hess_coord!(nlp, badx, y, hvals)
      @test_throws DimensionError hess_coord!(nlp, x, bady, hvals)
      @test_throws DimensionError hess_coord!(nlp, x, y, badhvals)
    end
    if jth_hess ∉ exclude
      @test_throws DimensionError jth_hess(nlp, badx, 1)
    end
    if jth_hess_coord ∉ exclude
      @test_throws DimensionError jth_hess_coord!(nlp, badx, 1, hvals)
      @test_throws DimensionError jth_hess_coord!(nlp, x, 1, badhvals)
    end
    if jth_hprod ∉ exclude
      @test_throws DimensionError jth_hprod(nlp, badx, v, 1)
      @test_throws DimensionError jth_hprod(nlp, x, badv, 1)
      @test_throws DimensionError jth_hprod!(nlp, badx, v, 1, Hv)
      @test_throws DimensionError jth_hprod!(nlp, x, badv, 1, Hv)
      @test_throws DimensionError jth_hprod!(nlp, x, v, 1, badHv)
    end
    if ghjvprod ∉ exclude
      @test_throws DimensionError ghjvprod(nlp, badx, v, v)
      @test_throws DimensionError ghjvprod(nlp, x, badv, v)
      @test_throws DimensionError ghjvprod(nlp, x, v, badv)
    end
    if cons ∉ exclude
      @test_throws DimensionError cons(nlp, badx)
      @test_throws DimensionError cons!(nlp, badx, w)
      @test_throws DimensionError cons!(nlp, x, badw)

      if linear_api && nlp.meta.nlin > 0
        @test_throws DimensionError cons_lin(nlp, badx)
        @test_throws DimensionError cons_lin!(nlp, badx, w)
        @test_throws DimensionError cons_lin!(nlp, x, badw)
      end

      if linear_api && nlp.meta.nnln > 0
        @test_throws DimensionError cons_nln(nlp, badx)
        @test_throws DimensionError cons_nln!(nlp, badx, w)
        @test_throws DimensionError cons_nln!(nlp, x, badw)
      end
    end
    if jac ∉ exclude
      @test_throws DimensionError jac(nlp, badx)
      linear_api && nlp.meta.nlin > 0 && @test_throws DimensionError jac_lin(nlp, badx)
      linear_api && nlp.meta.nnln > 0 && @test_throws DimensionError jac_nln(nlp, badx)
    end
    if jprod ∉ exclude
      @test_throws DimensionError jprod(nlp, badx, v)
      @test_throws DimensionError jprod(nlp, x, badv)
      @test_throws DimensionError jprod!(nlp, badx, v, Jv)
      @test_throws DimensionError jprod!(nlp, x, badv, Jv)
      @test_throws DimensionError jprod!(nlp, x, v, badJv)

      if linear_api && nlp.meta.nlin > 0
        @test_throws DimensionError jprod_lin(nlp, badx, v)
        @test_throws DimensionError jprod_lin(nlp, x, badv)
        @test_throws DimensionError jprod_lin!(nlp, badx, v, Jv_lin)
        @test_throws DimensionError jprod_lin!(nlp, x, badv, Jv_lin)
        @test_throws DimensionError jprod_lin!(nlp, x, v, badJv_lin)
      end

      if linear_api && nlp.meta.nnln > 0
        @test_throws DimensionError jprod_nln(nlp, badx, v)
        @test_throws DimensionError jprod_nln(nlp, x, badv)
        @test_throws DimensionError jprod_nln!(nlp, badx, v, Jv_nln)
        @test_throws DimensionError jprod_nln!(nlp, x, badv, Jv_nln)
        @test_throws DimensionError jprod_nln!(nlp, x, v, badJv_nln)
      end
    end
    if jtprod ∉ exclude
      @test_throws DimensionError jtprod(nlp, badx, w)
      @test_throws DimensionError jtprod(nlp, x, badw)
      @test_throws DimensionError jtprod!(nlp, badx, w, Jtw)
      @test_throws DimensionError jtprod!(nlp, x, badw, Jtw)
      @test_throws DimensionError jtprod!(nlp, x, w, badJtw)

      if linear_api && nlp.meta.nlin > 0
        @test_throws DimensionError jtprod_lin(nlp, badx, w_lin)
        @test_throws DimensionError jtprod_lin(nlp, x, badw_lin)
        @test_throws DimensionError jtprod_lin!(nlp, badx, w_lin, Jtw)
        @test_throws DimensionError jtprod_lin!(nlp, x, badw_lin, Jtw)
        @test_throws DimensionError jtprod_lin!(nlp, x, w_lin, badJtw)
      end

      if linear_api && nlp.meta.nnln > 0
        @test_throws DimensionError jtprod_nln(nlp, badx, w_nln)
        @test_throws DimensionError jtprod_nln(nlp, x, badw_nln)
        @test_throws DimensionError jtprod_nln!(nlp, badx, w_nln, Jtw)
        @test_throws DimensionError jtprod_nln!(nlp, x, badw_nln, Jtw)
        @test_throws DimensionError jtprod_nln!(nlp, x, w_nln, badJtw)
      end
    end
    if jac_coord ∉ exclude
      @test_throws DimensionError jac_structure!(nlp, badjrows, jcols)
      @test_throws DimensionError jac_structure!(nlp, jrows, badjcols)
      @test_throws DimensionError jac_coord(nlp, badx)
      @test_throws DimensionError jac_coord!(nlp, badx, jvals)
      @test_throws DimensionError jac_coord!(nlp, x, badjvals)

      if linear_api && nlp.meta.nlin > 0
        @test_throws DimensionError jac_lin_structure!(nlp, badjrows_lin, jcols_lin)
        @test_throws DimensionError jac_lin_structure!(nlp, jrows_lin, badjcols_lin)
        @test_throws DimensionError jac_lin_coord(nlp, badx)
        @test_throws DimensionError jac_lin_coord!(nlp, badx, jvals_lin)
        @test_throws DimensionError jac_lin_coord!(nlp, x, badjvals_lin)
      end

      if linear_api && nlp.meta.nnln > 0
        @test_throws DimensionError jac_nln_structure!(nlp, badjrows_nln, jcols_nln)
        @test_throws DimensionError jac_nln_structure!(nlp, jrows_nln, badjcols_nln)
        @test_throws DimensionError jac_nln_coord(nlp, badx)
        @test_throws DimensionError jac_nln_coord!(nlp, badx, jvals_nln)
        @test_throws DimensionError jac_nln_coord!(nlp, x, badjvals_nln)
      end
    end
    if jac_op ∉ exclude
      @test_throws DimensionError jac_op(nlp, badx)
      @test_throws DimensionError jac_op!(nlp, badx, Jv, Jtw)
      @test_throws DimensionError jac_op!(nlp, x, badJv, Jtw)
      @test_throws DimensionError jac_op!(nlp, x, Jv, badJtw)
      @test_throws DimensionError jac_op!(nlp, badjrows, jcols, jvals, Jv, Jtw)
      @test_throws DimensionError jac_op!(nlp, jrows, badjcols, jvals, Jv, Jtw)
      @test_throws DimensionError jac_op!(nlp, jrows, jcols, badjvals, Jv, Jtw)
      @test_throws DimensionError jac_op!(nlp, jrows, jcols, jvals, badJv, Jtw)
      @test_throws DimensionError jac_op!(nlp, jrows, jcols, jvals, Jv, badJtw)

      if linear_api && nlp.meta.nlin > 0
        @test_throws DimensionError jac_lin_op(nlp, badx)
        @test_throws DimensionError jac_lin_op!(nlp, badx, Jv_lin, Jtw)
        @test_throws DimensionError jac_lin_op!(nlp, x, badJv_lin, Jtw)
        @test_throws DimensionError jac_lin_op!(nlp, x, Jv_lin, badJtw)
        @test_throws DimensionError jac_lin_op!(nlp, badjrows_lin, jcols_lin, jvals_lin, Jv_lin, Jtw)
        @test_throws DimensionError jac_lin_op!(nlp, jrows_lin, badjcols_lin, jvals_lin, Jv_lin, Jtw)
        @test_throws DimensionError jac_lin_op!(nlp, jrows_lin, jcols_lin, badjvals_lin, Jv_lin, Jtw)
        @test_throws DimensionError jac_lin_op!(nlp, jrows_lin, jcols_lin, jvals_lin, badJv_lin, Jtw)
        @test_throws DimensionError jac_lin_op!(nlp, jrows_lin, jcols_lin, jvals_lin, Jv_lin, badJtw)
      end

      if linear_api && nlp.meta.nnln > 0
        @test_throws DimensionError jac_nln_op(nlp, badx)
        @test_throws DimensionError jac_nln_op!(nlp, badx, Jv_nln, Jtw)
        @test_throws DimensionError jac_nln_op!(nlp, x, badJv_nln, Jtw)
        @test_throws DimensionError jac_nln_op!(nlp, x, Jv_nln, badJtw)
        @test_throws DimensionError jac_nln_op!(nlp, badjrows_nln, jcols_nln, jvals_nln, Jv_nln, Jtw)
        @test_throws DimensionError jac_nln_op!(nlp, jrows_nln, badjcols_nln, jvals_nln, Jv_nln, Jtw)
        @test_throws DimensionError jac_nln_op!(nlp, jrows_nln, jcols_nln, badjvals_nln, Jv_nln, Jtw)
        @test_throws DimensionError jac_nln_op!(nlp, jrows_nln, jcols_nln, jvals_nln, badJv_nln, Jtw)
        @test_throws DimensionError jac_nln_op!(nlp, jrows_nln, jcols_nln, jvals_nln, Jv_nln, badJtw)
      end
    end
  end
end
