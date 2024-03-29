export consistent_nlps

"""
    consistent_nlps(nlps; exclude=[], rtol=1e-8)

Check that the all `nlp`s of the vector `nlps` are consistent, in the sense that
- Their counters are the same.
- Their `meta` information is the same.
- The API functions return the same output given the same input.

In other words, if you create two models of the same problem, they should be consistent.

The keyword `exclude` can be used to pass functions to be ignored, if some of the models don't implement that function.
"""
function consistent_nlps(
  nlps;
  exclude = [jth_hess, jth_hess_coord, jth_hprod, ghjvprod],
  linear_api = false,
  reimplemented = ["jtprod"],
  test_meta = true,
  test_slack = true,
  test_qn = true,
  test_derivative = true,
  rtol = 1.0e-8,
)
  consistent_counters(nlps, linear_api = linear_api, reimplemented = reimplemented)
  test_meta && consistent_meta(nlps, rtol = rtol)
  consistent_functions(nlps, linear_api = linear_api, rtol = rtol, exclude = exclude)
  consistent_counters(nlps, linear_api = linear_api, reimplemented = reimplemented)
  for nlp in nlps
    reset!(nlp)
  end
  consistent_counters(nlps, linear_api = linear_api, reimplemented = reimplemented)
  if test_derivative
    for nlp in nlps
      @test length(gradient_check(nlp)) == 0
      @test length(jacobian_check(nlp)) == 0
      @test sum(map(length, values(hessian_check(nlp)))) == 0
      @test sum(map(length, values(hessian_check_from_grad(nlp)))) == 0
    end
  end

  if test_qn
    # Test Quasi-Newton models
    qnmodels = [
      [LBFGSModel(nlp) for nlp in nlps]
      [LSR1Model(nlp) for nlp in nlps]
    ]
    consistent_functions(
      [nlps; qnmodels],
      linear_api = linear_api,
      exclude = [hess, hess_coord, hprod, jth_hess, jth_hess_coord, jth_hprod, ghjvprod] ∪ exclude,
    )
    consistent_counters([nlps; qnmodels], linear_api = linear_api, reimplemented = reimplemented)
  end

  if test_slack && has_inequalities(nlps[1])
    reset!.(nlps)
    slack_nlps = SlackModel.(nlps)
    consistent_functions(
      slack_nlps,
      linear_api = linear_api,
      exclude = [jth_hess, jth_hess_coord, jth_hprod] ∪ exclude,
    )
    consistent_counters(slack_nlps, linear_api = linear_api, reimplemented = reimplemented)
  end
end

function consistent_meta(nlps; rtol = 1.0e-8)
  fields = [:nvar, :x0, :lvar, :uvar, :ifix, :ilow, :iupp, :irng, :ifree, :ncon, :y0]
  N = length(nlps)
  for field in fields
    @testset "Field $field" begin
      for i = 1:(N - 1)
        fi = getfield(nlps[i].meta, field)
        fj = getfield(nlps[i + 1].meta, field)
        @test isapprox(fi, fj, rtol = rtol)
      end
    end
  end
end

function consistent_counters(nlps; linear_api = false, reimplemented = String[])
  N = length(nlps)
  V = zeros(Int, N)
  check_fields = filter(
    x -> !(occursin("lin", string(x)) | occursin("nln", string(x))),
    collect(fieldnames(Counters)),
  )
  for field in check_fields
    V = [eval(field)(nlp) for nlp in nlps]
    @testset "Field $field" begin
      for i = 1:(N - 1)
        @test V[i] == V[i + 1]
      end
    end
  end
  if linear_api
    V = [sum_counters(nlp) for nlp in nlps]
    @test (reimplemented != []) | all(V .== V[1])
    for field in setdiff(collect(fieldnames(Counters)), check_fields)
      if any(x -> occursin(x, string(field)), reimplemented)
        continue
      end
      V = [eval(field)(nlp) for nlp in nlps]
      @testset "Field $field" begin
        for i = 1:(N - 1)
          @test V[i] == V[i + 1]
        end
      end
    end
  end
end

function consistent_functions(nlps; linear_api = false, rtol = 1.0e-8, exclude = [])
  N = length(nlps)
  n = nlps[1].meta.nvar
  m = nlps[1].meta.ncon
  test_lin = linear_api && (nlps[1].meta.nlin > 0)
  mlin = nlps[1].meta.nlin
  test_nln = linear_api && (nlps[1].meta.nnln > 0)
  mnln = nlps[1].meta.nnln

  tmp_n = zeros(n)
  tmp_m = zeros(m)
  tmp_mlin = zeros(mlin)
  tmp_mnln = zeros(mnln)
  tmp_nn = zeros(n, n)

  x = 10 * [-(-1.0)^i for i = 1:n]

  if !(obj in exclude)
    fs = [obj(nlp, x) for nlp in nlps]
    fmin = minimum(map(abs, fs))
    for i = 1:N
      for j = (i + 1):N
        @test isapprox(fs[i], fs[j], atol = rtol * max(fmin, 1.0))
      end

      if !(objcons in exclude)
        # Test objcons for unconstrained problems
        if m == 0
          f, c = objcons(nlps[i], x)
          @test isapprox(fs[i], f, rtol = rtol)
          @test c == []
          f, tmpc = objcons!(nlps[i], x, c)
          @test isapprox(fs[i], f, rtol = rtol)
          @test c == []
          @test tmpc == []
        end
      end
    end
  end

  if !(grad in exclude)
    gs = Any[grad(nlp, x) for nlp in nlps]
    gmin = minimum(map(norm, gs))
    for i = 1:N
      for j = (i + 1):N
        @test isapprox(gs[i], gs[j], atol = rtol * max(gmin, 1.0))
      end
      tmpg = grad!(nlps[i], x, tmp_n)
      @test isapprox(gs[i], tmp_n, atol = rtol * max(gmin, 1.0))
      @test isapprox(tmpg, tmp_n, atol = rtol * max(gmin, 1.0))

      if !(objgrad in exclude)
        f, g = objgrad(nlps[i], x)
        @test isapprox(fs[i], f, atol = rtol * max(abs(f), 1.0))
        @test isapprox(gs[i], g, atol = rtol * max(gmin, 1.0))
        f, tmpg = objgrad!(nlps[i], x, g)
        @test isapprox(fs[i], f, atol = rtol * max(abs(f), 1.0))
        @test isapprox(gs[i], g, atol = rtol * max(gmin, 1.0))
        @test isapprox(g, tmpg, atol = rtol * max(gmin, 1.0))
      end
    end
  end

  if !(hess_coord in exclude)
    Hs = Vector{Any}(undef, N)
    for i = 1:N
      V = hess_coord(nlps[i], x)
      I, J = hess_structure(nlps[i])
      Hs[i] = sparse(I, J, V, n, n)
    end
    Hmin = minimum(map(norm, Hs))
    for i = 1:N
      for j = (i + 1):N
        @test isapprox(Hs[i], Hs[j], atol = rtol * max(Hmin, 1.0))
      end
      V = hess_coord(nlps[i], x, obj_weight = 0.0)
      @test norm(V) ≈ 0
      σ = 3.14
      V = hess_coord(nlps[i], x, obj_weight = σ)
      I, J = hess_structure(nlps[i])
      tmp_h = sparse(I, J, V, n, n)
      @test isapprox(σ * Hs[i], tmp_h, atol = rtol * max(Hmin, 1.0))
      tmp_V = zeros(nlps[i].meta.nnzh)
      hess_coord!(nlps[i], x, tmp_V, obj_weight = σ)
      @test tmp_V == V
    end
  end

  if !(hess in exclude)
    Hs = Any[hess(nlp, x) for nlp in nlps]
    Hmin = minimum(map(norm, Hs))
    for i = 1:N
      for j = (i + 1):N
        @test isapprox(Hs[i], Hs[j], atol = rtol * max(Hmin, 1.0))
      end
      @test Hs[i] isa Symmetric
      tmp_nn = hess(nlps[i], x, obj_weight = 0.0)
      @test norm(tmp_nn) ≈ 0
      σ = 3.14
      tmp_nn = hess(nlps[i], x, obj_weight = σ)
      @test isapprox(σ * Hs[i], tmp_nn, atol = rtol * max(Hmin, 1.0))
    end
  end

  v = 10 * [-(-1.0)^i for i = 1:n]

  if !(hprod in exclude)
    for σ in [1.0; 0.5; 0.0]
      Hvs = Any[hprod(nlp, x, v, obj_weight = σ) for nlp in nlps]
      Hopvs = Any[hess_op(nlp, x, obj_weight = σ) * v for nlp in nlps]
      Hvmin = minimum(map(norm, Hvs))
      for i = 1:N
        for j = (i + 1):N
          @test isapprox(Hvs[i], Hvs[j], atol = rtol * max(Hvmin, 1.0))
          @test isapprox(Hvs[i], Hopvs[j], atol = rtol * max(Hvmin, 1.0))
        end
        tmphv = hprod!(nlps[i], x, v, tmp_n, obj_weight = σ)
        @test isapprox(Hvs[i], tmp_n, atol = rtol * max(Hvmin, 1.0))
        @test isapprox(tmphv, tmp_n, atol = rtol * max(Hvmin, 1.0))
        fill!(tmp_n, 0)
        H = hess_op!(nlps[i], x, tmp_n, obj_weight = σ)
        res = H * v
        @test isapprox(res, Hvs[i], atol = rtol * max(Hvmin, 1.0))
        @test isapprox(res, tmp_n, atol = rtol * max(Hvmin, 1.0))

        if !(hess_coord in exclude)
          rows, cols = hess_structure(nlps[i])
          vals = hess_coord(nlps[i], x, obj_weight = σ)
          hprod!(nlps[i], rows, cols, vals, v, tmp_n)
          @test isapprox(Hvs[i], tmp_n, atol = rtol * max(Hvmin, 1.0))
          H = hess_op!(nlps[i], x, tmp_n, obj_weight = σ)
          res = H * v
          @test isapprox(Hvs[i], res, atol = rtol * max(Hvmin, 1.0))
        end
        if σ == 1 # Check hprod! with default obj_weight
          hprod!(nlps[i], x, v, tmp_n)
          @test isapprox(Hvs[i], tmp_n, atol = rtol * max(Hvmin, 1.0))
        end
      end
    end
  end

  if intersect([hess, hess_coord], exclude) == []
    for i = 1:N
      nlp = nlps[i]
      Hx = hess(nlp, x, obj_weight = 0.5)
      V = hess_coord(nlp, x, obj_weight = 0.5)
      I, J = hess_structure(nlp)
      @test length(I) == length(J) == length(V) == nlp.meta.nnzh
      @test Symmetric(sparse(I, J, V, n, n), :L) == Hx
    end
  end

  if m > 0
    if !(cons in exclude)
      cs = Any[cons(nlp, x) for nlp in nlps]
      cls = [nlp.meta.lcon for nlp in nlps]
      cus = [nlp.meta.ucon for nlp in nlps]
      cmin = minimum(map(norm, cs))
      for i = 1:N
        tmpc = cons!(nlps[i], x, tmp_m)
        @test isapprox(cs[i], tmp_m, atol = rtol * max(cmin, 1.0))
        @test isapprox(tmpc, tmp_m, atol = rtol * max(cmin, 1.0))
        ci, li, ui = copy(cs[i]), cls[i], cus[i]
        for k = 1:m
          if li[k] > -Inf
            ci[k] -= li[k]
          elseif ui[k] < Inf
            ci[k] -= ui[k]
          end
        end
        for j = (i + 1):N
          cj, lj, uj = copy(cs[j]), cls[j], cus[j]
          for k = 1:m
            if lj[k] > -Inf
              cj[k] -= lj[k]
            elseif uj[k] < Inf
              cj[k] -= uj[k]
            end
          end
          @test isapprox(norm(ci), norm(cj), atol = rtol * max(cmin, 1.0))
        end

        if !(objcons in exclude)
          f, c = objcons(nlps[i], x)
          @test isapprox(fs[i], f, atol = rtol * max(abs(f), 1.0))
          @test isapprox(cs[i], c, atol = rtol * max(cmin, 1.0))
          f, tmpc = objcons!(nlps[i], x, c)
          @test isapprox(fs[i], f, atol = rtol * max(abs(f), 1.0))
          @test isapprox(cs[i], c, atol = rtol * max(cmin, 1.0))
          @test isapprox(c, tmpc, atol = rtol * max(cmin, 1.0))
        end
      end
    end

    if !(cons in exclude) && test_lin
      cs = Any[cons_lin(nlp, x) for nlp in nlps]
      cls = [nlp.meta.lcon[nlp.meta.lin] for nlp in nlps]
      cus = [nlp.meta.ucon[nlp.meta.lin] for nlp in nlps]
      cmin = minimum(map(norm, cs))
      for i = 1:N
        tmpc = cons_lin!(nlps[i], x, tmp_mlin)
        @test isapprox(cs[i], tmp_mlin, atol = rtol * max(cmin, 1.0))
        @test isapprox(tmpc, tmp_mlin, atol = rtol * max(cmin, 1.0))
        ci, li, ui = copy(cs[i]), cls[i], cus[i]
        for k = 1:mlin
          if li[k] > -Inf
            ci[k] -= li[k]
          elseif ui[k] < Inf
            ci[k] -= ui[k]
          end
        end
        for j = (i + 1):N
          cj, lj, uj = copy(cs[j]), cls[j], cus[j]
          for k = 1:mlin
            if lj[k] > -Inf
              cj[k] -= lj[k]
            elseif uj[k] < Inf
              cj[k] -= uj[k]
            end
          end
          @test isapprox(norm(ci), norm(cj), atol = rtol * max(cmin, 1.0))
        end
      end
    end

    if !(cons in exclude) && test_nln
      cs = Any[cons_nln(nlp, x) for nlp in nlps]
      cls = [nlp.meta.lcon[nlp.meta.nln] for nlp in nlps]
      cus = [nlp.meta.ucon[nlp.meta.nln] for nlp in nlps]
      cmin = minimum(map(norm, cs))
      for i = 1:N
        tmpc = cons_nln!(nlps[i], x, tmp_mnln)
        @test isapprox(cs[i], tmp_mnln, atol = rtol * max(cmin, 1.0))
        @test isapprox(tmpc, tmp_mnln, atol = rtol * max(cmin, 1.0))
        ci, li, ui = copy(cs[i]), cls[i], cus[i]
        for k = 1:mnln
          if li[k] > -Inf
            ci[k] -= li[k]
          elseif ui[k] < Inf
            ci[k] -= ui[k]
          end
        end
        for j = (i + 1):N
          cj, lj, uj = copy(cs[j]), cls[j], cus[j]
          for k = 1:mnln
            if lj[k] > -Inf
              cj[k] -= lj[k]
            elseif uj[k] < Inf
              cj[k] -= uj[k]
            end
          end
          @test isapprox(norm(ci), norm(cj), atol = rtol * max(cmin, 1.0))
        end
      end
    end

    if intersect([jac, jac_coord], exclude) == []
      Js = [jac(nlp, x) for nlp in nlps]
      Jmin = minimum(map(norm, Js))
      for i = 1:N
        vi = norm(Js[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Js[j]), atol = rtol * max(Jmin, 1.0))
        end
        V = jac_coord(nlps[i], x)
        I, J = jac_structure(nlps[i])
        @test length(I) == length(J) == length(V) == nlps[i].meta.nnzj
        @test isapprox(sparse(I, J, V, m, n), Js[i], atol = rtol * max(Jmin, 1.0))
        IS, JS = zeros(Int, nlps[i].meta.nnzj), zeros(Int, nlps[i].meta.nnzj)
        jac_structure!(nlps[i], IS, JS)
        @test IS == I
        @test JS == J
        tmp_V = zeros(nlps[i].meta.nnzj)
        jac_coord!(nlps[i], x, tmp_V)
        @test tmp_V == V
      end
    end

    if (intersect([jac, jac_coord], exclude) == []) && test_lin
      Js = [jac_lin(nlp, x) for nlp in nlps]
      Jmin = minimum(map(norm, Js))
      for i = 1:N
        vi = norm(Js[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Js[j]), atol = rtol * max(Jmin, 1.0))
        end
        V = jac_lin_coord(nlps[i], x)
        I, J = jac_lin_structure(nlps[i])
        @test length(I) == length(J) == length(V) == nlps[i].meta.lin_nnzj
        @test isapprox(sparse(I, J, V, mlin, n), Js[i], atol = rtol * max(Jmin, 1.0))
        IS, JS = zeros(Int, nlps[i].meta.lin_nnzj), zeros(Int, nlps[i].meta.lin_nnzj)
        jac_lin_structure!(nlps[i], IS, JS)
        @test IS == I
        @test JS == J
        tmp_V = zeros(nlps[i].meta.lin_nnzj)
        jac_lin_coord!(nlps[i], x, tmp_V)
        @test tmp_V == V
      end
    end

    if (intersect([jac, jac_coord], exclude) == []) && test_nln
      Js = [jac_nln(nlp, x) for nlp in nlps]
      Jmin = minimum(map(norm, Js))
      for i = 1:N
        vi = norm(Js[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Js[j]), atol = rtol * max(Jmin, 1.0))
        end
        V = jac_nln_coord(nlps[i], x)
        I, J = jac_nln_structure(nlps[i])
        @test length(I) == length(J) == length(V) == nlps[i].meta.nln_nnzj
        @test isapprox(sparse(I, J, V, mnln, n), Js[i], atol = rtol * max(Jmin, 1.0))
        IS, JS = zeros(Int, nlps[i].meta.nln_nnzj), zeros(Int, nlps[i].meta.nln_nnzj)
        jac_nln_structure!(nlps[i], IS, JS)
        @test IS == I
        @test JS == J
        tmp_V = zeros(nlps[i].meta.nln_nnzj)
        jac_nln_coord!(nlps[i], x, tmp_V)
        @test tmp_V == V
      end
    end

    if !(jprod in exclude)
      Jops = Any[jac_op(nlp, x) for nlp in nlps]
      Jps = Any[jprod(nlp, x, v) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jps[i], Jops[i] * v, atol = rtol * max(Jmin, 1.0))
        vi = norm(Jps[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Jps[j]), atol = rtol * max(Jmin, 1.0))
        end
        tmpjv = jprod!(nlps[i], x, v, tmp_m)
        @test isapprox(tmpjv, tmp_m, atol = rtol * max(Jmin, 1.0))
        @test isapprox(Jps[i], tmp_m, atol = rtol * max(Jmin, 1.0))
        fill!(tmp_m, 0)
        J = jac_op!(nlps[i], x, tmp_m, tmp_n)
        res = J * v
        @test isapprox(res, Jps[i], atol = rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_m, atol = rtol * max(Jmin, 1.0))

        if !(jac_coord in exclude)
          rows, cols = jac_structure(nlps[i])
          vals = jac_coord(nlps[i], x)
          jprod!(nlps[i], rows, cols, vals, v, tmp_m)
          @test isapprox(Jps[i], tmp_m, atol = rtol * max(Jmin, 1.0))

          J = jac_op!(nlps[i], rows, cols, vals, tmp_m, tmp_n)
          res = J * v
          @test isapprox(res, Jps[i], atol = rtol * max(Jmin, 1.0))
        end
      end
    end

    if (!(jprod in exclude)) && test_lin
      Jlinops = Any[jac_lin_op(nlp, x) for nlp in nlps]
      Jps = Any[jprod_lin(nlp, x, v) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jps[i], Jlinops[i] * v, atol = rtol * max(Jmin, 1.0))
        vi = norm(Jps[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Jps[j]), atol = rtol * max(Jmin, 1.0))
        end
        tmpjv = jprod_lin!(nlps[i], x, v, tmp_mlin)
        @test isapprox(tmpjv, tmp_mlin, atol = rtol * max(Jmin, 1.0))
        @test isapprox(Jps[i], tmp_mlin, atol = rtol * max(Jmin, 1.0))
        fill!(tmp_mlin, 0)
        J = jac_lin_op!(nlps[i], x, tmp_mlin, tmp_n)
        res = J * v
        @test isapprox(res, Jps[i], atol = rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_mlin, atol = rtol * max(Jmin, 1.0))

        if !(jac_coord in exclude)
          rows, cols = jac_lin_structure(nlps[i])
          vals = jac_lin_coord(nlps[i], x)
          jprod_lin!(nlps[i], rows, cols, vals, v, tmp_mlin)
          @test isapprox(Jps[i], tmp_mlin, atol = rtol * max(Jmin, 1.0))

          J = jac_lin_op!(nlps[i], rows, cols, vals, tmp_mlin, tmp_n)
          res = J * v
          @test isapprox(res, Jps[i], atol = rtol * max(Jmin, 1.0))
        end
      end
    end

    if (!(jprod in exclude)) && test_nln
      Jnlnops = Any[jac_nln_op(nlp, x) for nlp in nlps]
      Jps = Any[jprod_nln(nlp, x, v) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jps[i], Jnlnops[i] * v, atol = rtol * max(Jmin, 1.0))
        vi = norm(Jps[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Jps[j]), atol = rtol * max(Jmin, 1.0))
        end
        tmpjv = jprod_nln!(nlps[i], x, v, tmp_mnln)
        @test isapprox(tmpjv, tmp_mnln, atol = rtol * max(Jmin, 1.0))
        @test isapprox(Jps[i], tmp_mnln, atol = rtol * max(Jmin, 1.0))
        fill!(tmp_mnln, 0)
        J = jac_nln_op!(nlps[i], x, tmp_mnln, tmp_n)
        res = J * v
        @test isapprox(res, Jps[i], atol = rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_mnln, atol = rtol * max(Jmin, 1.0))

        if !(jac_coord in exclude)
          rows, cols = jac_nln_structure(nlps[i])
          vals = jac_nln_coord(nlps[i], x)
          jprod_nln!(nlps[i], rows, cols, vals, v, tmp_mnln)
          @test isapprox(Jps[i], tmp_mnln, atol = rtol * max(Jmin, 1.0))

          J = jac_nln_op!(nlps[i], rows, cols, vals, tmp_mnln, tmp_n)
          res = J * v
          @test isapprox(res, Jps[i], atol = rtol * max(Jmin, 1.0))
        end
      end
    end

    if !(jtprod in exclude)
      w = 10 * [-(-1.0)^i for i = 1:m]
      Jtps = Any[jtprod(nlp, x, w) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jtps[i], Jops[i]' * w, atol = rtol * max(Jmin, 1.0))
        vi = norm(Jtps[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Jtps[j]), atol = rtol * max(Jmin, 1.0))
        end
        tmpjtv = jtprod!(nlps[i], x, w, tmp_n)
        @test isapprox(Jtps[i], tmp_n, atol = rtol * max(Jmin, 1.0))
        @test isapprox(tmpjtv, tmp_n, atol = rtol * max(Jmin, 1.0))
        fill!(tmp_n, 0)
        J = jac_op!(nlps[i], x, tmp_m, tmp_n)
        res = J' * w
        @test isapprox(res, Jtps[i], atol = rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_n, atol = rtol * max(Jmin, 1.0))

        if !(jac_coord in exclude)
          rows, cols = jac_structure(nlps[i])
          vals = jac_coord(nlps[i], x)
          jtprod!(nlps[i], rows, cols, vals, w, tmp_n)
          @test isapprox(Jtps[i], tmp_n, atol = rtol * max(Jmin, 1.0))

          J = jac_op!(nlps[i], rows, cols, vals, tmp_m, tmp_n)
          res = J' * w
          @test isapprox(res, Jtps[i], atol = rtol * max(Jmin, 1.0))
        end
      end
    end

    if (!(jtprod in exclude)) && test_lin
      w = 10 * [-(-1.0)^i for i = 1:mlin]
      Jtps = Any[jtprod_lin(nlp, x, w) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jtps[i], Jlinops[i]' * w, atol = rtol * max(Jmin, 1.0))
        vi = norm(Jtps[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Jtps[j]), atol = rtol * max(Jmin, 1.0))
        end
        tmpjtv = jtprod_lin!(nlps[i], x, w, tmp_n)
        @test isapprox(Jtps[i], tmp_n, atol = rtol * max(Jmin, 1.0))
        @test isapprox(tmpjtv, tmp_n, atol = rtol * max(Jmin, 1.0))
        fill!(tmp_n, 0)
        J = jac_lin_op!(nlps[i], x, tmp_mlin, tmp_n)
        res = J' * w
        @test isapprox(res, Jtps[i], atol = rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_n, atol = rtol * max(Jmin, 1.0))

        if !(jac_coord in exclude)
          rows, cols = jac_lin_structure(nlps[i])
          vals = jac_lin_coord(nlps[i], x)
          jtprod_lin!(nlps[i], rows, cols, vals, w, tmp_n)
          @test isapprox(Jtps[i], tmp_n, atol = rtol * max(Jmin, 1.0))

          J = jac_lin_op!(nlps[i], rows, cols, vals, tmp_mlin, tmp_n)
          res = J' * w
          @test isapprox(res, Jtps[i], atol = rtol * max(Jmin, 1.0))
        end
      end
    end

    if (!(jtprod in exclude)) && test_nln
      w = 10 * [-(-1.0)^i for i = 1:mnln]
      Jtps = Any[jtprod_nln(nlp, x, w) for nlp in nlps]
      for i = 1:N
        @test isapprox(Jtps[i], Jnlnops[i]' * w, atol = rtol * max(Jmin, 1.0))
        vi = norm(Jtps[i])
        for j = (i + 1):N
          @test isapprox(vi, norm(Jtps[j]), atol = rtol * max(Jmin, 1.0))
        end
        tmpjtv = jtprod_nln!(nlps[i], x, w, tmp_n)
        @test isapprox(Jtps[i], tmp_n, atol = rtol * max(Jmin, 1.0))
        @test isapprox(tmpjtv, tmp_n, atol = rtol * max(Jmin, 1.0))
        fill!(tmp_n, 0)
        J = jac_nln_op!(nlps[i], x, tmp_mnln, tmp_n)
        res = J' * w
        @test isapprox(res, Jtps[i], atol = rtol * max(Jmin, 1.0))
        @test isapprox(res, tmp_n, atol = rtol * max(Jmin, 1.0))

        if !(jac_coord in exclude)
          rows, cols = jac_nln_structure(nlps[i])
          vals = jac_nln_coord(nlps[i], x)
          jtprod_nln!(nlps[i], rows, cols, vals, w, tmp_n)
          @test isapprox(Jtps[i], tmp_n, atol = rtol * max(Jmin, 1.0))

          J = jac_nln_op!(nlps[i], rows, cols, vals, tmp_mnln, tmp_n)
          res = J' * w
          @test isapprox(res, Jtps[i], atol = rtol * max(Jmin, 1.0))
        end
      end
    end

    y = 3.14 * ones(m)

    if !(hess_coord in exclude)
      Ls = Vector{Any}(undef, N)
      for i = 1:N
        V = hess_coord(nlps[i], x, y)
        I, J = hess_structure(nlps[i])
        Ls[i] = sparse(I, J, V, n, n)
      end
      Lmin = minimum(map(norm, Ls))
      for i = 1:N
        for j = (i + 1):N
          @test isapprox(Ls[i], Ls[j], atol = rtol * max(Lmin, 1.0))
        end
        V = hess_coord(nlps[i], x, 0 * y, obj_weight = 0.0)
        @test norm(V) ≈ 0
        σ = 3.14
        V = hess_coord(nlps[i], x, σ * y, obj_weight = σ)
        I, J = hess_structure(nlps[i])
        tmp_h = sparse(I, J, V, n, n)
        @test isapprox(σ * Ls[i], tmp_h, atol = rtol * max(Lmin, 1.0))
        tmp_V = zeros(nlps[i].meta.nnzh)
        hess_coord!(nlps[i], x, σ * y, tmp_V, obj_weight = σ)
        @test tmp_V == V
      end
    end

    if !(hess in exclude)
      Ls = Any[hess(nlp, x, y) for nlp in nlps]
      Lmin = minimum(map(norm, Ls))
      for i = 1:N
        for j = (i + 1):N
          @test isapprox(Ls[i], Ls[j], atol = rtol * max(Lmin, 1.0))
        end
        @test Ls[i] isa Symmetric
        tmp_nn = hess(nlps[i], x, 0 * y, obj_weight = 0.0)
        @test norm(tmp_nn) ≈ 0
        σ = 3.14
        tmp_nn = hess(nlps[i], x, σ * y, obj_weight = σ)
        @test isapprox(σ * Ls[i], tmp_nn, atol = rtol * max(Hmin, 1.0))
      end
    end

    if intersect([hess, hess_coord], exclude) == []
      for i = 1:N
        nlp = nlps[i]
        Hx = hess(nlp, x, y, obj_weight = 0.5)
        V = hess_coord(nlp, x, y, obj_weight = 0.5)
        I, J = hess_structure(nlp)
        @test length(I) == length(J) == length(V) == nlp.meta.nnzh
        @test Symmetric(sparse(I, J, V, n, n), :L) == Hx
      end
    end

    if !(hprod in exclude)
      for σ in [1.0; 0.5; 0.0]
        Lps = Any[hprod(nlp, x, y, v, obj_weight = σ) for nlp in nlps]
        Hopvs = Any[hess_op(nlp, x, y, obj_weight = σ) * v for nlp in nlps]
        Lpmin = minimum(map(norm, Lps))
        for i = 1:N
          for j = (i + 1):N
            @test isapprox(Lps[i], Lps[j], atol = rtol * max(Lpmin, 1.0))
            @test isapprox(Lps[i], Hopvs[j], atol = rtol * max(Lpmin, 1.0))
          end

          if !(hess_coord in exclude)
            rows, cols = hess_structure(nlps[i])
            vals = hess_coord(nlps[i], x, y, obj_weight = σ)
            hprod!(nlps[i], rows, cols, vals, v, tmp_n)
            @test isapprox(Lps[i], tmp_n, atol = rtol * max(Lpmin, 1.0))
            H = hess_op!(nlps[i], x, y, tmp_n, obj_weight = σ)
            res = H * v
            @test isapprox(Lps[i], res, atol = rtol * max(Lpmin, 1.0))
          end
        end
      end
    end

    if !(jth_hess_coord in exclude)
      Ls = Vector{Any}(undef, N)
      for i = 1:N
        V = jth_hess_coord(nlps[i], x, 1)
        I, J = hess_structure(nlps[i])
        Ls[i] = sparse(I, J, V, n, n)
      end
      Lmin = minimum(map(norm, Ls))
      for i = 1:N
        for j = (i + 1):N
          @test isapprox(Ls[i], Ls[j], atol = rtol * max(Lmin, 1.0))
        end
      end
    end

    if !(jth_hess in exclude)
      Ls = [jth_hess(nlp, x, m) for nlp in nlps]
      Lmin = minimum(map(norm, Ls))
      for i = 1:N
        for j = (i + 1):N
          @test isapprox(Ls[i], Ls[j], atol = rtol * max(Lmin, 1.0))
        end
      end
    end

    if intersect([jth_hess, jth_hess_coord], exclude) == []
      for i = 1:N
        nlp = nlps[i]
        Hx = jth_hess(nlp, x, 1)
        V = jth_hess_coord(nlp, x, 1)
        I, J = hess_structure(nlp)
        @test length(I) == length(J) == length(V) == nlp.meta.nnzh
        @test Symmetric(sparse(I, J, V, n, n), :L) == Hx
      end
    end

    if intersect([jth_hess, jth_hprod], exclude) == []
      Lps = [jth_hprod(nlp, x, v, max(m - 1, 1)) for nlp in nlps]
      Lpmin = minimum(map(norm, Lps))
      for i = 1:N
        for j = (i + 1):N
          @test isapprox(Lps[i], Lps[j], atol = rtol * max(Lpmin, 1.0))
        end

        if !(jth_hess_coord in exclude)
          rows, cols = hess_structure(nlps[i])
          vals = jth_hess_coord(nlps[i], x, max(m - 1, 1))
          tmp_n = similar(Lps[i])
          coo_sym_prod!(rows, cols, vals, v, tmp_n)
          @test isapprox(Lps[i], tmp_n, atol = rtol * max(Lpmin, 1.0))
        end
      end
    end

    g = 0.707 * ones(n)

    if !(ghjvprod in exclude)
      Ls = Any[ghjvprod(nlp, x, g, v) for nlp in nlps]
      Lmin = minimum(map(norm, Ls))
      for i = 1:N
        for j = (i + 1):N
          @test isapprox(Ls[i], Ls[j], atol = rtol * max(Lmin, 1.0))
        end
      end
    end

    if intersect([ghjvprod, jth_hprod], exclude) == []
      for i = 1:N
        nlp = nlps[i]
        gHjv = ghjvprod(nlp, x, g, v)
        tmp_ghjv = [dot(g, jth_hprod(nlp, x, v, j)) for j = 1:m]
        @test isapprox(gHjv, tmp_ghjv, atol = rtol * max(norm(gHjv), 1.0))
      end
    end
  end
end
