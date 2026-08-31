export view_subarray_nlp

"""
    view_subarray_nlp(nlp; exclude = [])

Check that the API work with views, and that the results is correct.
"""
function view_subarray_nlp(nlp; exclude = [])
  @testset "Test view subarray of NLPs" begin
    n, m, nnzh = nlp.meta.nvar, nlp.meta.ncon, nlp.meta.nnzh

    # Inputs
    x = [-(-1.1)^i for i = 1:n] # Instead of [1, -1, …], because it needs to
    v = [-(-1.1)^i for i = 1:n] # access different parts of the vector and
    y = [-(-1.1)^i for i = 1:m] # make a difference

    # Outputs
    g = zeros(n)
    g2 = zeros(2n)
    c = zeros(m)
    c2 = zeros(2m)
    jv = zeros(m)
    jv2 = zeros(2m)
    jty = zeros(n)
    jty2 = zeros(2n)
    hv = zeros(n)
    hv2 = zeros(2n)
    hval = zeros(nnzh)
    hval2 = zeros(2nnzh)

    I = 1:2:2n
    Iv = 1:nnzh
    J = 1:2:2m
      
    if hess_coord ∉ exclude
      Hval = @view hval2[Iv]
      vals1 = hess_coord!(nlp, x, Hval)
      vals2 = hess_coord!(nlp, x, hval)
      @test hval ≈ hval2[Iv]
    end

    if hess_coord ∉ exclude && m > 0
      Hval = @view hval2[Iv]
      vals1 = hess_coord!(nlp, x, y, Hval)
      vals2 = hess_coord!(nlp, x, y, hval)
      @test hval ≈ hval2[Iv]
    end

    if grad ∉ exclude
      gv = @view g2[I]
      grad!(nlp, x, gv)
      grad!(nlp, x, g)
      @test g ≈ g2[I]
    end

    if cons ∉ exclude && m > 0
      cv = @view c2[J]
      cons!(nlp, x, cv)
      cons!(nlp, x, c)
      @test c ≈ c2[J]
    end

    if jprod ∉ exclude && m > 0
      jvv = @view jv2[J]
      jprod!(nlp, x, v, jvv)
      jprod!(nlp, x, v, jv)
      @test jv ≈ jv2[J]
    end

    if jtprod ∉ exclude && m > 0
      jtyv = @view jty2[I]
      jtprod!(nlp, x, y, jtyv)
      jtprod!(nlp, x, y, jty)
      @test jty ≈ jty2[I]
    end

    if hprod ∉ exclude
      hvv = @view hv2[I]
      hprod!(nlp, x, v, hvv)
      hprod!(nlp, x, v, hv)
      @test hv ≈ hv2[I]
      if m > 0
        hprod!(nlp, x, y, v, hvv)
        hprod!(nlp, x, y, v, hv)
        @test hv ≈ hv2[I]
      end
    end
  end
end
