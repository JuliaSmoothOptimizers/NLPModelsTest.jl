export view_subarray_nls

"""
    view_subarray_nls(nls; exclude = [])

Check that the API work with views, and that the results is correct.
"""
function view_subarray_nls(nls; exclude = [])
  @testset "Test view subarray of NLSs" begin
    n, ne = nls.meta.nvar, nls.nls_meta.nequ  

    # Inputs
    x = [-(-1.1)^i for i = 1:n] # Instead of [1, -1, …], because it needs to
    v = [-(-1.1)^i for i = 1:n] # access different parts of the vector and
    y = [-(-1.1)^i for i = 1:ne] # make a difference

    # Outputs
    F = zeros(ne)
    F2 = zeros(2ne)
    jv = zeros(ne)
    jv2 = zeros(2ne)
    jty = zeros(n)
    jty2 = zeros(2n)
    hv = zeros(n)
    hv2 = zeros(2n)

    # Vidxs = [1:n, n .+ (1:n), 1:2:N, collect(N:-2:1)]
    I = collect(2n:-2:1)
    # Fidxs = [1:ne, ne .+ (1:ne), 1:2:N, collect(N:-2:1)]
    J = ne .+ (1:ne)

    if residual ∉ exclude
      Fv = @view F2[J]
      residual!(nls, x, Fv)
      residual!(nls, x, F)
      @test F ≈ F2[J]
    end

    if jprod_residual ∉ exclude
      jvv = @view jv2[J]
      jprod_residual!(nls, x, v, jvv)
      jprod_residual!(nls, x, v, jv)
      @test jv ≈ jv2[J]
    end

    if jtprod_residual ∉ exclude
      jtyv = @view jty2[I]
      jtprod_residual!(nls, x, y, jtyv)
      jtprod_residual!(nls, x, y, jty)
      @test jty ≈ jty2[I]
    end

    for i = 1:ne
      if hprod_residual ∉ exclude
        hvv = @view hv2[I]
        hprod_residual!(nls, x, i, v, hvv)
        hprod_residual!(nls, x, i, v, hv)
        @test hv ≈ hv2[I]
      end
    end
  end
end
