for problem in TestUtils.nls_problems
  @testset "Checking TestUtils tests on problem $problem" begin
    nls_ad = eval(Meta.parse(lowercase(problem) * "_autodiff"))()
    nls_man = eval(Meta.parse(problem))()

    nlss = AbstractNLSModel[nls_ad]
    # *_special problems are variant definitions of a model
    spc = "$(problem)_special"
    if isdefined(TestUtils, Symbol(spc))
      push!(nlss, eval(Meta.parse(spc))())
    end

    for nls in nlss
      show(IOBuffer(), nls)
    end

    @testset "Check Consistency" begin
      consistent_nlss([nlss; nls_man], exclude = [])
    end
    @testset "Check dimensions" begin
      check_nls_dimensions.(nlss, exclude = [])
      check_nlp_dimensions.(nlss, exclude = [])
    end
    @testset "Check multiple precision" begin
      for nls in nlss
          multiple_precision_nls(nls, exclude = [])
      end
    end
    @testset "Check view subarray" begin
      view_subarray_nls.(nlss, exclude = [])
    end
  end
end