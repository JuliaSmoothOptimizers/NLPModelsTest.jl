using Distributed

np = Sys.CPU_THREADS
addprocs(np - 1)

@everywhere using NLPModels, NLPModelsTest, Test

@everywhere function nlp_tests(p)
  @testset "NLP tests of problem $p" begin
    nlp_from_T = eval(Symbol(p))
    nlp = nlp_from_T()
    @testset "Consistency of problem $p" begin
      consistent_nlps([nlp, nlp], exclude = [])
    end
    @testset "Check dimensions of problem $p" begin
      check_nlp_dimensions(nlp, linear_api = true, exclude = [])
    end
    @testset "Multiple precision support of problem $p" begin
      multiple_precision_nlp(nlp_from_T, linear_api = true, exclude = [])
    end
    @testset "View subarray of problem $p" begin
      view_subarray_nlp(nlp, exclude = [])
    end
    @testset "Test coord memory of problem $p" begin
      coord_memory_nlp(nlp, exclude = [])
    end
  end
end

@everywhere function nls_tests(p)
  @testset "NLS tests of problem $p" begin
    nls_from_T = eval(Symbol(p))
    nls = nls_from_T()
    exclude = p == "LLS" ? [hess_coord, hess] : []
    @testset "Consistency of problem $p" begin
      consistent_nlss([nls, nls], exclude = exclude)
    end
    @testset "Check dimensions of problem $p" begin
      check_nls_dimensions(nls, exclude = exclude)
    end
    @testset "Multiple precision support of problem $p" begin
      multiple_precision_nls(nls_from_T, linear_api = true, exclude = exclude)
    end
    @testset "View subarray of problem $p" begin
      view_subarray_nls(nls, exclude = exclude)
    end
  end
end

pmap(nlp_tests, NLPModelsTest.nlp_problems)
pmap(nls_tests, NLPModelsTest.nls_problems)

rmprocs()
