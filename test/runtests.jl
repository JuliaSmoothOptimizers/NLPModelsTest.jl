using Distributed

np = Sys.CPU_THREADS
addprocs(np - 1)

@everywhere using NLPModels, NLPModelsTest, Test

@everywhere function nlp_tests(p)
  @testset "NLP tests of problem $p" begin
    nlp = eval(Symbol(p))()
    @testset "Consistency of problem $p" begin
      consistent_nlps([nlp, nlp], exclude = [])
    end
    @testset "Check dimensions of problem $p" begin
      check_nlp_dimensions(nlp, exclude = [])
    end
    @testset "Multiple precision support of problem $p" begin
      multiple_precision_nlp(p, exclude = [])
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
    nls = eval(Symbol(p))()
    exclude = p == "LLS" ? [hess_coord, hess] : []
    @testset "Consistency of problem $p" begin
      consistent_nlss([nls, nls], exclude = exclude)
    end
    @testset "Check dimensions of problem $p" begin
      check_nls_dimensions(nls, exclude = exclude)
    end
    @testset "Multiple precision support of problem $p" begin
      multiple_precision_nls(p, exclude = exclude)
    end
    @testset "View subarray of problem $p" begin
      view_subarray_nls(nls, exclude = exclude)
    end
  end
end

pmap(nlp_tests, NLPModelsTest.nlp_problems)
pmap(nls_tests, NLPModelsTest.nls_problems)

rmprocs()
