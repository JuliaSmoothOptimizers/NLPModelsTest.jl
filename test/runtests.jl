using NLPModelsTest, Test

@testset "NLP tests" begin
  for p in NLPModelsTest.nlp_problems
    @testset "Problem $p" begin
      nlp = eval(Symbol(p))()
      @testset "Consistency" begin
        consistent_nlps([nlp, nlp])
      end
      @testset "Check dimensions" begin
        check_nlp_dimensions(nlp)
      end
      @testset "Multiple precision support" begin
        multiple_precision_nlp(nlp)
      end
      @testset "View subarray" begin
        view_subarray_nlp(nlp)
      end
      @testset "Test coord memory" begin
        coord_memory_nlp(nlp)
      end
    end
  end
end

@testset "NLS tests" begin
  for p in NLPModelsTest.nls_problems
    @testset "Problem $p" begin
      nls = eval(Symbol(p))()
      @testset "Consistency" begin
        consistent_nlss([nls, nls])
      end
      @testset "Check dimensions" begin
        check_nls_dimensions(nls)
      end
      @testset "Multiple precision support" begin
        multiple_precision_nls(nls)
      end
      @testset "View subarray" begin
        view_subarray_nls(nls)
      end
    end
  end
end