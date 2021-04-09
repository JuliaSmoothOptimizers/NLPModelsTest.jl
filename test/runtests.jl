using NLPModelsTest, Test

@testset "NLP tests" begin
  for p in NLPModelsTest.nlp_problems
    @testset "Problem $p" begin
      nlp = eval(Symbol(p))()
      @testset "Consistency" begin
        consistent_nlps([nlp, nlp], exclude = [])
      end
      @testset "Check dimensions" begin
        check_nlp_dimensions(nlp, exclude = [])
      end
      @testset "Multiple precision support" begin
        multiple_precision_nlp(nlp, exclude = [])
      end
      @testset "View subarray" begin
        view_subarray_nlp(nlp, exclude = [])
      end
      @testset "Test coord memory" begin
        coord_memory_nlp(nlp, exclude = [])
      end
    end
  end
end

@testset "NLS tests" begin
  for p in NLPModelsTest.nls_problems
    @testset "Problem $p" begin
      nls = eval(Symbol(p))()
      @testset "Consistency" begin
        consistent_nlss([nls, nls], exclude = [])
      end
      @testset "Check dimensions" begin
        check_nls_dimensions(nls, exclude = [])
      end
      @testset "Multiple precision support" begin
        multiple_precision_nls(nls, exclude = [])
      end
      @testset "View subarray" begin
        view_subarray_nls(nls, exclude = [])
      end
    end
  end
end
