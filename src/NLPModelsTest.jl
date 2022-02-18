module NLPModelsTest

#stdlib
using LinearAlgebra, SparseArrays, Test
#jso
using NLPModels, NLPModelsModifiers

const nlp_problems =
  ["BROWNDEN", "HS5", "HS6", "HS10", "HS13", "HS11", "HS14", "LINCON", "LINSV", "MGH01Feas"]
const nls_problems = ["LLS", "MGH01", "NLSHS20", "NLSLC"]

# Including problems so that they won't be multiply loaded
# GENROSE does not have a manual version, so it's separate
for problem in nlp_problems
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

include("dercheck.jl")

for f in ["check-dimensions", "consistency", "multiple-precision", "view-subarray"]
  include("nlp/$f.jl")
  include("nls/$f.jl")
end
include("nlp/coord-memory.jl")

end
