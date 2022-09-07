using Documenter, NLPModelsTest

makedocs(
  modules = [NLPModelsTest],
  doctest = true,
  linkcheck = false,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "NLPModelsTest.jl",
  pages = ["Home" => "index.md", "Problems" => "problems.md", "Test allocations of NLPModels" => "allocs.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/NLPModelsTest.jl.git",
  push_preview = true,
  devbranch = "main",
)
