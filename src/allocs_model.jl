export test_allocs_nlpmodels, test_allocs_nlsmodels, test_zero_allocations, print_nlp_allocations

"""
    test_allocs_nlpmodels(nlp::AbstractNLPModel; linear_api = false, exclude = [])

Returns a `Dict` containing allocations of the in-place functions of NLPModel API.

The keyword `exclude` takes a Array of Function to be excluded from the tests. Use `hess` (resp. `jac`) to exclude `hess_coord` and `hess_structure` (resp. `jac_coord` and `jac_structure`).
Setting `linear_api` to `true` will also checks the functions specific to linear and nonlinear constraints.
"""
function test_allocs_nlpmodels(nlp::AbstractNLPModel; linear_api = false, exclude = [])
  nlp_allocations = Dict(
    :obj => NaN,
    :grad! => NaN,
    :hess_structure! => NaN,
    :hess_coord! => NaN,
    :hprod! => NaN,
    :hess_op_prod! => NaN,
    :cons! => NaN,
    :jac_structure! => NaN,
    :jac_coord! => NaN,
    :jprod! => NaN,
    :jtprod! => NaN,
    :jac_op_prod! => NaN,
    :jac_op_transpose_prod! => NaN,
    :hess_lag_coord! => NaN,
    :hprod_lag! => NaN,
    :hess_lag_op! => NaN,
    :hess_lag_op_prod! => NaN,
  )

  test_obj_grad!(nlp_allocations, nlp, exclude)

  if !(hess in exclude) && nlp.meta.hess_available
    rows = Vector{Int}(undef, nlp.meta.nnzh)
    cols = Vector{Int}(undef, nlp.meta.nnzh)
    hess_structure!(nlp, rows, cols)
    nlp_allocations[:hess_structure!] = @allocated hess_structure!(nlp, rows, cols)
    x = get_x0(nlp)
    vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
    hess_coord!(nlp, x, vals)
    nlp_allocations[:hess_coord!] = @allocated hess_coord!(nlp, x, vals)
    if get_ncon(nlp) > 0
      y = get_y0(nlp)
      hess_coord!(nlp, x, y, vals)
      nlp_allocations[:hess_lag_coord!] = @allocated hess_coord!(nlp, x, y, vals)
    end
  end

  if !(hprod in exclude) && nlp.meta.hprod_available
    x = get_x0(nlp)
    v = copy(x)
    Hv = similar(x)
    hprod!(nlp, x, v, Hv)
    nlp_allocations[:hprod!] = @allocated hprod!(nlp, x, v, Hv)
    if get_ncon(nlp) > 0
      y = get_y0(nlp)
      hprod!(nlp, x, y, v, Hv)
      nlp_allocations[:hprod_lag!] = @allocated hprod!(nlp, x, y, v, Hv)
    end
  end

  if !(hess_op in exclude) && nlp.meta.hprod_available
    x = get_x0(nlp)
    Hv = similar(x)
    v = copy(x)
    H = hess_op!(nlp, x, Hv)
    mul!(Hv, H, v)
    nlp_allocations[:hess_op_prod!] = @allocated mul!(Hv, H, v)
    if get_ncon(nlp) > 0
      y = get_y0(nlp)
      H = hess_op!(nlp, x, y, Hv)
      mul!(Hv, H, v)
      nlp_allocations[:hess_lag_op_prod!] = @allocated mul!(Hv, H, v)
    end
  end

  if get_ncon(nlp) > 0 && !(cons in exclude)
    x = get_x0(nlp)
    c = Vector{eltype(x)}(undef, get_ncon(nlp))
    cons!(nlp, x, c)
    nlp_allocations[:cons!] = @allocated cons!(nlp, x, c)
  end

  if get_ncon(nlp) > 0 && !(jac in exclude) && nlp.meta.jac_available
    rows = Vector{Int}(undef, nlp.meta.nnzj)
    cols = Vector{Int}(undef, nlp.meta.nnzj)
    jac_structure!(nlp, rows, cols)
    nlp_allocations[:jac_structure!] = @allocated jac_structure!(nlp, rows, cols)
    x = get_x0(nlp)
    vals = Vector{eltype(x)}(undef, nlp.meta.nnzj)
    jac_coord!(nlp, x, vals)
    nlp_allocations[:jac_coord!] = @allocated jac_coord!(nlp, x, vals)
  end

  if get_ncon(nlp) > 0 && !(jprod in exclude) && nlp.meta.jprod_available
    x = get_x0(nlp)
    v = copy(x)
    Jv = Vector{eltype(x)}(undef, get_ncon(nlp))
    jprod!(nlp, x, v, Jv)
    nlp_allocations[:jprod!] = @allocated jprod!(nlp, x, v, Jv)
  end

  if get_ncon(nlp) > 0 && !(jtprod in exclude) && nlp.meta.jtprod_available
    x = get_x0(nlp)
    v = copy(get_y0(nlp))
    Jtv = similar(x)
    jtprod!(nlp, x, v, Jtv)
    nlp_allocations[:jtprod!] = @allocated jtprod!(nlp, x, v, Jtv)
  end

  if get_ncon(nlp) > 0 && !(jac_op in exclude) && nlp.meta.jprod_available && nlp.meta.jtprod_available
    x = get_x0(nlp)
    Jtv = similar(x)
    Jv = Vector{eltype(x)}(undef, get_ncon(nlp))

    v = copy(x)
    w = copy(get_y0(nlp))
    J = jac_op!(nlp, x, Jv, Jtv)
    mul!(Jv, J, v)
    nlp_allocations[:jac_op_prod!] = @allocated mul!(Jv, J, v)
    Jt = J'
    mul!(Jtv, Jt, w)
    nlp_allocations[:jac_op_transpose_prod!] = @allocated mul!(Jtv, Jt, w)
  end

  for type in (:nln, :lin)
    nn = type == :lin ? nlp.meta.nlin : nlp.meta.nnln
    nnzj = type == :lin ? nlp.meta.lin_nnzj : nlp.meta.nln_nnzj
    if !linear_api || (nn == 0)
      continue
    end

    if !(cons in exclude)
      x = get_x0(nlp)
      c = Vector{eltype(x)}(undef, nn)
      fun = Symbol(:cons_, type, :!)
      eval(fun)(nlp, x, c)
      nlp_allocations[fun] = @allocated eval(fun)(nlp, x, c)
    end

    if !(jac in exclude) && nlp.meta.jac_available
      rows = Vector{Int}(undef, nnzj)
      cols = Vector{Int}(undef, nnzj)
      fun = type == :lin ? jac_lin_structure! : jac_nln_structure! # eval(fun) would allocate here
      fun(nlp, rows, cols)
      nlp_allocations[Symbol(fun)] = @allocated fun(nlp, rows, cols)
      x = get_x0(nlp)
      vals = Vector{eltype(x)}(undef, nnzj)
      fun = Symbol(:jac_, type, :_coord!)
      eval(fun)(nlp, x, vals)
      nlp_allocations[fun] = @allocated eval(fun)(nlp, x, vals)
    end

    if !(jprod in exclude) && nlp.meta.jprod_available
      x = get_x0(nlp)
      v = copy(x)
      Jv = Vector{eltype(x)}(undef, nn)
      fun = Symbol(:jprod_, type, :!)
      eval(fun)(nlp, x, v, Jv)
      nlp_allocations[fun] = @allocated eval(fun)(nlp, x, v, Jv)
    end

    if !(jtprod in exclude) && nlp.meta.jtprod_available
      x = get_x0(nlp)
      v = copy(get_y0(nlp)[1:nn])
      Jtv = similar(x)
      fun = Symbol(:jtprod_, type, :!)
      eval(fun)(nlp, x, v, Jtv)
      nlp_allocations[fun] = @allocated eval(fun)(nlp, x, v, Jtv)
    end

    if !(jac_op in exclude) && nlp.meta.jprod_available && nlp.meta.jtprod_available
      x = get_x0(nlp)
      Jtv = similar(x)
      Jv = Vector{eltype(x)}(undef, nn)

      v = copy(x)
      w = randn(eltype(x), nn)
      fun = Symbol(:jac_, type, :_op!)
      if type == :lin
        J = jac_lin_op!(nlp, x, Jv, Jtv)
        mul!(Jv, J, v)
        nlp_allocations[Symbol(:jac_lin_op_prod!)] = @allocated mul!(Jv, J, v)
        Jt = J'
        mul!(Jtv, Jt, w)
        nlp_allocations[Symbol(:jac_lin_op_transpose_prod!)] = @allocated mul!(Jtv, Jt, w)
      else
        J = jac_nln_op!(nlp, x, Jv, Jtv)
        mul!(Jv, J, v)
        nlp_allocations[Symbol(:jac_nln_op_prod!)] = @allocated mul!(Jv, J, v)
        Jt = J'
        mul!(Jtv, Jt, w)
        nlp_allocations[Symbol(:jac_nln_op_transpose_prod!)] = @allocated mul!(Jtv, Jt, w)
      end
    end
  end
  return nlp_allocations
end

"""
    test_obj_grad!(nlp_allocations, nlp::AbstractNLPModel, exclude)

Update `nlp_allocations` with allocations of the in-place `obj` and `grad` functions.

For `AbstractNLSModel`, this uses `obj` and `grad` with pre-allocated residual.
"""
function test_obj_grad!(nlp_allocations, nlp::AbstractNLPModel, exclude)
  if !(obj in exclude)
    x = get_x0(nlp)
    obj(nlp, x)
    nlp_allocations[:obj] = @allocated obj(nlp, x)
  end

  if !(grad in exclude) && nlp.meta.grad_available
    x = get_x0(nlp)
    g = similar(x)
    grad!(nlp, x, g)
    nlp_allocations[:grad!] = @allocated grad!(nlp, x, g)
  end
  return nlp_allocations
end

function test_obj_grad!(nlp_allocations, nls::AbstractNLSModel, exclude)
  if !(obj in exclude)
    x = get_x0(nls)
    Fx = Vector{eltype(x)}(undef, get_nequ(nls))
    obj(nls, x, Fx)
    nlp_allocations[:obj] = @allocated obj(nls, x, Fx)
  end

  if !(grad in exclude) && nls_meta(nls).jtprod_residual_available
    x = get_x0(nls)
    Fx = Vector{eltype(x)}(undef, get_nequ(nls))
    g = similar(x)
    grad!(nls, x, g, Fx)
    nlp_allocations[:grad!] = @allocated grad!(nls, x, g, Fx)
  end
  return nlp_allocations
end

"""
    test_allocs_nlsmodels(nls::AbstractNLSModel; exclude = [])

Returns a `Dict` containing allocations of the in-place functions specialized to nonlinear least squares of NLPModel API.

The keyword `exclude` takes a Array of Function to be excluded from the tests. 
Use `hess_residual` (resp. `jac_residual`) to exclude `hess_residual_coord` and `hess_residual_structure` (resp. `jac_residual_coord` and `jac_residual_structure`).
The hessian-vector product is tested for all the component of the residual function, so exclude `hprod_residual` and `hess_op_residual` if you want to avoid this.
"""
function test_allocs_nlsmodels(nls::AbstractNLSModel; exclude = [])
  nls_allocations = Dict(
    :residual! => NaN,
    :hess_structure_residual! => NaN,
    :hess_coord_residual! => NaN,
    :hprod_residual! => NaN,
    :hess_op_residual_prod! => NaN,
    :jac_structure_residual! => NaN,
    :jac_coord_residual! => NaN,
    :jprod_residual! => NaN,
    :jtprod_residual! => NaN,
    :jac_op_residual_prod! => NaN,
    :jac_op_residual_transpose_prod! => NaN,
  )

  if !(residual in exclude)
    x = get_x0(nls)
    Fx = Vector{eltype(x)}(undef, get_nequ(nls))
    residual!(nls, x, Fx)
    nls_allocations[:residual!] = @allocated residual!(nls, x, Fx)
  end

  if !(jac_residual in exclude) && nls_meta(nls).jac_residual_available
    rows = Vector{Int}(undef, nls.nls_meta.nnzj)
    cols = Vector{Int}(undef, nls.nls_meta.nnzj)
    jac_structure_residual!(nls, rows, cols)
    nls_allocations[:jac_structure_residual!] = @allocated jac_structure_residual!(nls, rows, cols)
    x = get_x0(nls)
    vals = Vector{eltype(x)}(undef, nls.nls_meta.nnzj)
    jac_coord_residual!(nls, x, vals)
    nls_allocations[:jac_coord_residual!] = @allocated jac_coord_residual!(nls, x, vals)
  end

  if !(jprod_residual in exclude) && nls_meta(nls).jprod_residual_available
    x = get_x0(nls)
    v = copy(x)
    Jv = Vector{eltype(x)}(undef, get_nequ(nls))
    jprod_residual!(nls, x, v, Jv)
    nls_allocations[:jprod_residual!] = @allocated jprod_residual!(nls, x, v, Jv)
  end

  if !(jtprod_residual in exclude) && nls_meta(nls).jtprod_residual_available
    x = get_x0(nls)
    w = zeros(eltype(x), get_nequ(nls))
    Jtv = similar(x)
    jtprod_residual!(nls, x, w, Jtv)
    nls_allocations[:jtprod_residual!] = @allocated jtprod_residual!(nls, x, w, Jtv)
  end

  if !(jac_op_residual in exclude) &&
    nls_meta(nls).jprod_residual_available &&
    nls_meta(nls).jtprod_residual_available
    x = get_x0(nls)
    Jtv = similar(x)
    Jv = Vector{eltype(x)}(undef, get_nequ(nls))

    v = copy(x)
    w = zeros(eltype(x), get_nequ(nls))
    J = jac_op_residual!(nls, x, Jv, Jtv)
    mul!(Jv, J, v)
    nls_allocations[:jac_op_residual_prod!] = @allocated mul!(Jv, J, v)
    Jt = J'
    mul!(Jtv, Jt, w)
    nls_allocations[:jac_op_residual_transpose_prod!] = @allocated mul!(Jtv, Jt, w)
  end

  if !(hess_residual in exclude) && nls_meta(nls).hess_residual_available
    rows = Vector{Int}(undef, nls.nls_meta.nnzh)
    cols = Vector{Int}(undef, nls.nls_meta.nnzh)
    hess_structure_residual!(nls, rows, cols)
    nls_allocations[:hess_structure_residual!] =
      @allocated hess_structure_residual!(nls, rows, cols)
    x = get_x0(nls)
    v = ones(eltype(x), get_nequ(nls))
    vals = Vector{eltype(x)}(undef, nls.nls_meta.nnzh)
    hess_coord_residual!(nls, x, v, vals)
    nls_allocations[:hess_coord_residual!] = @allocated hess_coord_residual!(nls, x, v, vals)
  end

  for i = 1:get_nequ(nls)
    if !(hprod_residual in exclude) && nls_meta(nls).hprod_residual_available
      x = get_x0(nls)
      v = copy(x)
      Hv = similar(x)
      hprod_residual!(nls, x, i, v, Hv)
      nls_allocations[:hprod_residual!] = @allocated hprod_residual!(nls, x, i, v, Hv)
    end

    if !(hess_op_residual in exclude) && nls_meta(nls).jprod_residual_available
      x = get_x0(nls)
      Hv = similar(x)
      v = copy(x)
      H = hess_op_residual!(nls, x, i, Hv)
      mul!(Hv, H, v)
      nls_allocations[:hess_op_residual_prod!] = @allocated mul!(Hv, H, v)
    end
  end
  return nls_allocations
end

function NLPModels.histline(s::String, v::Integer, maxv::Integer)
  @assert 0 ≤ v ≤ maxv
  λ = maxv == 0 ? 0 : ceil(Int, 20 * v / maxv)
  return @sprintf("%31s: %s %-6s", s, "█"^λ * "⋅"^(20 - λ), v)
end

"""
    print_nlp_allocations([io::IO = stdout], nlp::AbstractNLPModel, table::Dict; only_nonzeros::Bool = false)
    print_nlp_allocations([io::IO = stdout], nlp::AbstractNLPModel; kwargs...)

Print in a convenient way the result of `test_allocs_nlpmodels(nlp)`.

The keyword arguments may contain:
- `only_nonzeros::Bool`: shows only non-zeros if true.
- `linear_api::Bool`: checks the functions specific to linear and nonlinear constraints, see [`test_allocs_nlpmodels`](@ref).
- `exclude` takes a Array of Function to be excluded from the tests, see [`test_allocs_nlpmodels`](@ref).
"""
function print_nlp_allocations(nlp::AbstractNLPModel, table::Dict; kwargs...)
  return print_nlp_allocations(stdout, nlp, table; kwargs...)
end

function print_nlp_allocations(nlp::AbstractNLPModel; kwargs...)
  return print_nlp_allocations(stdout, nlp; kwargs...)
end

function print_nlp_allocations(
  io,
  nlp::AbstractNLPModel;
  only_nonzeros::Bool = false,
  linear_api = false,
  kwargs...,
)
  table = test_allocs_nlpmodels(nlp; linear_api = linear_api, kwargs...)
  return print_nlp_allocations(io, nlp, table, only_nonzeros = only_nonzeros)
end

function print_nlp_allocations(
  io,
  nlp::AbstractNLSModel;
  only_nonzeros::Bool = false,
  linear_api = false,
  kwargs...,
)
  table_nlp = test_allocs_nlpmodels(nlp; linear_api = linear_api, kwargs...)
  table_nls = test_allocs_nlsmodels(nlp; kwargs...)
  table = merge(table_nlp, table_nls)
  return print_nlp_allocations(io, nlp, table, only_nonzeros = only_nonzeros)
end

function print_nlp_allocations(io, nlp::AbstractNLPModel, table::Dict; only_nonzeros::Bool = false)
  for k in keys(table)
    if isnan(table[k])
      pop!(table, k)
    end
  end
  if only_nonzeros
    for k in keys(table)
      if table[k] == 0
        pop!(table, k)
      end
    end
  end
  println(io, "  Problem name: $(get_name(nlp))")
  lines = NLPModels.lines_of_hist(keys(table), values(table))
  println(io, join(lines, "\n") * "\n")
  return table
end

"""
    test_zero_allocations(table::Dict, name::String = "Generic")
    test_zero_allocations(nlp::AbstractNLPModel; kwargs...)

Test wether the result of `test_allocs_nlpmodels(nlp)` and `test_allocs_nlsmodels(nlp)` is 0.
"""
function test_zero_allocations(nlp::AbstractNLPModel; kwargs...)
  table = test_allocs_nlpmodels(nlp; kwargs...)
  return test_zero_allocations(table, get_name(nlp))
end

function test_zero_allocations(nlp::AbstractNLSModel; linear_api = linear_api, kwargs...)
  table_nlp = test_allocs_nlpmodels(nlp; linear_api = linear_api, kwargs...)
  table_nls = test_allocs_nlsmodels(nlp; kwargs...)
  table = merge(table_nlp, table_nls)
  return test_zero_allocations(table, get_name(nlp))
end

function test_zero_allocations(table::Dict, name::String = "Generic")
  @testset "Test 0-allocations of NLPModel API for $name" begin
    for k in keys(table)
      if !isnan(table[k])
        (table[k] != 0) && @info "Allocation of $k is $(table[k])"
        @test table[k] == 0
      end
    end
  end
end
