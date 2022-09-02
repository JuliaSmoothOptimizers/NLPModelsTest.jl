export test_allocs_nlpmodels, test_allocs_nlsmodels, print_nlp_allocations

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

  if !(obj in exclude)
    x = get_x0(nlp)
    obj(nlp, x)
    nlp_allocations[:obj] = @allocated obj(nlp, x)
  end
  if !(grad in exclude)
    x = get_x0(nlp)
    g = similar(x)
    grad!(nlp, x, g)
    nlp_allocations[:grad!] = @allocated grad!(nlp, x, g)
  end
  if !(hess in exclude)
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
  if !(hprod in exclude)
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
  if !(hess_op in exclude)
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
  if get_ncon(nlp) > 0 && !(jac in exclude)
    rows = Vector{Int}(undef, nlp.meta.nnzj)
    cols = Vector{Int}(undef, nlp.meta.nnzj)
    @show "Do we pass here?"
    jac_structure!(nlp, rows, cols)
    nlp_allocations[:jac_structure!] = @allocated jac_structure!(nlp, rows, cols)
    x = get_x0(nlp)
    vals = Vector{eltype(x)}(undef, nlp.meta.nnzj)
    jac_coord!(nlp, x, vals)
    nlp_allocations[:jac_coord!] = @allocated jac_coord!(nlp, x, vals)
  end
  if get_ncon(nlp) > 0 && !(jprod in exclude)
    x = get_x0(nlp)
    v = copy(x)
    Jv = Vector{eltype(x)}(undef, get_ncon(nlp))
    jprod!(nlp, x, v, Jv)
    nlp_allocations[:jprod!] = @allocated jprod!(nlp, x, v, Jv)
  end
  if get_ncon(nlp) > 0 && !(jtprod in exclude)
    x = get_x0(nlp)
    v = copy(get_y0(nlp))
    Jtv = similar(x)
    jtprod!(nlp, x, v, Jtv)
    nlp_allocations[:jtprod!] = @allocated jtprod!(nlp, x, v, Jtv)
  end
  if get_ncon(nlp) > 0 && !(jac_op in exclude)
    x = get_x0(nlp)
    Jtv = similar(x)
    Jv = Vector{eltype(x)}(undef, get_ncon(nlp))

    v = copy(x)
    w = copy(get_y0(nlp))
    J = jac_op!(nlp, x, Jv, Jtv)
    mul!(Jv, J, v)
    nlp_allocations[:jac_op_prod!] = @allocated mul!(Jv, J, v)
    mul!(Jtv, J', w)
    nlp_allocations[:jac_op_transpose_prod!] = @allocated mul!(Jtv, J', w)
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
    if !(jac in exclude)
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
    if !(jprod in exclude)
      x = get_x0(nlp)
      v = copy(x)
      Jv = Vector{eltype(x)}(undef, nn)
      fun = Symbol(:jprod_, type, :!)
      eval(fun)(nlp, x, v, Jv)
      nlp_allocations[fun] = @allocated eval(fun)(nlp, x, v, Jv)
    end
    if !(jtprod in exclude)
      x = get_x0(nlp)
      v = copy(get_y0(nlp)[1:nn])
      Jtv = similar(x)
      fun = Symbol(:jtprod_, type, :!)
      eval(fun)(nlp, x, v, Jtv)
      nlp_allocations[fun] = @allocated eval(fun)(nlp, x, v, Jtv)
    end
    if !(jac_op in exclude)
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
        mul!(Jtv, J', w)
        nlp_allocations[Symbol(:jac_lin_op_transpose_prod!)] = @allocated mul!(Jtv, J', w)
      else
        J = jac_nln_op!(nlp, x, Jv, Jtv)
        mul!(Jv, J, v)
        nlp_allocations[Symbol(:jac_nln_op_prod!)] = @allocated mul!(Jv, J, v)
        mul!(Jtv, J', w)
        nlp_allocations[Symbol(:jac_nln_op_transpose_prod!)] = @allocated mul!(Jtv, J', w)
      end
    end
  end
  return nlp_allocations
end

"""
    test_allocs_nlsmodels(nlp::AbstractNLSModel; exclude = [])

Returns a `Dict` containing allocations of the in-place functions specialized to nonlinear least squares of NLPModel API.

The keyword `exclude` takes a Array of Function to be excluded from the tests. 
Use `hess_residual` (resp. `jac_residual`) to exclude `hess_residual_coord` and `hess_residual_structure` (resp. `jac_residual_coord` and `jac_residual_structure`).
The hessian-vector product is tested for all the component of the residual function, so exclude `hprod_residual` and `hess_op_residual` if you want to avoid this.
"""
function test_allocs_nlsmodels(nlp::AbstractNLSModel; exclude = [])
  nlp_allocations = Dict(
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
    x = get_x0(nlp)
    Fx = Vector{eltype(x)}(undef, get_nequ(nlp))
    residual!(nlp, x, Fx)
    nlp_allocations[:residual!] = @allocated residual!(nlp, x, Fx)
  end
  if !(jac_residual in exclude)
    rows = Vector{Int}(undef, nlp.nls_meta.nnzj)
    cols = Vector{Int}(undef, nlp.nls_meta.nnzj)
    jac_structure_residual!(nlp, rows, cols)
    nlp_allocations[:jac_structure_residual!] = @allocated jac_structure_residual!(nlp, rows, cols)
    x = get_x0(nlp)
    vals = Vector{eltype(x)}(undef, nlp.nls_meta.nnzj)
    jac_coord_residual!(nlp, x, vals)
    nlp_allocations[:jac_coord_residual!] = @allocated jac_coord_residual!(nlp, x, vals)
  end
  if !(jprod_residual in exclude)
    x = get_x0(nlp)
    v = copy(x)
    Jv = Vector{eltype(x)}(undef, get_nequ(nlp))
    jprod_residual!(nlp, x, v, Jv)
    nlp_allocations[:jprod_residual!] = @allocated jprod_residual!(nlp, x, v, Jv)
  end
  if !(jtprod_residual in exclude)
    x = get_x0(nlp)
    w = zeros(eltype(x), get_nequ(nlp))
    Jtv = similar(x)
    jtprod_residual!(nlp, x, w, Jtv)
    nlp_allocations[:jtprod_residual!] = @allocated jtprod_residual!(nlp, x, w, Jtv)
  end
  if !(jac_op_residual in exclude)
    x = get_x0(nlp)
    Jtv = similar(x)
    Jv = Vector{eltype(x)}(undef, get_nequ(nlp))

    v = copy(x)
    w = zeros(eltype(x), get_nequ(nlp))
    J = jac_op_residual!(nlp, x, Jv, Jtv)
    mul!(Jv, J, v)
    nlp_allocations[:jac_op_residual_prod!] = @allocated mul!(Jv, J, v)
    mul!(Jtv, J', w)
    nlp_allocations[:jac_op_residual_transpose_prod!] = @allocated mul!(Jtv, J', w)
  end
  if !(hess_residual in exclude)
    rows = Vector{Int}(undef, nlp.nls_meta.nnzh)
    cols = Vector{Int}(undef, nlp.nls_meta.nnzh)
    hess_structure_residual!(nlp, rows, cols)
    nlp_allocations[:hess_structure_residual!] =
      @allocated hess_structure_residual!(nlp, rows, cols)
    x = get_x0(nlp)
    v = ones(eltype(x), get_nequ(nlp))
    vals = Vector{eltype(x)}(undef, nlp.nls_meta.nnzh)
    hess_coord_residual!(nlp, x, v, vals)
    nlp_allocations[:hess_coord_residual!] = @allocated hess_coord_residual!(nlp, x, v, vals)
  end
  for i = 1:get_nequ(nlp)
    if !(hprod_residual in exclude)
      x = get_x0(nlp)
      v = copy(x)
      Hv = similar(x)
      hprod_residual!(nlp, x, i, v, Hv)
      nlp_allocations[:hprod_residual!] = @allocated hprod_residual!(nlp, x, i, v, Hv)
    end
    if !(hess_op_residual in exclude)
      x = get_x0(nlp)
      Hv = similar(x)
      v = copy(x)
      H = hess_op_residual!(nlp, x, i, Hv)
      mul!(Hv, H, v)
      nlp_allocations[:hess_op_residual_prod!] = @allocated mul!(Hv, H, v)
    end
  end
  return nlp_allocations
end

function NLPModels.histline(s, v, maxv)
  @assert 0 ≤ v ≤ maxv
  λ = maxv == 0 ? 0 : ceil(Int, 20 * v / maxv)
  return @sprintf("%27s: %s %-6s", s, "█"^λ * "⋅"^(20 - λ), v)
end

"""
print_nlp_allocations([io::IO = stdout], nlp::AbstractNLPModel, table::Dict)

Print in a convenient way the result of `test_allocs_nlpmodels(nlp)`
"""
function print_nlp_allocations(nlp::AbstractNLPModel, table::Dict)
  return print_nlp_allocations(stdout, nlp, table)
end

function print_nlp_allocations(io, nlp::AbstractNLPModel, table::Dict)
  for k in keys(table)
    if isnan(table[k])
      pop!(table, k)
    end
  end
  println(io, "  Problem name: $(get_name(nlp))")
  lines = NLPModels.lines_of_hist(keys(table), values(table))
  println(io, join(lines, "\n") * "\n")
  return table
end
