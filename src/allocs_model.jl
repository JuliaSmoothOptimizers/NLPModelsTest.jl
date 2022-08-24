export test_allocs_nlpmodels, print_nlp_allocations

"""
    test_allocs_nlpmodels(nlp::AbstractNLPModel; exclude = [])

Returns a `Dict` containing allocations of the in-place functions of NLPModel API.

The keyword `exclude` takes a Array of Function to be excluded from the tests. Use `hess` (resp. `jac`) to exclude `hess_coord` and `hess_structure` (resp. `jac_coord` and `jac_structure`).
"""
function test_allocs_nlpmodels(nlp::AbstractNLPModel; exclude = [])
  nlp_allocations = Dict(
    :obj => NaN,
    :grad! => NaN,
    :hess_structure! => NaN,
    :hess_coord! => NaN,
    :hess_op! => NaN,
    :hess_op_prod! => NaN,
    :cons! => NaN,
    :jac_structure! => NaN,
    :jac_coord! => NaN,
    :jac_op! => NaN,
    :jac_op_prod! => NaN,
    :jac_op_transpose_prod! => NaN,
    :hess_lag_coord! => NaN,
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
    rows = Vector{Int}(undef, get_nnzh(nlp))
    cols = Vector{Int}(undef, get_nnzh(nlp))
    hess_structure!(nlp, rows, cols)
    nlp_allocations[:hess_structure!] = @allocated hess_structure!(nlp, rows, cols)
    x = get_x0(nlp)
    vals = Vector{eltype(x)}(undef, get_nnzh(nlp))
    hess_coord!(nlp, x, vals)
    nlp_allocations[:hess_coord!] = @allocated hess_coord!(nlp, x, vals)
    if get_ncon(nlp) > 0
        y = get_y0(nlp)
        hess_coord!(nlp, x, y, vals)
        nlp_allocations[:hess_lag_coord!] = @allocated hess_coord!(nlp, x, y, vals)
    end
  end
  if !(hess_op in exclude)
    # First we test the definition of the operator
    x = get_x0(nlp)
    Hv = similar(x)
    hess_op!(nlp, x, Hv)
    nlp_allocations[:hess_op!] = @allocated hess_op!(nlp, x, Hv)
    if get_ncon(nlp) > 0
        y = get_y0(nlp)
        hess_op!(nlp, x, y, Hv)
        nlp_allocations[:hess_lag_op!] = @allocated hess_op!(nlp, x, y, Hv)
    end
    # Then, we test the product
    v = copy(x)
    H = hess_op!(nlp, x, Hv)
    H * v
    nlp_allocations[:hess_op_prod!] = @allocated H * v
    if get_ncon(nlp) > 0
        y = get_y0(nlp)
        H = hess_op!(nlp, x, y, Hv)
        H * v
        nlp_allocations[:hess_lag_op_prod!] = @allocated H * v
    end
  end

  if get_ncon(nlp) > 0 && !(cons in exclude)
    x = get_x0(nlp)
    c = Vector{eltype(x)}(undef, get_ncon(nlp))
    cons!(nlp, x, c)
    nlp_allocations[:cons!] = @allocated cons!(nlp, x, c)
  end
  if get_ncon(nlp) > 0 && !(jac in exclude)
    rows = Vector{Int}(undef, get_nnzj(nlp))
    cols = Vector{Int}(undef, get_nnzj(nlp))
    jac_structure!(nlp, rows, cols)
    nlp_allocations[:jac_structure!] = @allocated jac_structure!(nlp, rows, cols)
    x = get_x0(nlp)
    vals = Vector{eltype(x)}(undef, get_nnzj(nlp))
    jac_coord!(nlp, x, vals)
    nlp_allocations[:jac_coord!] = @allocated jac_coord!(nlp, x, vals)
  end
  if get_ncon(nlp) > 0 && !(jac_op in exclude)
    # First we test the definition of the operator
    x = get_x0(nlp)
    Jtv = similar(x)
    Jv = Vector{eltype(x)}(undef, get_ncon(nlp))
    jac_op!(nlp, x, Jv, Jtv)
    nlp_allocations[:jac_op!] = @allocated jac_op!(nlp, x, Jv, Jtv)

    v = copy(x)
    w = copy(get_y0(nlp))
    J = jac_op!(nlp, x, Jv, Jtv)
    J * v
    nlp_allocations[:jac_op_prod!] = @allocated J * v
    J' * w
    nlp_allocations[:jac_op_transpose_prod!] = @allocated J' * w
  end
  return nlp_allocations
end

function NLPModels.histline(s, v, maxv)
  @assert 0 ≤ v ≤ maxv
  λ = maxv == 0 ? 0 : ceil(Int, 20 * v / maxv)
  return @sprintf("%22s: %s %-6s", s, "█"^λ * "⋅"^(20 - λ), v)
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
