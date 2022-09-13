var documenterSearchIndex = {"docs":
[{"location":"problems/#Problems","page":"Problems","title":"Problems","text":"","category":"section"},{"location":"problems/","page":"Problems","title":"Problems","text":"We have implemented a few problems to be used with the consistency checks, explained here.","category":"page"},{"location":"problems/","page":"Problems","title":"Problems","text":"NLP problems:\nBROWNDEN\nHS5\nHS6\nHS10\nHS11\nHS13\nHS14\nLINCON\nLINSV\nMGH01Feas\nNLS problems:\nLLS\nMGH01\nNLSHS20\nNLSLC","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [NLPModelsTest]","category":"page"},{"location":"reference/#NLPModelsTest.BROWNDEN","page":"Reference","title":"NLPModelsTest.BROWNDEN","text":"nlp = BROWNDEN()\n\nBrown and Dennis function.\n\nSource: Problem 16 in\nJ.J. Moré, B.S. Garbow and K.E. Hillstrom,\n\"Testing Unconstrained Optimization Software\",\nACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981\n\nclassification SUR2-AN-4-0\n\nmin_x  sum_i=1^20 left(left(x_1 + tfraci5 x_2 - e^i  5right)^2\n+ left(x_3 + sin(tfraci5) x_4 - cos(tfraci5)right)^2right)^2\n\nStarting point: [25.0; 5.0; -5.0; -1.0]\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.HS10","page":"Reference","title":"NLPModelsTest.HS10","text":"nlp = HS10()\n\nProblem 10 in the Hock-Schittkowski suite\n\nbeginaligned\nmin quad  x_1 - x_2 \ntexts to quad  -3x_1^2 + 2x_1 x_2 - x_2^2 + 1 geq 0\nendaligned\n\nStarting point: [-10; 10].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.HS11","page":"Reference","title":"NLPModelsTest.HS11","text":"nlp = HS11()\n\nProblem 11 in the Hock-Schittkowski suite\n\nbeginaligned\nmin quad  (x_1 - 5)^2 + x_2^2 - 25 \ntexts to quad  0 leq -x_1^2 + x_2\nendaligned\n\nStarting point: [-4.9; 0.1].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.HS13","page":"Reference","title":"NLPModelsTest.HS13","text":"nlp = HS13()\n\nProblem 13 in the Hock-Schittkowski suite\n\nbeginaligned\nmin quad  (x_1 - 2)^2 + x_2^2 \ntexts to quad  (1 - x_1)^3 - x_2 geq 0\nquad  0 leq x_1 \n 0 leq x_2\nendaligned\n\nStarting point: [-2; -2].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.HS14","page":"Reference","title":"NLPModelsTest.HS14","text":"nlp = HS14()\n\nProblem 14 in the Hock-Schittkowski suite\n\nbeginaligned\nmin quad  (x_1 - 2)^2 + (x_2 - 1)^2 \ntexts to quad  x_1 - 2x_2 = -1 \n -tfrac14 x_1^2 - x_2^2 + 1 geq 0\nendaligned\n\nStarting point: [2; 2].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.HS5","page":"Reference","title":"NLPModelsTest.HS5","text":"nlp = HS5()\n\nProblem 5 in the Hock-Schittkowski suite\n\nbeginaligned\nmin quad  sin(x_1 + x_2) + (x_1 - x_2)^2 - tfrac32x_1 + tfrac52x_2 + 1 \ntexts to quad  -15 leq x_1 leq 4 \n -3 leq x_2 leq 3\nendaligned\n\nStarting point: [0.0; 0.0].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.HS6","page":"Reference","title":"NLPModelsTest.HS6","text":"nlp = HS6()\n\nProblem 6 in the Hock-Schittkowski suite\n\nbeginaligned\nmin quad  (1 - x_1)^2 \ntexts to quad  10 (x_2 - x_1^2) = 0\nendaligned\n\nStarting point: [-1.2; 1.0].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.LINCON","page":"Reference","title":"NLPModelsTest.LINCON","text":"nlp = LINCON()\n\nLinearly constrained problem\n\nbeginaligned\nmin quad  (i + x_i^4) \ntexts to quad  x_15 = 0 \n x_10 + 2x_11 + 3x_12 geq 1 \n x_13 - x_14 leq 16 \n -11 leq 5x_8 - 6x_9 leq 9 \n -2x_7 = -1 \n 4x_6 = 1 \n x_1 + 2x_2 geq -5 \n 3x_1 + 4x_2 geq -6 \n 9x_3 leq 1 \n 12x_4 leq 2 \n 15x_5 leq 3\nendaligned\n\nStarting point: zeros(15).\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.LINSV","page":"Reference","title":"NLPModelsTest.LINSV","text":"nlp = LINSV()\n\nLinear problem\n\nbeginaligned\nmin quad  x_1 \ntexts to quad  x_1 + x_2 geq 3 \n x_2 geq 1\nendaligned\n\nStarting point: [0; 0].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.LLS","page":"Reference","title":"NLPModelsTest.LLS","text":"nls = LLS()\n\nLinear least squares\n\nbeginaligned\nmin quad  tfrac12 F(x) ^2 \ntexts to quad  x_1 + x_2 geq 0\nendaligned\n\nwhere\n\nF(x) = beginbmatrix\nx_1 - x_2 \nx_1 + x_2 - 2 \nx_2 - 2\nendbmatrix\n\nStarting point: [0; 0].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.MGH01","page":"Reference","title":"NLPModelsTest.MGH01","text":"nls = MGH01()\n\nRosenbrock function in nonlinear least squares format\n\nSource: Problem 1 in\nJ.J. Moré, B.S. Garbow and K.E. Hillstrom,\n\"Testing Unconstrained Optimization Software\",\nACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981\n\nbeginaligned\nmin quad  tfrac12 F(x) ^2\nendaligned\n\nwhere\n\nF(x) = beginbmatrix\n1 - x_1 \n10 (x_2 - x_1^2)\nendbmatrix\n\nStarting point: [-1.2; 1].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.MGH01Feas","page":"Reference","title":"NLPModelsTest.MGH01Feas","text":"nlp = MGH01Feas()\n\nRosenbrock function in feasibility format\n\nSource: Problem 1 in\nJ.J. Moré, B.S. Garbow and K.E. Hillstrom,\n\"Testing Unconstrained Optimization Software\",\nACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981\n\nbeginaligned\nmin quad  0 \ntexts to quad  x_1 = 1 \n 10 (x_2 - x_1^2) = 0\nendaligned\n\nStarting point: [-1.2; 1].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.NLSHS20","page":"Reference","title":"NLPModelsTest.NLSHS20","text":"nls = NLSH20()\n\nProblem 20 in the Hock-Schittkowski suite in nonlinear least squares format\n\nbeginaligned\nmin quad  tfrac12 F(x) ^2 \ntexts to quad  x_1 + x_2^2 geq 0 \n x_1^2 + x_2 geq 0 \n x_1^2 + x_2^2 -1 geq 0 \n -05 leq x_1 leq 05\nendaligned\n\nwhere\n\nF(x) = beginbmatrix\n1 - x_1 \n10 (x_2 - x_1^2)\nendbmatrix\n\nStarting point: [-2; 1].\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.NLSLC","page":"Reference","title":"NLPModelsTest.NLSLC","text":"nls = NLSLC()\n\nLinearly constrained nonlinear least squares problem\n\nbeginaligned\nmin quad  tfrac12 F(x) ^2 \ntexts to quad  x_15 = 0 \n x_10 + 2x_11 + 3x_12 geq 1 \n x_13 - x_14 leq 16 \n -11 leq 5x_8 - 6x_9 leq 9 \n -2x_7 = -1 \n 4x_6 = 1 \n x_1 + 2x_2 geq -5 \n 3x_1 + 4x_2 geq -6 \n 9x_3 leq 1 \n 12x_4 leq 2 \n 15x_5 leq 3\nendaligned\n\nwhere\n\nF(x) = beginbmatrix\nx_1^2 - 1 \nx_2^2 - 2^2 \nvdots \nx_15^2 - 15^2\nendbmatrix\n\nStarting point: zeros(15).\n\n\n\n\n\n","category":"type"},{"location":"reference/#NLPModelsTest.check_nlp_dimensions-Tuple{Any}","page":"Reference","title":"NLPModelsTest.check_nlp_dimensions","text":"check_nlp_dimensions(nlp; exclude = [ghjvprod])\n\nMake sure NLP API functions will throw DimensionError if the inputs are not the correct dimension. To make this assertion in your code use\n\n@lencheck size input [more inputs separated by spaces]\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.check_nls_dimensions-Tuple{Any}","page":"Reference","title":"NLPModelsTest.check_nls_dimensions","text":"check_nls_dimensions(nlp; exclude = [])\n\nMake sure NLS API functions will throw DimensionError if the inputs are not the correct dimension. To make this assertion in your code use\n\n@lencheck size input [more inputs separated by spaces]\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.consistent_nlps-Tuple{Any}","page":"Reference","title":"NLPModelsTest.consistent_nlps","text":"consistent_nlps(nlps; exclude=[], rtol=1e-8)\n\nCheck that the all nlps of the vector nlps are consistent, in the sense that\n\nTheir counters are the same.\nTheir meta information is the same.\nThe API functions return the same output given the same input.\n\nIn other words, if you create two models of the same problem, they should be consistent.\n\nThe keyword exclude can be used to pass functions to be ignored, if some of the models don't implement that function.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.consistent_nlss-Tuple{Any}","page":"Reference","title":"NLPModelsTest.consistent_nlss","text":"consistent_nlss(nlps; exclude=[hess, hprod, hess_coord])\n\nCheck that the all nlss of the vector nlss are consistent, in the sense that\n\nTheir counters are the same.\nTheir meta information is the same.\nThe API functions return the same output given the same input.\n\nIn other words, if you create two models of the same problem, they should be consistent.\n\nBy default, the functions hess, hprod and hess_coord (and therefore associated functions) are excluded from this check, since some models don't implement them.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.coord_memory_nlp-Tuple{NLPModels.AbstractNLPModel}","page":"Reference","title":"NLPModelsTest.coord_memory_nlp","text":"coord_memory_nlp(nlp; exclude = [])\n\nCheck that the allocated memory for in place coord methods is sufficiently smaller than their allocating counter parts.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.gradient_check-Tuple{NLPModels.AbstractNLPModel}","page":"Reference","title":"NLPModelsTest.gradient_check","text":"gradient_check(nlp; x=nlp.meta.x0, atol=1e-6, rtol=1e-4)\n\nCheck the first derivatives of the objective at x against centered finite differences.\n\nThis function returns a dictionary indexed by components of the gradient for which the relative error exceeds rtol.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.hessian_check-Tuple{NLPModels.AbstractNLPModel}","page":"Reference","title":"NLPModelsTest.hessian_check","text":"hessian_check(nlp; x=nlp.meta.x0, atol=1e-6, rtol=1e-4, sgn=1)\n\nCheck the second derivatives of the objective and each constraints at x against centered finite differences. This check does not rely on exactness of the first derivatives, only on objective and constraint values.\n\nThe sgn arguments refers to the formulation of the Lagrangian in the problem. It should have a positive value if the Lagrangian is formulated as\n\nL(xy) = f(x) + sum_j yⱼ cⱼ(x)\n\nand a negative value if the Lagrangian is formulated as\n\nL(xy) = f(x) - sum_j yⱼ cⱼ(x)\n\nOnly the sign of sgn is important.\n\nThis function returns a dictionary indexed by functions. The 0-th function is the objective while the k-th function (for k > 0) is the k-th constraint. The values of the dictionary are dictionaries indexed by tuples (i, j) such that the relative error in the second derivative ∂²fₖ/∂xᵢ∂xⱼ exceeds rtol.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.hessian_check_from_grad-Tuple{NLPModels.AbstractNLPModel}","page":"Reference","title":"NLPModelsTest.hessian_check_from_grad","text":"hessian_check_from_grad(nlp; x=nlp.meta.x0, atol=1e-6, rtol=1e-4, sgn=1)\n\nCheck the second derivatives of the objective and each constraints at x against centered finite differences. This check assumes exactness of the first derivatives.\n\nThe sgn arguments refers to the formulation of the Lagrangian in the problem. It should have a positive value if the Lagrangian is formulated as\n\nL(xy) = f(x) + sum_j yⱼ cⱼ(x)\n\nand a negative value if the Lagrangian is formulated as\n\nL(xy) = f(x) - sum_j yⱼ cⱼ(x)\n\nOnly the sign of sgn is important.\n\nThis function returns a dictionary indexed by functions. The 0-th function is the objective while the k-th function (for k > 0) is the k-th constraint. The values of the dictionary are dictionaries indexed by tuples (i, j) such that the relative error in the second derivative ∂²fₖ/∂xᵢ∂xⱼ exceeds rtol.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.jacobian_check-Tuple{NLPModels.AbstractNLPModel}","page":"Reference","title":"NLPModelsTest.jacobian_check","text":"jacobian_check(nlp; x=nlp.meta.x0, atol=1e-6, rtol=1e-4)\n\nCheck the first derivatives of the constraints at x against centered finite differences.\n\nThis function returns a dictionary indexed by (j, i) tuples such that the relative error in the i-th partial derivative of the j-th constraint exceeds rtol.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.multiple_precision_nlp-Tuple{Any}","page":"Reference","title":"NLPModelsTest.multiple_precision_nlp","text":"multiple_precision_nlp(nlp_from_T; precisions=[...], exclude = [ghjvprod])\n\nCheck that the NLP API functions output type are the same as the input. In other words, make sure that the model handles multiple precisions.\n\nThe input nlp_from_T is a function that returns an nlp from a type T. The array precisions are the tested floating point types. Defaults to [Float16, Float32, Float64, BigFloat].\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.multiple_precision_nls-Tuple{Any}","page":"Reference","title":"NLPModelsTest.multiple_precision_nls","text":"multiple_precision_nls(nls_from_T; precisions=[...], exclude = [])\n\nCheck that the NLS API functions output type are the same as the input. In other words, make sure that the model handles multiple precisions.\n\nThe input nls_from_T is a function that returns an nls from a type T. The array precisions are the tested floating point types. Defaults to [Float16, Float32, Float64, BigFloat].\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.print_nlp_allocations-Tuple{NLPModels.AbstractNLPModel, Dict}","page":"Reference","title":"NLPModelsTest.print_nlp_allocations","text":"print_nlp_allocations([io::IO = stdout], nlp::AbstractNLPModel, table::Dict; only_nonzeros::Bool = false)\nprint_nlp_allocations([io::IO = stdout], nlp::AbstractNLPModel; kwargs...)\n\nPrint in a convenient way the result of test_allocs_nlpmodels(nlp).\n\nThe keyword arguments may contain:\n\nonly_nonzeros::Bool: shows only non-zeros if true.\nlinear_api::Bool: checks the functions specific to linear and nonlinear constraints, see test_allocs_nlpmodels.\nexclude takes a Array of Function to be excluded from the tests, see test_allocs_nlpmodels.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.test_allocs_nlpmodels-Tuple{NLPModels.AbstractNLPModel}","page":"Reference","title":"NLPModelsTest.test_allocs_nlpmodels","text":"test_allocs_nlpmodels(nlp::AbstractNLPModel; linear_api = false, exclude = [])\n\nReturns a Dict containing allocations of the in-place functions of NLPModel API.\n\nThe keyword exclude takes a Array of Function to be excluded from the tests. Use hess (resp. jac) to exclude hess_coord and hess_structure (resp. jac_coord and jac_structure). Setting linear_api to true will also checks the functions specific to linear and nonlinear constraints.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.test_allocs_nlsmodels-Tuple{NLPModels.AbstractNLSModel}","page":"Reference","title":"NLPModelsTest.test_allocs_nlsmodels","text":"test_allocs_nlsmodels(nlp::AbstractNLSModel; exclude = [])\n\nReturns a Dict containing allocations of the in-place functions specialized to nonlinear least squares of NLPModel API.\n\nThe keyword exclude takes a Array of Function to be excluded from the tests.  Use hess_residual (resp. jac_residual) to exclude hess_residual_coord and hess_residual_structure (resp. jac_residual_coord and jac_residual_structure). The hessian-vector product is tested for all the component of the residual function, so exclude hprod_residual and hess_op_residual if you want to avoid this.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.test_obj_grad!-Tuple{Any, NLPModels.AbstractNLPModel, Any}","page":"Reference","title":"NLPModelsTest.test_obj_grad!","text":"test_obj_grad!(nlp_allocations, nlp::AbstractNLPModel, exclude)\n\nUpdate nlp_allocations with allocations of the in-place obj and grad functions.\n\nFor AbstractNLSModel, this uses obj and grad with pre-allocated residual.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.test_zero_allocations-Tuple{NLPModels.AbstractNLPModel}","page":"Reference","title":"NLPModelsTest.test_zero_allocations","text":"test_zero_allocations(table::Dict, name::String = \"Generic\")\ntest_zero_allocations(nlp::AbstractNLPModel; kwargs...)\n\nTest wether the result of test_allocs_nlpmodels(nlp) and test_allocs_nlsmodels(nlp) is 0.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.view_subarray_nlp-Tuple{Any}","page":"Reference","title":"NLPModelsTest.view_subarray_nlp","text":"view_subarray_nlp(nlp; exclude = [])\n\nCheck that the API work with views, and that the results is correct.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModelsTest.view_subarray_nls-Tuple{Any}","page":"Reference","title":"NLPModelsTest.view_subarray_nls","text":"view_subarray_nls(nls; exclude = [])\n\nCheck that the API work with views, and that the results is correct.\n\n\n\n\n\n","category":"method"},{"location":"#Home","page":"Home","title":"NLPModelsTest.jl documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides testing functions for packages implementing optimization models using the NLPModels API.","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This packages export commonly used problems and functions to test optimization models using the NLPModels API. There are currently the following tests in this package:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Consistency: Given 2 or more models of the same problem, do they behave the same way?\nMultiple precision: Given a model in a floating point type, do the API functions output have the same type?\nInput dimension check: Do the functions in this model correctly check the input dimensions, and throw the correct error otherwise?\nView subarray support: Check that your model accepts @view subarrays.\nCoord memory: (incomplete) Check that in place version of coord functions don't use too much memory.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The TL;DR section shows an example using these functions.","category":"page"},{"location":"#consistency","page":"Home","title":"Consistency","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Two functions are given, one for NLP problems and another for NLS problems:","category":"page"},{"location":"","page":"Home","title":"Home","text":"consistent_nlps\nconsistent_nlss","category":"page"},{"location":"#NLPModelsTest.consistent_nlps","page":"Home","title":"NLPModelsTest.consistent_nlps","text":"consistent_nlps(nlps; exclude=[], rtol=1e-8)\n\nCheck that the all nlps of the vector nlps are consistent, in the sense that\n\nTheir counters are the same.\nTheir meta information is the same.\nThe API functions return the same output given the same input.\n\nIn other words, if you create two models of the same problem, they should be consistent.\n\nThe keyword exclude can be used to pass functions to be ignored, if some of the models don't implement that function.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModelsTest.consistent_nlss","page":"Home","title":"NLPModelsTest.consistent_nlss","text":"consistent_nlss(nlps; exclude=[hess, hprod, hess_coord])\n\nCheck that the all nlss of the vector nlss are consistent, in the sense that\n\nTheir counters are the same.\nTheir meta information is the same.\nThe API functions return the same output given the same input.\n\nIn other words, if you create two models of the same problem, they should be consistent.\n\nBy default, the functions hess, hprod and hess_coord (and therefore associated functions) are excluded from this check, since some models don't implement them.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"To use them, implement a few or all of these Problems, and call these functions on an array with both the model you created and the model we have here.","category":"page"},{"location":"#Multiple-precision","page":"Home","title":"Multiple precision","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Two functions are given, one for NLP problems and another for NLS problems:","category":"page"},{"location":"","page":"Home","title":"Home","text":"multiple_precision_nlp\nmultiple_precision_nls","category":"page"},{"location":"#NLPModelsTest.multiple_precision_nlp","page":"Home","title":"NLPModelsTest.multiple_precision_nlp","text":"multiple_precision_nlp(nlp_from_T; precisions=[...], exclude = [ghjvprod])\n\nCheck that the NLP API functions output type are the same as the input. In other words, make sure that the model handles multiple precisions.\n\nThe input nlp_from_T is a function that returns an nlp from a type T. The array precisions are the tested floating point types. Defaults to [Float16, Float32, Float64, BigFloat].\n\n\n\n\n\n","category":"function"},{"location":"#NLPModelsTest.multiple_precision_nls","page":"Home","title":"NLPModelsTest.multiple_precision_nls","text":"multiple_precision_nls(nls_from_T; precisions=[...], exclude = [])\n\nCheck that the NLS API functions output type are the same as the input. In other words, make sure that the model handles multiple precisions.\n\nThe input nls_from_T is a function that returns an nls from a type T. The array precisions are the tested floating point types. Defaults to [Float16, Float32, Float64, BigFloat].\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"To use this function simply call it on your model.","category":"page"},{"location":"#Check-dimensions","page":"Home","title":"Check dimensions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Two functions are given, one for NLP problems and another for NLS problems:","category":"page"},{"location":"","page":"Home","title":"Home","text":"check_nlp_dimensions\ncheck_nls_dimensions","category":"page"},{"location":"#NLPModelsTest.check_nlp_dimensions","page":"Home","title":"NLPModelsTest.check_nlp_dimensions","text":"check_nlp_dimensions(nlp; exclude = [ghjvprod])\n\nMake sure NLP API functions will throw DimensionError if the inputs are not the correct dimension. To make this assertion in your code use\n\n@lencheck size input [more inputs separated by spaces]\n\n\n\n\n\n","category":"function"},{"location":"#NLPModelsTest.check_nls_dimensions","page":"Home","title":"NLPModelsTest.check_nls_dimensions","text":"check_nls_dimensions(nlp; exclude = [])\n\nMake sure NLS API functions will throw DimensionError if the inputs are not the correct dimension. To make this assertion in your code use\n\n@lencheck size input [more inputs separated by spaces]\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"To use this function simply call it on your model.","category":"page"},{"location":"#View-subarray-support","page":"Home","title":"View subarray support","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Two functions are given, one for NLP problems and another for NLS problems:","category":"page"},{"location":"","page":"Home","title":"Home","text":"view_subarray_nlp\nview_subarray_nls","category":"page"},{"location":"#NLPModelsTest.view_subarray_nlp","page":"Home","title":"NLPModelsTest.view_subarray_nlp","text":"view_subarray_nlp(nlp; exclude = [])\n\nCheck that the API work with views, and that the results is correct.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModelsTest.view_subarray_nls","page":"Home","title":"NLPModelsTest.view_subarray_nls","text":"view_subarray_nls(nls; exclude = [])\n\nCheck that the API work with views, and that the results is correct.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"To use this function simply call it on your model.","category":"page"},{"location":"#Coordinate-functions-memory-usage","page":"Home","title":"Coordinate functions memory usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Disclaimer: This function is incomplete.","category":"page"},{"location":"","page":"Home","title":"Home","text":"coord_memory_nlp","category":"page"},{"location":"#NLPModelsTest.coord_memory_nlp","page":"Home","title":"NLPModelsTest.coord_memory_nlp","text":"coord_memory_nlp(nlp; exclude = [])\n\nCheck that the allocated memory for in place coord methods is sufficiently smaller than their allocating counter parts.\n\n\n\n\n\n","category":"function"},{"location":"#Derivative-Checker","page":"Home","title":"Derivative Checker","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Inside the consistency check, the following functions are used to check whether the derivatives are correct. You can also use the manually.","category":"page"},{"location":"","page":"Home","title":"Home","text":"gradient_check\njacobian_check\nhessian_check_from_grad\nhessian_check","category":"page"},{"location":"#NLPModelsTest.gradient_check","page":"Home","title":"NLPModelsTest.gradient_check","text":"gradient_check(nlp; x=nlp.meta.x0, atol=1e-6, rtol=1e-4)\n\nCheck the first derivatives of the objective at x against centered finite differences.\n\nThis function returns a dictionary indexed by components of the gradient for which the relative error exceeds rtol.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModelsTest.jacobian_check","page":"Home","title":"NLPModelsTest.jacobian_check","text":"jacobian_check(nlp; x=nlp.meta.x0, atol=1e-6, rtol=1e-4)\n\nCheck the first derivatives of the constraints at x against centered finite differences.\n\nThis function returns a dictionary indexed by (j, i) tuples such that the relative error in the i-th partial derivative of the j-th constraint exceeds rtol.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModelsTest.hessian_check_from_grad","page":"Home","title":"NLPModelsTest.hessian_check_from_grad","text":"hessian_check_from_grad(nlp; x=nlp.meta.x0, atol=1e-6, rtol=1e-4, sgn=1)\n\nCheck the second derivatives of the objective and each constraints at x against centered finite differences. This check assumes exactness of the first derivatives.\n\nThe sgn arguments refers to the formulation of the Lagrangian in the problem. It should have a positive value if the Lagrangian is formulated as\n\nL(xy) = f(x) + sum_j yⱼ cⱼ(x)\n\nand a negative value if the Lagrangian is formulated as\n\nL(xy) = f(x) - sum_j yⱼ cⱼ(x)\n\nOnly the sign of sgn is important.\n\nThis function returns a dictionary indexed by functions. The 0-th function is the objective while the k-th function (for k > 0) is the k-th constraint. The values of the dictionary are dictionaries indexed by tuples (i, j) such that the relative error in the second derivative ∂²fₖ/∂xᵢ∂xⱼ exceeds rtol.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModelsTest.hessian_check","page":"Home","title":"NLPModelsTest.hessian_check","text":"hessian_check(nlp; x=nlp.meta.x0, atol=1e-6, rtol=1e-4, sgn=1)\n\nCheck the second derivatives of the objective and each constraints at x against centered finite differences. This check does not rely on exactness of the first derivatives, only on objective and constraint values.\n\nThe sgn arguments refers to the formulation of the Lagrangian in the problem. It should have a positive value if the Lagrangian is formulated as\n\nL(xy) = f(x) + sum_j yⱼ cⱼ(x)\n\nand a negative value if the Lagrangian is formulated as\n\nL(xy) = f(x) - sum_j yⱼ cⱼ(x)\n\nOnly the sign of sgn is important.\n\nThis function returns a dictionary indexed by functions. The 0-th function is the objective while the k-th function (for k > 0) is the k-th constraint. The values of the dictionary are dictionaries indexed by tuples (i, j) such that the relative error in the second derivative ∂²fₖ/∂xᵢ∂xⱼ exceeds rtol.\n\n\n\n\n\n","category":"function"},{"location":"#TL;DR","page":"Home","title":"TL;DR","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TODO after CUTEst.jl and NLPModelsJuMP are updated.","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This content is released under the MPL2.0 License.","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
