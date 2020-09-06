"""Utility functions and classes for the MindtPy solver."""
from __future__ import division
import logging
from math import fabs, floor, log
from pyomo.contrib.mindtpy.cut_generation import (add_oa_cuts,
                                                  add_nogood_cuts, add_affine_cuts)

from pyomo.core import (Any, Binary, Block, Constraint, NonNegativeReals,
                        Objective, Reals, Suffix, Var, minimize, value)
from pyomo.core.expr import differentiate
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.opt.results import ProblemSense
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
import time
import os


class MindtPySolveData(object):
    """Data container to hold solve-instance data.
    Key attributes:
        - original_model: the original model that the user gave us to solve
        - working_model: the original model after preprocessing
    """
    pass


def model_is_valid(solve_data, config):
    """
    Determines whether the model is solveable by MindtPy.

    This function returns True if the given model is solveable by MindtPy (and performs some preprocessing such
    as moving the objective to the constraints).

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: MindtPy configurations
        contains the specific configurations for the algorithm

    Returns
    -------
    Boolean value (True if model is solveable in MindtPy else False)
    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils

    # Handle LP/NLP being passed to the solver
    prob = solve_data.results.problem
    if (prob.number_of_binary_variables == 0 and
        prob.number_of_integer_variables == 0 and
            prob.number_of_disjunctions == 0):
        config.logger.info('Problem has no discrete decisions.')
        obj = next(m.component_data_objects(ctype=Objective, active=True))
        if (any(c.body.polynomial_degree() not in (1, 0) for c in MindtPy.constraint_list) or
                obj.expr.polynomial_degree() not in (1, 0)):
            config.logger.info(
                "Your model is an NLP (nonlinear program). "
                "Using NLP solver %s to solve." % config.nlp_solver)
            SolverFactory(config.nlp_solver).solve(
                solve_data.original_model, tee=config.solver_tee, **config.nlp_solver_args)
            return False
        else:
            config.logger.info(
                "Your model is an LP (linear program). "
                "Using LP solver %s to solve." % config.mip_solver)
            mipopt = SolverFactory(config.mip_solver)
            if isinstance(mipopt, PersistentSolver):
                mipopt.set_instance(solve_data.original_model)
            if config.threads > 0:
                masteropt.options["threads"] = config.threads
            mipopt.solve(solve_data.original_model,
                         tee=config.solver_tee, **config.mip_solver_args)
            return False

    if not hasattr(m, 'dual') and config.use_dual:  # Set up dual value reporting
        m.dual = Suffix(direction=Suffix.IMPORT)

    # TODO if any continuous variables are multiplied with binary ones,
    #  need to do some kind of transformation (Glover?) or throw an error message
    return True


def calc_jacobians(solve_data, config):
    """
    Generates a map of jacobians for the variables in the model

    This function generates a map of jacobians corresponding to the variables in the model and adds this
    ComponentMap to solve_data

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: MindtPy configurations
        contains the specific configurations for the algorithm
    """
    # Map nonlinear_constraint --> Map(
    #     variable --> jacobian of constraint wrt. variable)
    solve_data.jacobians = ComponentMap()
    if config.differentiate_mode == "reverse_symbolic":
        mode = differentiate.Modes.reverse_symbolic
    elif config.differentiate_mode == "sympy":
        mode = differentiate.Modes.sympy
    for c in solve_data.mip.MindtPy_utils.constraint_list:
        if c.body.polynomial_degree() in (1, 0):
            continue  # skip linear constraints
        vars_in_constr = list(EXPR.identify_variables(c.body))
        jac_list = differentiate(
            c.body, wrt_list=vars_in_constr, mode=mode)
        solve_data.jacobians[c] = ComponentMap(
            (var, jac_wrt_var)
            for var, jac_wrt_var in zip(vars_in_constr, jac_list))


def add_feas_slacks(m, config):
    """
    Adds feasibility slack variables according to config.feasibility_norm (given an infeasible problem)

    Parameters
    ----------
    m: model
        Pyomo model
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    MindtPy = m.MindtPy_utils
    # generate new constraints
    for i, constr in enumerate(MindtPy.constraint_list, 1):
        if constr.body.polynomial_degree() not in [0, 1]:
            if constr.has_ub():
                if config.feasibility_norm in {'L1', 'L2'}:
                    c = MindtPy.MindtPy_feas.feas_constraints.add(
                        constr.body - constr.upper
                        <= MindtPy.MindtPy_feas.slack_var[i])
                else:
                    c = MindtPy.MindtPy_feas.feas_constraints.add(
                        constr.body - constr.upper
                        <= MindtPy.MindtPy_feas.slack_var)
            if constr.has_lb():
                if config.feasibility_norm in {'L1', 'L2'}:
                    c = MindtPy.MindtPy_feas.feas_constraints.add(
                        constr.body - constr.lower
                        >= -MindtPy.MindtPy_feas.slack_var[i])
                else:
                    c = MindtPy.MindtPy_feas.feas_constraints.add(
                        constr.body - constr.lower
                        >= -MindtPy.MindtPy_feas.slack_var)


def var_bound_add(solve_data, config):
    """
    This function will add bounds for variables in nonlinear constraints if they are not bounded. (This is to avoid
    an unbounded master problem in the LP/NLP algorithm.) Thus, the model will be updated to include bounds for the
    unbounded variables in nonlinear constraints.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in (1, 0):
            for var in list(EXPR.identify_variables(c.body)):
                if var.has_lb() and var.has_ub():
                    continue
                elif not var.has_lb():
                    if var.is_integer():
                        var.setlb(-config.integer_var_bound - 1)
                    else:
                        var.setlb(-config.continuous_var_bound - 1)
                elif not var.has_ub():
                    if var.is_integer():
                        var.setub(config.integer_var_bound)
                    else:
                        var.setub(config.continuous_var_bound)


def add_baron_cuts(model):
    special_baron_path = "/home/canl1/work/SAAOA/baron4ieg"
    timea = time.time()
    output_filename, symbol_map, var_ids = model.baronwrite(
        "root_relaxation_baron.bar", format="bar")
    os.system("sed -i '1 a dolocal:0; ' root_relaxation_baron.bar")
    os.system("sed -i '1 a numloc:0; ' root_relaxation_baron.bar")
    os.system("sed -i '1 a maxtime:10000; ' root_relaxation_baron.bar")
    os.system("sed -i '1 a prlevel:0; ' root_relaxation_baron.bar")
    os.system("sed -i '1 a ppdo:0; ' root_relaxation_baron.bar")
    os.system("sed -i '1 a pscdo:0; ' root_relaxation_baron.bar")
    os.system('''sed -i '1 a CplexLibName: "/opt/ibm/ILOG/CPLEX_Studio129/cplex/bin/x86-64_linux/libcplex1290.so"; ' root_relaxation_baron.bar''')
    # os.system('''sed -i '1 a CplexLibName: "/Users/zedongpeng/opt/CPLEX_Studio1210/cplex/bin/x86-64_osx/libcplex12100.dylib"; ' root_relaxation_baron.bar''')

    os.system(special_baron_path + " root_relaxation_baron.bar")
    cplex_model = cplex.Cplex("relax.lp")

    timeb = time.time()
    print("lp file generation time", timeb - timea)
    # create additional variables in the pyomo model
    var_names = cplex_model.variables.get_names()
    num_bar_vars = sum("bar_var" in var for var in var_names)
    model.bar_set = RangeSet(num_bar_vars)
    model.bar_var = Var(model.bar_set)

    # create a map from cplex var id to pyomo var name
    varid_to_var = {}
    bar_var_indices = []
    for vid in var_ids:
        name = symbol_map.byObject[vid]
        var_data = symbol_map.bySymbol[name]()
        varid_cplex = cplex_model.variables.get_indices(name)
        varid_to_var[varid_cplex] = var_data

    cplex_var_names = cplex_model.variables.get_names()
    for i in range(len(cplex_var_names)):
        varname = cplex_var_names[i]
        if "bar_var" in varname:
            varid_pyomo = int(varname.split("bar_var")[1])
            varid_to_var[i] = model.bar_var[varid_pyomo]
            bar_var_indices.append(i)

    # update variable bounds in pyomo
    var_lb = cplex_model.variables.get_lower_bounds()
    var_ub = cplex_model.variables.get_upper_bounds()
    for i in range(len(var_lb)):
        if i in varid_to_var:
            var = varid_to_var[i]
            var.setlb(var_lb[i])
            var.setub(var_ub[i])
    # #To create a list that contain information of the linear constraints in the original pyomo model
    # for c in model.component_objects(Constraint):

    # add constraints that have bar_var
    model.baroncuts = ConstraintList()
    nconstraints = cplex_model.linear_constraints.get_num()
    for c in range(nconstraints):
        row = cplex_model.linear_constraints.get_rows(c)
        rhs = cplex_model.linear_constraints.get_rhs(c)
        sense = cplex_model.linear_constraints.get_senses(c)
        if sum(varid in bar_var_indices for varid in row.ind) > 0:
            expr = sum(row.val[i] * varid_to_var[row.ind[i]]
                       for i in range(len(row.ind)))
            if sense == 'G':
                model.baroncuts.add(expr >= rhs)
            if sense == 'L':
                model.baroncuts.add(expr <= rhs)
            if sense == 'E':
                model.baroncuts.add(expr == rhs)
    # change objective
    model.obj.deactivate()
    coeff = cplex_model.objective.get_linear()
    if cplex_model.objective.get_sense() == 1:
        model.baron_obj = Objective(expr=sum(varid_to_var[i] * coeff[i] for i in range(
            cplex_model.variables.get_num()) if i in varid_to_var.keys()), sense=minimize)
    else:
        model.baron_obj = Objective(expr=sum(varid_to_var[i] * coeff[i] for i in range(
            cplex_model.variables.get_num()) if i in varid_to_var.keys()), sense=maximize)

    timec = time.time()
    print("time to add the cuts to pyomo model", timec-timeb)
