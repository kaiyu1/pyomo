#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Iteration loop for MindtPy."""
from __future__ import division
from pyomo.contrib.mindtpy.util import set_solver_options, get_integer_solution, copy_var_list_values_from_solution_pool, add_feas_slacks, add_var_bound, epigraph_reformulation
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts, add_oa_cuts, add_affine_cuts, add_no_good_cuts
from pyomo.core import minimize, maximize, Objective, VarList, Reals, ConstraintList, Constraint, Block, TransformationFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt import SolverFactory
from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from operator import itemgetter
from pyomo.common.collections import Bunch
from io import StringIO
from pyomo.contrib.gdpopt.util import (copy_var_list_values, 
                                       time_code, lower_logger_level_to)
import math
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory, SolverResults, ProblemSense
from pyomo.repn import generate_standard_repn
import logging
from pyomo.util.model_size import build_model_size_report
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core import (Block, Constraint, VarList, NonNegativeReals,
                        Objective, Reals, Suffix, Var, minimize, RangeSet, ConstraintList, TransformationFactory)
from pyomo.core import (ConstraintList, Objective,
                        TransformationFactory, maximize, minimize,
                        value, Var)
from math import copysign
from pyomo.contrib.gdpopt.util import (SuppressInfeasibleWarning, _DoNothing,
                                       copy_var_list_values, get_main_elapsed_time)
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.opt import SolutionStatus, SolverStatus
from pyomo.contrib.gdpopt.solve_discrete_problem import distinguish_mip_infeasible_or_unbounded
from pyomo.contrib.mindtpy.util import generate_norm1_objective_function, generate_norm2sq_objective_function, generate_norm_inf_objective_function, generate_lag_objective_function, set_solver_options, GurobiPersistent4MindtPy
from pyomo.contrib.mindtpy.util import calc_jacobians, MindtPySolveData
from pyomo.core import Constraint, Expression, Objective, minimize, value
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.contrib.mindtpy.util import add_baron_cuts

single_tree, single_tree_available = attempt_import(
    'pyomo.contrib.mindtpy.single_tree')
tabu_list, tabu_list_available = attempt_import(
    'pyomo.contrib.mindtpy.tabu_list')


tabu_list, tabu_list_available = attempt_import(
    'pyomo.contrib.mindtpy.tabu_list')

__version__ = (0, 1, 0)

class _MindtPyAlgorithm(object):
    
    def __init__(self, **kwds):
        """
        This is a common init method for all the MindtPy algorithms, so that we
        correctly set up the config arguments and initialize the generic parts
        of the algorithm state.

        """
        self.working_model = None
        self.mip = None
        self.fixed_nlp = None

        # We store bounds, timing info, iteration count, incumbent, and the
        # expression of the original (possibly nonlinear) objective function.
        self.results = SolverResults()
        self.timing = Bunch()
        self.curr_int_sol = []
        self.should_terminate = False
        self.integer_list = []

        # Set up iteration counters
        self.nlp_iter = 0
        self.mip_iter = 0
        self.mip_subiter = 0
        self.nlp_infeasible_counter = 0
        self.fp_iter = 1

        self.primal_bound_progress_time = [0]
        self.dual_bound_progress_time = [0]
        self.abs_gap = float('inf')
        self.rel_gap = float('inf')
        self.log_formatter = ' {:>9}   {:>15}   {:>15g}   {:>12g}   {:>12g}   {:>7.2%}   {:>7.2f}'
        self.fixed_nlp_log_formatter = '{:1}{:>9}   {:>15}   {:>15g}   {:>12g}   {:>12g}   {:>7.2%}   {:>7.2f}'
        self.log_note_formatter = ' {:>9}   {:>15}   {:>15}'

        # Flag indicating whether the solution improved in the past
        # iteration or not
        self.primal_bound_improved = False
        self.dual_bound_improved = False

        # Store the initial model state as the best solution found. If we
        # find no better solution, then we will restore from this copy.
        self.best_solution_found = None
        self.best_solution_found_time = None

        self.stored_bound = {}
        self.num_no_good_cuts_added = {}

    # Support use as a context manager under current solver API
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Solver is always available. Though subsolvers may not be, they will
        raise an error when the time comes.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    _metasolver = False

    def _log_solver_intro_message(self, config):
        config.logger.info(
            "Starting MindtPy version %s using %s algorithm"
            % (".".join(map(str, self.version())), self.algorithm)
        )
        os = StringIO()
        config.display(ostream=os)
        config.logger.info(os.getvalue())
        config.logger.info(
                '---------------------------------------------------------------------------------------------\n'
                '              Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy)               \n'
                '---------------------------------------------------------------------------------------------\n'
                'For more information, please visit \n'
                'https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html')
        config.logger.info("""
        If you use this software, you may cite the following:
        - Implementation:
        Bernal, David E., et al. "Mixed-integer nonlinear decomposition toolbox for Pyomo (MindtPy)." 
        Computer Aided Chemical Engineering. Vol. 44. Elsevier, 2018. 895-900.
        """.strip())
    
    def set_up_logger(self):
        """Set up the formatter and handler for logger.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        """
        self.config.logger.handlers.clear()
        self.config.logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(self.config.logging_level)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        # add the handlers to logger
        self.config.logger.addHandler(ch)

    def _log_header(self, logger):
        # TODO: rewrite
        logger.info(
            '================================================================='
            '============================')
        logger.info(
            '{:^9} | {:^15} | {:^11} | {:^11} | {:^8} | {:^7}\n'.format(
                'Iteration', 'Subproblem Type', 'Lower Bound', 'Upper Bound',
                ' Gap ', 'Time(s)'))

    def create_utility_block(self, model, name):
        created_util_block = False
        # Create a model block on which to store GDPopt-specific utility
        # modeling objects.
        if hasattr(model, name):
            raise RuntimeError(
                "MindtPy needs to create a Block named %s "
                "on the model object, but an attribute with that name "
                "already exists." % name)
        else:
            created_util_block = True
            setattr(model, name, Block(
                doc="Container for MindtPy solver utility modeling objects"))
            self.util_block_name = name

            # Save ordered lists of main modeling components, so that data can
            # be easily transferred between future model clones.
            self.build_ordered_component_lists(model)
            self.add_cuts_components(model)
        # TODO yeild
        # if created_util_block:
        #     model.del_component(name)


    def model_is_valid(self):
        """Determines whether the model is solvable by MindtPy.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.

        Returns
        -------
        bool
            True if model is solvable in MindtPy, False otherwise.
        """
        m = self.working_model
        MindtPy = m.MindtPy_utils
        config = self.config

        # Handle LP/NLP being passed to the solver
        prob = self.results.problem
        if len(MindtPy.discrete_variable_list) == 0:
            config.logger.info('Problem has no discrete decisions.')
            obj = next(m.component_data_objects(ctype=Objective, active=True))
            if (any(c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree for c in MindtPy.constraint_list) or
                    obj.expr.polynomial_degree() not in self.mip_objective_polynomial_degree):
                config.logger.info(
                    'Your model is a NLP (nonlinear program). '
                    'Using NLP solver %s to solve.' % config.nlp_solver)
                nlpopt = SolverFactory(config.nlp_solver)
                # TODO: rewrite
                set_solver_options(nlpopt, self.timing, config, solver_type='nlp')
                nlpopt.solve(self.original_model,
                            tee=config.nlp_solver_tee, **config.nlp_solver_args)
                return False
            else:
                config.logger.info(
                    'Your model is an LP (linear program). '
                    'Using LP solver %s to solve.' % config.mip_solver)
                mainopt = SolverFactory(config.mip_solver)
                if isinstance(mainopt, PersistentSolver):
                    mainopt.set_instance(self.original_model)
                set_solver_options(mainopt, self.timing, config, solver_type='mip')
                results = mainopt.solve(self.original_model,
                                        tee=config.mip_solver_tee,
                                        load_solutions=False,
                                        **config.mip_solver_args
                                        )
                if len(results.solution) > 0:
                    self.original_model.solutions.load_from(results)
                return False

        if not hasattr(m, 'dual') and config.calculate_dual_at_solution:  # Set up dual value reporting
            m.dual = Suffix(direction=Suffix.IMPORT)

        # TODO if any continuous variables are multiplied with binary ones,
        #  need to do some kind of transformation (Glover?) or throw an error message
        return True


    def build_ordered_component_lists(self, model):
        """Define lists used for future data transfer.

        Also attaches ordered lists of the variables, constraints, disjuncts, and
        disjunctions to the model so that they can be used for mapping back and
        forth.

        """
        util_blk = getattr(model, self.util_block_name)
        var_set = ComponentSet()
        setattr(
            util_blk, 'constraint_list', list(
                model.component_data_objects(
                    ctype=Constraint, active=True,
                    descend_into=(Block, Disjunct))))
        # if hasattr(solve_data,'mip_constraint_polynomial_degree'):
        mip_constraint_polynomial_degree = self.mip_constraint_polynomial_degree
        # else:
        #     mip_constraint_polynomial_degree = {0, 1}
        setattr(
            util_blk, 'linear_constraint_list', list(
                c for c in model.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
                if c.body.polynomial_degree() in mip_constraint_polynomial_degree))
        setattr(
            util_blk, 'nonlinear_constraint_list', list(
                c for c in model.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
                if c.body.polynomial_degree() not in mip_constraint_polynomial_degree))
        setattr(
            util_blk, 'disjunct_list', list(
                model.component_data_objects(
                    ctype=Disjunct, active=True,
                    descend_into=(Block, Disjunct))))
        setattr(
            util_blk, 'disjunction_list', list(
                model.component_data_objects(
                    ctype=Disjunction, active=True,
                    descend_into=(Disjunct, Block))))
        setattr(
            util_blk, 'objective_list', list(
                model.component_data_objects(
                    ctype=Objective, active=True,
                    descend_into=(Block))))

        # Identify the non-fixed variables in (potentially) active constraints and
        # objective functions
        for constr in getattr(util_blk, 'constraint_list'):
            for v in EXPR.identify_variables(constr.body, include_fixed=False):
                var_set.add(v)
        for obj in model.component_data_objects(ctype=Objective, active=True):
            for v in EXPR.identify_variables(obj.expr, include_fixed=False):
                var_set.add(v)
        # Disjunct indicator variables might not appear in active constraints. In
        # fact, if we consider them Logical variables, they should not appear in
        # active algebraic constraints. For now, they need to be added to the
        # variable set.
        for disj in getattr(util_blk, 'disjunct_list'):
            var_set.add(disj.binary_indicator_var)

        # We use component_data_objects rather than list(var_set) in order to
        # preserve a deterministic ordering.
        var_list = list(
            v for v in model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v in var_set)
        setattr(util_blk, 'variable_list', var_list)
        discrete_variable_list = list(
            v for v in model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v in var_set and v.is_integer())
        setattr(util_blk, 'discrete_variable_list', discrete_variable_list)
        continuous_variable_list = list(
            v for v in model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v in var_set and v.is_continuous())
        setattr(util_blk, 'continuous_variable_list', continuous_variable_list)

    def add_cuts_components(self, model):
        config = self.config
        MindtPy = model.MindtPy_utils

        # Create a model block in which to store the generated feasibility
        # slack constraints. Do not leave the constraints on by default.
        feas = MindtPy.feas_opt = Block()
        feas.deactivate()
        feas.feas_constraints = ConstraintList(
            doc='Feasibility Problem Constraints')

        # Create a model block in which to store the generated linear
        # constraints. Do not leave the constraints on by default.
        lin = MindtPy.cuts = Block()
        lin.deactivate()

        # no-good cuts exclude particular discrete decisions
        lin.no_good_cuts = ConstraintList(doc='no-good cuts')
        # Feasible no-good cuts exclude discrete realizations that have
        # been explored via an NLP subproblem. Depending on model
        # characteristics, the user may wish to revisit NLP subproblems
        # (with a different initialization, for example). Therefore, these
        # cuts are not enabled by default.
        #
        # Note: these cuts will only exclude integer realizations that are
        # not already in the primary no_good_cuts ConstraintList.
        # TODO: this is not used.
        lin.feasible_no_good_cuts = ConstraintList(
            doc='explored no-good cuts')
        lin.feasible_no_good_cuts.deactivate()

        if config.feasibility_norm == 'L1' or config.feasibility_norm == 'L2':
            feas.nl_constraint_set = RangeSet(len(MindtPy.nonlinear_constraint_list),
                                                doc='Integer index set over the nonlinear constraints.')
            # Create slack variables for feasibility problem
            feas.slack_var = Var(feas.nl_constraint_set,
                                    domain=NonNegativeReals, initialize=1)
        else:
            feas.slack_var = Var(domain=NonNegativeReals, initialize=1)

        # Create slack variables for OA cuts
        if config.add_slack:
            lin.slack_vars = VarList(
                bounds=(0, config.max_slack), initialize=0, domain=NonNegativeReals)

    def get_dual_integral(self):
        """Calculate the dual integral.
        Ref: The confined primal integral. [http://www.optimization-online.org/DB_FILE/2020/07/7910.pdf]

        Returns
        -------
        float
            The dual integral.
        """    
        dual_integral = 0
        dual_bound_progress = self.dual_bound_progress.copy()
        # Initial dual bound is set to inf or -inf. To calculate dual integral, we set
        # initial_dual_bound to 10% greater or smaller than the first_found_dual_bound.
        # TODO: check if the calculation of initial_dual_bound needs to be modified.
        for dual_bound in dual_bound_progress:
            if dual_bound != dual_bound_progress[0]:
                break
        for i in range(len(dual_bound_progress)):
            if dual_bound_progress[i] == self.dual_bound_progress[0]:
                dual_bound_progress[i] = dual_bound * (1 - self.config.initial_bound_coef * self.objective_sense * math.copysign(1,dual_bound))
            else:
                break
        for i in range(len(dual_bound_progress)):
            if i == 0:
                dual_integral += abs(dual_bound_progress[i] - self.dual_bound) * (self.dual_bound_progress_time[i])
            else:
                dual_integral += abs(dual_bound_progress[i] - self.dual_bound) * (self.dual_bound_progress_time[i] - self.dual_bound_progress_time[i-1])
        self.config.logger.info(' {:<25}:   {:>7.4f} '.format('Dual integral', dual_integral))
        return dual_integral

    def get_primal_integral(self):
        """Calculate the primal integral.
        Ref: The confined primal integral. [http://www.optimization-online.org/DB_FILE/2020/07/7910.pdf]

        Parameters
        ----------
        self : MindtPySolveData
            Data container that holds solve-instance data.

        Returns
        -------
        float
            The primal integral.
        """    
        primal_integral = 0
        primal_bound_progress = self.primal_bound_progress.copy()
        # Initial primal bound is set to inf or -inf. To calculate primal integral, we set
        # initial_primal_bound to 10% greater or smaller than the first_found_primal_bound.
        # TODO: check if the calculation of initial_primal_bound needs to be modified.
        for primal_bound in primal_bound_progress:
            if primal_bound != primal_bound_progress[0]:
                break
        for i in range(len(primal_bound_progress)):
            if primal_bound_progress[i] == self.primal_bound_progress[0]:
                primal_bound_progress[i] = primal_bound * (1 + self.config.initial_bound_coef * self.objective_sense * math.copysign(1,primal_bound))
            else:
                break
        for i in range(len(primal_bound_progress)):
            if i == 0:
                primal_integral += abs(primal_bound_progress[i] - self.primal_bound) * (self.primal_bound_progress_time[i])
            else:
                primal_integral += abs(primal_bound_progress[i] - self.primal_bound) * (self.primal_bound_progress_time[i] - self.primal_bound_progress_time[i-1])

        self.config.logger.info(' {:<25}:   {:>7.4f} '.format('Primal integral', primal_integral))
        return primal_integral


    def update_gap(self):
        """Update the relative gap and the absolute gap.

        """
        if self.objective_sense == minimize:
            self.abs_gap = self.primal_bound - self.dual_bound
        else:
            self.abs_gap = self.dual_bound - self.primal_bound
        self.rel_gap = self.abs_gap / (abs(self.primal_bound) + 1E-10)


    def update_dual_bound(self, bound_value):
        """Update the dual bound.

        Call after solving relaxed problem, including relaxed NLP and MIP master problem.
        Use the optimal primal bound of the relaxed problem to update the dual bound.

        Parameters
        ----------
        bound_value : float
            The input value used to update the dual bound.
        """
        if math.isnan(bound_value):
            return
        if self.objective_sense == minimize:
            self.dual_bound = max(bound_value, self.dual_bound)
            self.dual_bound_improved = self.dual_bound > self.dual_bound_progress[-1]
        else:
            self.dual_bound = min(bound_value, self.dual_bound)
            self.dual_bound_improved = self.dual_bound < self.dual_bound_progress[-1]
        self.dual_bound_progress.append(self.dual_bound)
        self.dual_bound_progress_time.append(get_main_elapsed_time(self.timing))
        if self.dual_bound_improved:
            self.update_gap()


    def update_suboptimal_dual_bound(self, results):
        """If the relaxed problem is not solved to optimality, the dual bound is updated 
        according to the dual bound of relaxed problem.

        Parameters
        ----------
        results : SolverResults
            Results from solving the relaxed problem.
            The dual bound of the relaxed problem can only be obtained from the result object.
        """
        if self.objective_sense == minimize:
            bound_value = results.problem.lower_bound
        else:
            bound_value = results.problem.upper_bound
        self.update_dual_bound(bound_value)

    def update_primal_bound(self, bound_value):
        """Update the primal bound.

        Call after solve fixed NLP subproblem.
        Use the optimal primal bound of the relaxed problem to update the dual bound.

        Parameters
        ----------
        bound_value : float
            The input value used to update the primal bound.
        """
        if math.isnan(bound_value):
            return
        if self.objective_sense == minimize:
            self.primal_bound = min(bound_value, self.primal_bound)
            self.primal_bound_improved = self.primal_bound < self.primal_bound_progress[-1]
        else:
            self.primal_bound = max(bound_value, self.primal_bound)
            self.primal_bound_improved = self.primal_bound > self.primal_bound_progress[-1]
        self.primal_bound_progress.append(self.primal_bound)
        self.primal_bound_progress_time.append(get_main_elapsed_time(self.timing))
        if self.primal_bound_improved:
            self.update_gap()


    def process_objective(self, config, move_objective=False,
                        use_mcpp=False, update_var_con_list=True,
                        partition_nonlinear_terms=True,
                        obj_handleable_polynomial_degree={0, 1},
                        constr_handleable_polynomial_degree={0, 1}):
        """Process model objective function.
        Check that the model has only 1 valid objective.
        If the objective is nonlinear, move it into the constraints.
        If no objective function exists, emit a warning and create a dummy 
        objective.
        Parameters
        ----------
        config (ConfigBlock): solver configuration options
        move_objective (bool): if True, move even linear
            objective functions to the constraints
        update_var_con_list (bool): if True, the variable/constraint/objective lists will not be updated. 
            This arg is set to True by default. Currently, update_var_con_list will be set to False only when
            add_regularization is not None in MindtPy.
        partition_nonlinear_terms (bool): if True, partition sum of nonlinear terms in the objective function.
        """
        m = self.working_model
        util_blk = getattr(m, self.util_block_name)
        # Handle missing or multiple objectives
        active_objectives = list(m.component_data_objects(
            ctype=Objective, active=True, descend_into=True))
        self.results.problem.number_of_objectives = len(active_objectives)
        if len(active_objectives) == 0:
            config.logger.warning(
                'Model has no active objectives. Adding dummy objective.')
            util_blk.dummy_objective = Objective(expr=1)
            main_obj = util_blk.dummy_objective
        elif len(active_objectives) > 1:
            raise ValueError('Model has multiple active objectives.')
        else:
            main_obj = active_objectives[0]
        self.results.problem.sense = ProblemSense.minimize if \
                                        main_obj.sense == 1 else \
                                        ProblemSense.maximize
        self.objective_sense = main_obj.sense

        # Move the objective to the constraints if it is nonlinear or move_objective is True.
        if main_obj.expr.polynomial_degree() not in obj_handleable_polynomial_degree or move_objective:
            if move_objective:
                config.logger.info("Moving objective to constraint set.")
            else:
                config.logger.info(
                    "Objective is nonlinear. Moving it to constraint set.")
            util_blk.objective_value = VarList(domain=Reals, initialize=0)
            util_blk.objective_constr = ConstraintList()
            if main_obj.expr.polynomial_degree() not in obj_handleable_polynomial_degree and partition_nonlinear_terms and main_obj.expr.__class__ is EXPR.SumExpression:
                repn = generate_standard_repn(main_obj.expr, quadratic=2 in obj_handleable_polynomial_degree)
                # the following code will also work if linear_subexpr is a constant.
                linear_subexpr = repn.constant + sum(coef*var for coef, var in zip(repn.linear_coefs, repn.linear_vars)) \
                    + sum(coef*var1*var2 for coef, (var1, var2) in zip(repn.quadratic_coefs, repn.quadratic_vars))
                # only need to generate one epigraph constraint for the sum of all linear terms and constant
                epigraph_reformulation(linear_subexpr, util_blk.objective_value, util_blk.objective_constr, use_mcpp, main_obj.sense)
                nonlinear_subexpr = repn.nonlinear_expr
                if nonlinear_subexpr.__class__ is EXPR.SumExpression:
                    for subsubexpr in nonlinear_subexpr.args:
                        epigraph_reformulation(subsubexpr, util_blk.objective_value, util_blk.objective_constr, use_mcpp, main_obj.sense)
                else:
                    epigraph_reformulation(nonlinear_subexpr, util_blk.objective_value, util_blk.objective_constr, use_mcpp, main_obj.sense)
            else:
                epigraph_reformulation(main_obj.expr, util_blk.objective_value, util_blk.objective_constr, use_mcpp, main_obj.sense)

            main_obj.deactivate()
            util_blk.objective = Objective(expr=sum(util_blk.objective_value[:]), sense=main_obj.sense)

            if main_obj.expr.polynomial_degree() not in obj_handleable_polynomial_degree or \
            (move_objective and update_var_con_list):
                util_blk.variable_list.extend(util_blk.objective_value[:])
                util_blk.continuous_variable_list.extend(util_blk.objective_value[:])
                util_blk.constraint_list.extend(util_blk.objective_constr[:])
                util_blk.objective_list.append(util_blk.objective)
                for constr in util_blk.objective_constr[:]:
                    if constr.body.polynomial_degree() in constr_handleable_polynomial_degree:
                        util_blk.linear_constraint_list.append(constr)
                    else:
                        util_blk.nonlinear_constraint_list.append(constr)


    def set_up_solve_data(self, model, config):
        """Set up the solve data.

        Parameters
        ----------
        model : Pyomo model
            The original model to be solved in MindtPy.
        config : ConfigBlock
            The specific configurations for MindtPy.

        """
        # if the objective function is a constant, dual bound constraint is not added.
        obj = next(model.component_data_objects(ctype=Objective, active=True))
        if obj.expr.polynomial_degree() == 0:
            config.use_dual_bound = False

        self.initial_model = model

        if config.use_fbbt:
            fbbt(model)
            # TODO: logging_level is not logging.INFO here
            config.logger.info(
                'Use the fbbt to tighten the bounds of variables')
        
        if config.use_baron_convexification:
            add_baron_cuts(model)
            # TODO: logging_level is not logging.INFO here
            config.logger.info(
                'Use the baron to tighten the bounds of variables')


        self.original_model = model
        self.working_model = model.clone()

        # set up bounds
        if obj.sense == minimize:
            self.primal_bound = float('inf')
            self.dual_bound = float('-inf')
        else:
            self.primal_bound = float('-inf')
            self.dual_bound = float('inf')
        self.primal_bound_progress = [self.primal_bound]
        self.dual_bound_progress = [self.dual_bound]

        if config.nlp_solver == 'ipopt':
            if not hasattr(self.working_model, 'ipopt_zL_out'):
                self.working_model.ipopt_zL_out = Suffix(
                    direction=Suffix.IMPORT)
            if not hasattr(self.working_model, 'ipopt_zU_out'):
                self.working_model.ipopt_zU_out = Suffix(
                    direction=Suffix.IMPORT)

        if config.quadratic_strategy == 0:
            self.mip_objective_polynomial_degree = {0, 1}
            self.mip_constraint_polynomial_degree = {0, 1}
        elif config.quadratic_strategy == 1:
            self.mip_objective_polynomial_degree = {0, 1, 2}
            self.mip_constraint_polynomial_degree = {0, 1}
        elif config.quadratic_strategy == 2:
            self.mip_objective_polynomial_degree = {0, 1, 2}
            self.mip_constraint_polynomial_degree = {0, 1, 2}


    # -----------------------------------------------------------------------------------------
    # initialization

    def MindtPy_initialization(self, config):
        """Initializes the decomposition algorithm.

        This function initializes the decomposition algorithm, which includes generating the
        initial cuts required to build the main MIP.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        """
        # Do the initialization
        if config.init_strategy == 'rNLP':
            self.init_rNLP(config)
        elif config.init_strategy == 'max_binary':
            self.init_max_binaries(config)
        elif config.init_strategy == 'initial_binary':
            self.curr_int_sol = get_integer_solution(
                self.working_model)
            self.integer_list.append(self.curr_int_sol)
            fixed_nlp, fixed_nlp_result = self.solve_subproblem(config)
            self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result, config)
        elif config.init_strategy == 'FP':
            self.init_rNLP(config)
            self.fp_loop(config)


    def init_rNLP(self, config):
        """Initialize the problem by solving the relaxed NLP and then store the optimal variable
        values obtained from solving the rNLP.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the relaxed NLP.
        """
        m = self.working_model.clone()
        config.logger.debug(
            'Relaxed NLP: Solve relaxed integrality')
        MindtPy = m.MindtPy_utils
        TransformationFactory('core.relax_integer_vars').apply_to(m)
        nlp_args = dict(config.nlp_solver_args)
        nlpopt = SolverFactory(config.nlp_solver)
        set_solver_options(nlpopt, self.timing, config, solver_type='nlp')
        with SuppressInfeasibleWarning():
            results = nlpopt.solve(m,
                                tee=config.nlp_solver_tee, 
                                load_solutions=False,
                                **nlp_args)
            if len(results.solution) > 0:
                m.solutions.load_from(results)
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond in {tc.optimal, tc.feasible, tc.locallyOptimal}:
            main_objective = MindtPy.objective_list[-1]
            if subprob_terminate_cond == tc.optimal:
                self.update_dual_bound(value(main_objective.expr))
            else:
                config.logger.info(
                    'relaxed NLP is not solved to optimality.')
                self.update_suboptimal_dual_bound(results)
            dual_values = list(
                m.dual[c] for c in MindtPy.constraint_list) if config.calculate_dual_at_solution else None
            config.logger.info(self.log_formatter.format('-', 'Relaxed NLP', value(main_objective.expr),
                                                            self.primal_bound, self.dual_bound, self.rel_gap,
                                                            get_main_elapsed_time(self.timing)))
            # Add OA cut
            if config.strategy in {'OA', 'GOA', 'FP'}:
                copy_var_list_values(m.MindtPy_utils.variable_list,
                                    self.mip.MindtPy_utils.variable_list,
                                    config)
                if config.init_strategy == 'FP':
                    copy_var_list_values(m.MindtPy_utils.variable_list,
                                        self.working_model.MindtPy_utils.variable_list,
                                        config)
                if config.strategy in {'OA', 'FP'}:
                    add_oa_cuts(self.mip, 
                    dual_values,
                    self.jacobians,
                    self.objective_sense,
                    self.mip_constraint_polynomial_degree,
                    self.mip_iter,
                    config,
                    self.timing)
                elif config.strategy == 'GOA':
                    add_affine_cuts(self.mip, config, self.timing)
                for var in self.mip.MindtPy_utils.discrete_variable_list:
                    # We don't want to trigger the reset of the global stale
                    # indicator, so we will set this variable to be "stale",
                    # knowing that set_value will switch it back to "not
                    # stale"
                    var.stale = True
                    var.set_value(int(round(var.value)), skip_validation=True)
        elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
            # TODO fail? try something else?
            config.logger.info(
                'Initial relaxed NLP problem is infeasible. '
                'Problem may be infeasible.')
        elif subprob_terminate_cond is tc.maxTimeLimit:
            config.logger.info(
                'NLP subproblem failed to converge within time limit.')
            self.results.solver.termination_condition = tc.maxTimeLimit
        elif subprob_terminate_cond is tc.maxIterations:
            config.logger.info(
                'NLP subproblem failed to converge within iteration limit.')
        else:
            raise ValueError(
                'MindtPy unable to handle relaxed NLP termination condition '
                'of %s. Solver message: %s' %
                (subprob_terminate_cond, results.solver.message))


    def init_max_binaries(self, config):
        """Modifies model by maximizing the number of activated binary variables.

        Note - The user would usually want to call solve_subproblem after an invocation 
        of this function.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MILP main problem is infeasible.
        ValueError
            MindtPy unable to handle the termination condition of the MILP main problem.
        """
        m = self.working_model.clone()
        if config.calculate_dual_at_solution:
            m.dual.deactivate()
        MindtPy = m.MindtPy_utils
        self.mip_subiter += 1
        config.logger.debug(
            'Initialization: maximize value of binaries')
        for c in MindtPy.nonlinear_constraint_list:
            c.deactivate()
        objective = next(m.component_data_objects(Objective, active=True))
        objective.deactivate()
        binary_vars = (v for v in m.MindtPy_utils.discrete_variable_list
                    if v.is_binary() and not v.fixed)
        MindtPy.max_binary_obj = Objective(
            expr=sum(v for v in binary_vars), sense=maximize)

        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        mipopt = SolverFactory(config.mip_solver)
        if isinstance(mipopt, PersistentSolver):
            mipopt.set_instance(m)
        mip_args = dict(config.mip_solver_args)
        set_solver_options(mipopt, self.timing, config, solver_type='mip')
        results = mipopt.solve(m, 
                            tee=config.mip_solver_tee, 
                            load_solutions=False,
                            **mip_args)
        if len(results.solution) > 0:
            m.solutions.load_from(results)

        solve_terminate_cond = results.solver.termination_condition
        if solve_terminate_cond is tc.optimal:
            copy_var_list_values(
                MindtPy.variable_list,
                self.working_model.MindtPy_utils.variable_list,
                config)
            config.logger.info(self.log_formatter.format('-',
                                                            'Max binary MILP', 
                                                            value(MindtPy.max_binary_obj.expr),
                                                            self.primal_bound,
                                                            self.dual_bound,
                                                            self.rel_gap,
                                                            get_main_elapsed_time(self.timing)))
        elif solve_terminate_cond is tc.infeasible:
            raise ValueError(
                'MILP main problem is infeasible. '
                'Problem may have no more feasible '
                'binary configurations.')
        elif solve_terminate_cond is tc.maxTimeLimit:
            config.logger.info(
                'NLP subproblem failed to converge within time limit.')
            self.results.solver.termination_condition = tc.maxTimeLimit
        elif solve_terminate_cond is tc.maxIterations:
            config.logger.info(
                'NLP subproblem failed to converge within iteration limit.')
        else:
            raise ValueError(
                'MindtPy unable to handle MILP main termination condition '
                'of %s. Solver message: %s' %
                (solve_terminate_cond, results.solver.message))

    ##################################################################################################################################################################################################################
    # nlp_solve.py

    def solve_subproblem(self, config):
        # TODO: we will keep working_model.clone first
        # This function is algorithm-dependent, therefore we will redefine it as method.
        """Solves the Fixed-NLP (with fixed integers).

        This function sets up the 'fixed_nlp' by fixing binaries, sets continuous variables to their intial var values,
        precomputes dual values, deactivates trivial constraints, and then solves NLP model.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.

        Returns
        -------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        results : SolverResults
            Results from solving the Fixed-NLP.
        """
        fixed_nlp = self.working_model.clone()
        MindtPy = fixed_nlp.MindtPy_utils
        self.nlp_iter += 1

        # Set up NLP
        TransformationFactory('core.fix_integer_vars').apply_to(fixed_nlp)

        MindtPy.cuts.deactivate()
        if config.calculate_dual_at_solution:
            fixed_nlp.tmp_duals = ComponentMap()
            # tmp_duals are the value of the dual variables stored before using deactivate trivial contraints
            # The values of the duals are computed as follows: (Complementary Slackness)
            #
            # | constraint | c_geq | status at x1 | tmp_dual (violation) |
            # |------------|-------|--------------|----------------------|
            # | g(x) <= b  | -1    | g(x1) <= b   | 0                    |
            # | g(x) <= b  | -1    | g(x1) > b    | g(x1) - b            |
            # | g(x) >= b  | +1    | g(x1) >= b   | 0                    |
            # | g(x) >= b  | +1    | g(x1) < b    | b - g(x1)            |
            evaluation_error = False
            for c in fixed_nlp.MindtPy_utils.constraint_list:
                # We prefer to include the upper bound as the right hand side since we are
                # considering c by default a (hopefully) convex function, which would make
                # c >= lb a nonconvex inequality which we wouldn't like to add linearizations
                # if we don't have to
                rhs = value(c.upper) if c.has_ub() else value(c.lower)
                c_geq = -1 if c.has_ub() else 1
                try:
                    fixed_nlp.tmp_duals[c] = c_geq * max(
                        0, c_geq*(rhs - value(c.body)))
                except (ValueError, OverflowError) as error:
                    fixed_nlp.tmp_duals[c] = None
                    evaluation_error = True
            if evaluation_error:
                for nlp_var, orig_val in zip(
                        MindtPy.variable_list,
                        self.initial_var_values):
                    if not nlp_var.fixed and not nlp_var.is_binary():
                        nlp_var.set_value(orig_val, skip_validation=True)
        try:
            TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
                fixed_nlp, tmp=True, ignore_infeasible=False, tolerance=config.constraint_tolerance)
        except InfeasibleConstraintException:
            config.logger.warning(
                'infeasibility detected in deactivate_trivial_constraints')
            results = SolverResults()
            results.solver.termination_condition = tc.infeasible
            return fixed_nlp, results
        # Solve the NLP
        nlpopt = SolverFactory(config.nlp_solver)
        nlp_args = dict(config.nlp_solver_args)
        # TODO: Can we move set_solver_options outside of this function?
        # if not, we can define this function as a method
        set_solver_options(nlpopt, self.timing, config, solver_type='nlp')
        with SuppressInfeasibleWarning():
            with time_code(self.timing, 'fixed subproblem'):
                results = nlpopt.solve(fixed_nlp,
                                    tee=config.nlp_solver_tee,
                                    load_solutions=False,
                                    **nlp_args)
                if len(results.solution) > 0:
                    fixed_nlp.solutions.load_from(results)
        return fixed_nlp, results


    def handle_nlp_subproblem_tc(self, fixed_nlp, result, config, cb_opt=None):
        """This function handles different terminaton conditions of the fixed-NLP subproblem.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        result : SolverResults
            Results from solving the NLP subproblem.
        config : ConfigBlock
            The specific configurations for MindtPy.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.
        """
        if result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            self.handle_subproblem_optimal(fixed_nlp, config, cb_opt)
        elif result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
            self.handle_subproblem_infeasible(fixed_nlp, config, cb_opt)
        elif result.solver.termination_condition is tc.maxTimeLimit:
            config.logger.info(
                'NLP subproblem failed to converge within the time limit.')
            self.results.solver.termination_condition = tc.maxTimeLimit
            self.should_terminate = True
        elif result.solver.termination_condition is tc.maxEvaluations:
            config.logger.info(
                'NLP subproblem failed due to maxEvaluations.')
            self.results.solver.termination_condition = tc.maxEvaluations
            self.should_terminate = True
        else:
            self.handle_subproblem_other_termination(fixed_nlp, result.solver.termination_condition, config)


    def handle_subproblem_optimal(self, fixed_nlp, config, cb_opt=None, fp=False):
        """This function copies the result of the NLP solver function ('solve_subproblem') to the working model, updates
        the bounds, adds OA and no-good cuts, and then stores the new solution if it is the new best solution. This
        function handles the result of the latest iteration of solving the NLP subproblem given an optimal solution.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        config : ConfigBlock
            The specific configurations for MindtPy.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.
        fp : bool, optional
            Whether it is in the loop of feasibility pump, by default False.
        """
        copy_var_list_values(
            fixed_nlp.MindtPy_utils.variable_list,
            self.working_model.MindtPy_utils.variable_list,
            config)
        if config.calculate_dual_at_solution:
            for c in fixed_nlp.tmp_duals:
                if fixed_nlp.dual.get(c, None) is None:
                    fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
            dual_values = list(fixed_nlp.dual[c]
                            for c in fixed_nlp.MindtPy_utils.constraint_list)
        else:
            dual_values = None
        main_objective = fixed_nlp.MindtPy_utils.objective_list[-1]
        self.update_primal_bound(value(main_objective.expr))
        if self.primal_bound_improved:
            self.best_solution_found = fixed_nlp.clone()
            self.best_solution_found_time = get_main_elapsed_time(
                self.timing)
            if config.strategy == 'GOA':
                self.num_no_good_cuts_added.update(
                        {self.primal_bound: len(self.mip.MindtPy_utils.cuts.no_good_cuts)})

            # add obj increasing constraint for fp
            if fp:
                self.mip.MindtPy_utils.cuts.del_component(
                    'improving_objective_cut')
                if self.objective_sense == minimize:
                    self.mip.MindtPy_utils.cuts.improving_objective_cut = Constraint(expr=sum(self.mip.MindtPy_utils.objective_value[:])
                                                                                        <= self.primal_bound - config.fp_cutoffdecr*max(1, abs(self.primal_bound)))
                else:
                    self.mip.MindtPy_utils.cuts.improving_objective_cut = Constraint(expr=sum(self.mip.MindtPy_utils.objective_value[:])
                                                                                        >= self.primal_bound + config.fp_cutoffdecr*max(1, abs(self.primal_bound)))
        # Add the linear cut
        if config.strategy == 'OA' or fp:
            copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
                                self.mip.MindtPy_utils.variable_list,
                                config)
            add_oa_cuts(self.mip, dual_values, self.jacobians, self.objective_sense,
                        self.mip_constraint_polynomial_degree, self.mip_iter, config,
                        self.timing, cb_opt=cb_opt)
        elif config.strategy == 'GOA':
            copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
                                self.mip.MindtPy_utils.variable_list,
                                config)
            add_affine_cuts(self.mip, config, self.timing)
        # elif config.strategy == 'PSC':
        #     # !!THIS SEEMS LIKE A BUG!! - mrmundt #
        #     add_psc_cut(solve_data, config)
        # elif config.strategy == 'GBD':
        #     # !!THIS SEEMS LIKE A BUG!! - mrmundt #
        #     add_gbd_cut(solve_data, config)

        var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        if config.add_no_good_cuts:
            # TODO: fix
            add_no_good_cuts(self.mip, var_values, config, self.timing)

        # TODO: fix
        config.call_after_subproblem_feasible(fixed_nlp)

        config.logger.info(self.fixed_nlp_log_formatter.format('*' if self.primal_bound_improved else ' ',
                                                                    self.nlp_iter if not fp else self.fp_iter,
                                                                    'Fixed NLP', 
                                                                    value(main_objective.expr),
                                                                    self.primal_bound, 
                                                                    self.dual_bound, 
                                                                    self.rel_gap,
                                                                    get_main_elapsed_time(self.timing)))


    def handle_subproblem_infeasible(self, fixed_nlp, config, cb_opt=None):
        """Solves feasibility problem and adds cut according to the specified strategy.

        This function handles the result of the latest iteration of solving the NLP subproblem given an infeasible
        solution and copies the solution of the feasibility problem to the working model.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        config : ConfigBlock
            The specific configurations for MindtPy.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.
        """
        # TODO try something else? Reinitialize with different initial
        # value?
        config.logger.info('NLP subproblem was locally infeasible.')
        self.nlp_infeasible_counter += 1
        if config.calculate_dual_at_solution:
            for c in fixed_nlp.MindtPy_utils.constraint_list:
                rhs = value(c.upper) if c. has_ub() else value(c.lower)
                c_geq = -1 if c.has_ub() else 1
                fixed_nlp.dual[c] = (c_geq
                                    * max(0, c_geq * (rhs - value(c.body))))
            dual_values = list(fixed_nlp.dual[c]
                            for c in fixed_nlp.MindtPy_utils.constraint_list)
        else:
            dual_values = None

        # if config.strategy == 'PSC' or config.strategy == 'GBD':
        #     for var in fixed_nlp.component_data_objects(ctype=Var, descend_into=True):
        #         fixed_nlp.ipopt_zL_out[var] = 0
        #         fixed_nlp.ipopt_zU_out[var] = 0
        #         if var.has_ub() and abs(var.ub - value(var)) < config.absolute_bound_tolerance:
        #             fixed_nlp.ipopt_zL_out[var] = 1
        #         elif var.has_lb() and abs(value(var) - var.lb) < config.absolute_bound_tolerance:
        #             fixed_nlp.ipopt_zU_out[var] = -1

        if config.strategy in {'OA', 'GOA'}:
            config.logger.info('Solving feasibility problem')
            feas_subproblem, feas_subproblem_results = self.solve_feasibility_subproblem(config)
            # TODO: do we really need this?
            if self.should_terminate:
                return
            copy_var_list_values(feas_subproblem.MindtPy_utils.variable_list,
                                self.mip.MindtPy_utils.variable_list,
                                config)
            if config.strategy == 'OA':
                add_oa_cuts(self.mip, dual_values, self.jacobians, self.objective_sense,
                                self.mip_constraint_polynomial_degree, self.mip_iter, config,
                                self.timing, cb_opt=cb_opt)
            elif config.strategy == 'GOA':
                add_affine_cuts(self.mip, config, self.timing)
        # Add a no-good cut to exclude this discrete option
        var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        if config.add_no_good_cuts:
            # excludes current discrete option
            add_no_good_cuts(self.mip, var_values, config, self.timing)


    def handle_subproblem_other_termination(self, fixed_nlp, termination_condition, config):
        """Handles the result of the latest iteration of solving the fixed NLP subproblem given
        a solution that is neither optimal nor infeasible.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        termination_condition : Pyomo TerminationCondition
            The termination condition of the fixed NLP subproblem.
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MindtPy unable to handle the NLP subproblem termination condition.
        """
        if termination_condition is tc.maxIterations:
            # TODO try something else? Reinitialize with different initial value?
            config.logger.info(
                'NLP subproblem failed to converge within iteration limit.')
            var_values = list(
                v.value for v in fixed_nlp.MindtPy_utils.variable_list)
            if config.add_no_good_cuts:
                # excludes current discrete option
                add_no_good_cuts(self.mip, var_values, config, self.timing)

        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(termination_condition))


    def solve_feasibility_subproblem(self, config):
        """Solves a feasibility NLP if the fixed_nlp problem is infeasible.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.

        Returns
        -------
        feas_subproblem : Pyomo model
            Feasibility NLP from the model.
        feas_soln : SolverResults
            Results from solving the feasibility NLP.
        """
        feas_subproblem = self.working_model.clone()
        add_feas_slacks(feas_subproblem, config)

        MindtPy = feas_subproblem.MindtPy_utils
        if MindtPy.find_component('objective_value') is not None:
            MindtPy.objective_value[:].set_value(0, skip_validation=True)

        next(feas_subproblem.component_data_objects(
            Objective, active=True)).deactivate()
        for constr in feas_subproblem.MindtPy_utils.nonlinear_constraint_list:
            constr.deactivate()

        MindtPy.feas_opt.activate()
        if config.feasibility_norm == 'L1':
            MindtPy.feas_obj = Objective(
                expr=sum(s for s in MindtPy.feas_opt.slack_var[...]),
                sense=minimize)
        elif config.feasibility_norm == 'L2':
            MindtPy.feas_obj = Objective(
                expr=sum(s*s for s in MindtPy.feas_opt.slack_var[...]),
                sense=minimize)
        else:
            MindtPy.feas_obj = Objective(
                expr=MindtPy.feas_opt.slack_var,
                sense=minimize)
        TransformationFactory('core.fix_integer_vars').apply_to(feas_subproblem)
        nlpopt = SolverFactory(config.nlp_solver)
        nlp_args = dict(config.nlp_solver_args)
        set_solver_options(nlpopt, self.timing, config, solver_type='nlp')
        with SuppressInfeasibleWarning():
            try:
                with time_code(self.timing, 'feasibility subproblem'):
                    feas_soln = nlpopt.solve(feas_subproblem,
                                            tee=config.nlp_solver_tee,
                                            load_solutions=config.nlp_solver!='appsi_ipopt',
                                            **nlp_args)
                    if len(feas_soln.solution) > 0:
                        feas_subproblem.solutions.load_from(feas_soln)
            except (ValueError, OverflowError) as error:
                for nlp_var, orig_val in zip(
                        MindtPy.variable_list,
                        self.initial_var_values):
                    if not nlp_var.fixed and not nlp_var.is_binary():
                        nlp_var.set_value(orig_val, skip_validation=True)
                with time_code(self.timing, 'feasibility subproblem'):
                    feas_soln = nlpopt.solve(feas_subproblem,
                                            tee=config.nlp_solver_tee,
                                            load_solutions=config.nlp_solver!='appsi_ipopt',
                                            **nlp_args)
                    if len(feas_soln.solution) > 0:
                        feas_soln.solutions.load_from(feas_soln)
        self.handle_feasibility_subproblem_tc(
            feas_soln.solver.termination_condition, MindtPy, config)
        return feas_subproblem, feas_soln


    def handle_feasibility_subproblem_tc(self, subprob_terminate_cond, MindtPy, config):
        """Handles the result of the latest iteration of solving the feasibility NLP subproblem given
        a solution that is neither optimal nor infeasible.

        Parameters
        ----------
        subprob_terminate_cond : Pyomo TerminationCondition
            The termination condition of the feasibility NLP subproblem.
        MindtPy : Pyomo Block
            The MindtPy_utils block.
        config : ConfigBlock
            The specific configurations for MindtPy.
        """
        if subprob_terminate_cond in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            copy_var_list_values(
                MindtPy.variable_list,
                self.working_model.MindtPy_utils.variable_list,
                config)
            if value(MindtPy.feas_obj.expr) <= config.zero_tolerance:
                config.logger.warning('The objective value %.4E of feasibility problem is less than zero_tolerance. '
                                    'This indicates that the nlp subproblem is feasible, although it is found infeasible in the previous step. '
                                    'Check the nlp solver output' % value(MindtPy.feas_obj.expr))
        elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
            config.logger.error('Feasibility subproblem infeasible. '
                                'This should never happen.')
            self.should_terminate = True
            self.results.solver.status = SolverStatus.error
        elif subprob_terminate_cond is tc.maxIterations:
            config.logger.error('Subsolver reached its maximum number of iterations without converging, '
                                'consider increasing the iterations limit of the subsolver or reviewing your formulation.')
            self.should_terminate = True
            self.results.solver.status = SolverStatus.error
        else:
            config.logger.error('MindtPy unable to handle feasibility subproblem termination condition '
                                'of {}'.format(subprob_terminate_cond))
            self.should_terminate = True
            self.results.solver.status = SolverStatus.error
    
    ######################################################################################################################################################
    # iterate.py


    def algorithm_should_terminate(self, config, check_cycling):
        """Checks if the algorithm should terminate at the given point.

        This function determines whether the algorithm should terminate based on the solver options and progress.
        (Sets the self.results.solver.termination_condition to the appropriate condition, i.e. optimal,
        maxIterations, maxTimeLimit).

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        check_cycling : bool
            Whether to check for a special case that causes the discrete variables to loop through the same values.

        Returns
        -------
        bool
            True if the algorithm should terminate, False otherwise.
        """
        if self.should_terminate:
            if self.primal_bound == self.primal_bound_progress[0]:
                self.results.solver.termination_condition = tc.noSolution
            else:
                self.results.solver.termination_condition = tc.feasible
            return True

        # Check bound convergence
        if self.abs_gap <= config.absolute_bound_tolerance:
            config.logger.info(
                'MindtPy exiting on bound convergence. '
                'Absolute gap: {} <= absolute tolerance: {} \n'.format(
                    self.abs_gap, config.absolute_bound_tolerance))
            self.results.solver.termination_condition = tc.optimal
            return True
        # Check relative bound convergence
        if self.best_solution_found is not None:
            if self.rel_gap <= config.relative_bound_tolerance:
                config.logger.info(
                    'MindtPy exiting on bound convergence. '
                    'Relative gap : {} <= relative tolerance: {} \n'.format(
                        self.rel_gap, config.relative_bound_tolerance))

        # Check iteration limit
        if self.mip_iter >= config.iteration_limit:
            config.logger.info(
                'MindtPy unable to converge bounds '
                'after {} main iterations.'.format(self.mip_iter))
            config.logger.info(
                'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                format(self.primal_bound, self.dual_bound))
            if config.single_tree:
                self.results.solver.termination_condition = tc.feasible
            else:
                self.results.solver.termination_condition = tc.maxIterations
            return True

        # Check time limit
        if get_main_elapsed_time(self.timing) >= config.time_limit:
            config.logger.info(
                'MindtPy unable to converge bounds '
                'before time limit of {} seconds. '
                'Elapsed: {} seconds'
                .format(config.time_limit, get_main_elapsed_time(self.timing)))
            config.logger.info(
                'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                format(self.primal_bound, self.dual_bound))
            self.results.solver.termination_condition = tc.maxTimeLimit
            return True

        # Check if algorithm is stalling
        if len(self.primal_bound_progress) >= config.stalling_limit:
            if abs(self.primal_bound_progress[-1] - self.primal_bound_progress[-config.stalling_limit]) <= config.zero_tolerance:
                config.logger.info(
                    'Algorithm is not making enough progress. '
                    'Exiting iteration loop.')
                config.logger.info(
                    'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                    format(self.primal_bound, self.dual_bound))
                if self.best_solution_found is not None:
                    self.results.solver.termination_condition = tc.feasible
                else:
                    # TODO: Is it correct to set self.working_model as the best_solution_found?
                    # In function copy_var_list_values, skip_fixed is set to True in default.
                    self.best_solution_found = self.working_model.clone()
                    config.logger.warning(
                        'Algorithm did not find a feasible solution. '
                        'Returning best bound solution. Consider increasing stalling_limit or absolute_bound_tolerance.')
                    self.results.solver.termination_condition = tc.noSolution
                return True

        if config.strategy == 'ECP':
            # check to see if the nonlinear constraints are satisfied
            MindtPy = self.working_model.MindtPy_utils
            nonlinear_constraints = [c for c in MindtPy.nonlinear_constraint_list]
            for nlc in nonlinear_constraints:
                if nlc.has_lb():
                    try:
                        lower_slack = nlc.lslack()
                    except (ValueError, OverflowError):
                        # Set lower_slack (upper_slack below) less than -config.ecp_tolerance in this case.
                        lower_slack = -10*config.ecp_tolerance
                    if lower_slack < -config.ecp_tolerance:
                        config.logger.debug(
                            'MindtPy-ECP continuing as {} has not met the '
                            'nonlinear constraints satisfaction.'
                            '\n'.format(nlc))
                        return False
                if nlc.has_ub():
                    try:
                        upper_slack = nlc.uslack()
                    except (ValueError, OverflowError):
                        upper_slack = -10*config.ecp_tolerance
                    if upper_slack < -config.ecp_tolerance:
                        config.logger.debug(
                            'MindtPy-ECP continuing as {} has not met the '
                            'nonlinear constraints satisfaction.'
                            '\n'.format(nlc))
                        return False
            # For ECP to know whether to know which bound to copy over (primal or dual)
            self.primal_bound = self.dual_bound
            config.logger.info(
                'MindtPy-ECP exiting on nonlinear constraints satisfaction. '
                'Primal Bound: {} Dual Bound: {}\n'.format(self.primal_bound, self.dual_bound))

            self.best_solution_found = self.working_model.clone()
            self.results.solver.termination_condition = tc.optimal
            return True

        # Cycling check
        if check_cycling:
            if config.cycling_check or config.use_tabu_list:
                self.curr_int_sol = get_integer_solution(self.mip)
                if config.cycling_check and self.mip_iter >= 1:
                    if self.curr_int_sol in set(self.integer_list):
                        config.logger.info(
                            'Cycling happens after {} main iterations. '
                            'The same combination is obtained in iteration {} '
                            'This issue happens when the NLP subproblem violates constraint qualification. '
                            'Convergence to optimal solution is not guaranteed.'
                            .format(self.mip_iter, self.integer_list.index(self.curr_int_sol)+1))
                        config.logger.info(
                            'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                            format(self.primal_bound, self.dual_bound))
                        # TODO determine self.primal_bound, self.dual_bound is inf or -inf.
                        self.results.solver.termination_condition = tc.feasible
                        return True
                self.integer_list.append(self.curr_int_sol)

        # if not algorithm_is_making_progress(solve_data, config):
        #     config.logger.debug(
        #         'Algorithm is not making enough progress. '
        #         'Exiting iteration loop.')
        #     return True
        return False


    def fix_dual_bound(self, config, last_iter_cuts):
        """Fix the dual bound when no-good cuts or tabu list is activated.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        last_iter_cuts : bool
            Whether the cuts in the last iteration have been added.
        """
        if config.single_tree:
            config.logger.info(
                'Fix the bound to the value of one iteration before optimal solution is found.')
            try:
                self.dual_bound = self.stored_bound[self.primal_bound]
            except KeyError:
                config.logger.info('No stored bound found. Bound fix failed.')
        else:
            config.logger.info(
                'Solve the main problem without the last no_good cut to fix the bound.'
                'zero_tolerance is set to 1E-4')
            config.zero_tolerance = 1E-4
            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
            if not last_iter_cuts:
                fixed_nlp, fixed_nlp_result = self.solve_subproblem(config)
                self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result, config)

            MindtPy = self.mip.MindtPy_utils
            # deactivate the integer cuts generated after the best solution was found.
            if config.strategy == 'GOA':
                try:
                    valid_no_good_cuts_num = self.num_no_good_cuts_added[self.primal_bound]
                    if config.add_no_good_cuts:
                        for i in range(valid_no_good_cuts_num+1, len(MindtPy.cuts.no_good_cuts)+1):
                            MindtPy.cuts.no_good_cuts[i].deactivate()
                    if config.use_tabu_list:
                        self.integer_list = self.integer_list[:valid_no_good_cuts_num]
                except KeyError:
                    config.logger.info('No-good cut deactivate failed.')
            elif config.strategy == 'OA':
                # Only deactive the last OA cuts may not be correct.
                # Since integer solution may also be cut off by OA cuts due to calculation approximation.
                if config.add_no_good_cuts:
                    MindtPy.cuts.no_good_cuts[len(
                        MindtPy.cuts.no_good_cuts)].deactivate()
                if config.use_tabu_list:
                    self.integer_list = self.integer_list[:-1]
            if config.add_regularization is not None and MindtPy.find_component('mip_obj') is None:
                MindtPy.objective_list[-1].activate()
            mainopt = SolverFactory(config.mip_solver)
            # determine if persistent solver is called.
            if isinstance(mainopt, PersistentSolver):
                mainopt.set_instance(self.mip, symbolic_solver_labels=True)
            if config.use_tabu_list:
                tabulist = mainopt._solver_model.register_callback(
                    tabu_list.IncumbentCallback_cplex)
                self.solve_data = MindtPySolveData()
                self.export_attributes()
                tabulist.solve_data = self.solve_data
                tabulist.opt = mainopt
                tabulist.config = config
                mainopt._solver_model.parameters.preprocessing.reduce.set(1)
                # If the callback is used to reject incumbents, the user must set the
                # parameter c.parameters.preprocessing.reduce either to the value 1 (one)
                # to restrict presolve to primal reductions only or to 0 (zero) to disable all presolve reductions
                mainopt._solver_model.set_warning_stream(None)
                mainopt._solver_model.set_log_stream(None)
                mainopt._solver_model.set_error_stream(None)
            mip_args = dict(config.mip_solver_args)
            set_solver_options(mainopt, self.timing, config, solver_type='mip')
            main_mip_results = mainopt.solve(self.mip, 
                                            tee=config.mip_solver_tee, 
                                            load_solutions=False,
                                            **mip_args)
            if config.use_tabu_list:
                self.update_attributes()
            if len(main_mip_results.solution) > 0:
                self.mip.solutions.load_from(main_mip_results)

            if main_mip_results.solver.termination_condition is tc.infeasible:
                config.logger.info(
                    'Bound fix failed. The bound fix problem is infeasible')
            else:
                self.update_suboptimal_dual_bound(main_mip_results)
                config.logger.info(
                    'Fixed bound values: Primal Bound: {}  Dual Bound: {}'.
                    format(self.primal_bound, self.dual_bound))
            # Check bound convergence
            if abs(self.primal_bound - self.dual_bound) <= config.absolute_bound_tolerance:
                self.results.solver.termination_condition = tc.optimal

    ##########################################################################################################################################
    # mip_solve.py

    def solve_main(self, config, fp=False, regularization_problem=False):
        """This function solves the MIP main problem.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        fp : bool, optional
            Whether it is in the loop of feasibility pump, by default False.
        regularization_problem : bool, optional
            Whether it is solving a regularization problem, by default False.

        Returns
        -------
        self.mip : Pyomo model
            The MIP stored in self.
        main_mip_results : SolverResults
            Results from solving the main MIP.
        """
        if not fp and not regularization_problem:
            self.mip_iter += 1

        # setup main problem
        self.setup_main(config, fp, regularization_problem)
        mainopt = self.set_up_mip_solver(config, regularization_problem)

        mip_args = dict(config.mip_solver_args)
        if config.mip_solver in {'cplex', 'cplex_persistent', 'gurobi', 'gurobi_persistent'}:
            mip_args['warmstart'] = True
        set_solver_options(mainopt, self.timing, config,
                        solver_type='mip', regularization=regularization_problem)
        try:
            with time_code(self.timing, 'regularization main' if regularization_problem else ('fp main' if fp else 'main')):
                main_mip_results = mainopt.solve(self.mip,
                                                # tee=config.mip_solver_tee, 
                                                tee = True,
                                                load_solutions=False,
                                                **mip_args)
                self.mip.display()  #!!!!!!
                # update_attributes should be before load_from(main_mip_results), since load_from(main_mip_results) may fail.
                if config.single_tree or config.use_tabu_list:
                    self.update_attributes()
                if len(main_mip_results.solution) > 0:
                    self.mip.solutions.load_from(main_mip_results)
        except (ValueError, AttributeError):
            if config.single_tree:
                config.logger.warning('Single tree terminate.')
                if get_main_elapsed_time(self.timing) >= config.time_limit - 2:
                    config.logger.warning('due to the timelimit.')
                    self.results.solver.termination_condition = tc.maxTimeLimit
                if config.strategy == 'GOA' or config.add_no_good_cuts:
                    config.logger.warning('ValueError: Cannot load a SolverResults object with bad status: error. '
                                        'MIP solver failed. This usually happens in the single-tree GOA algorithm. '
                                        "No-good cuts are added and GOA algorithm doesn't converge within the time limit. "
                                        'No integer solution is found, so the cplex solver will report an error status. ')
            return None, None
        if config.solution_pool:
            main_mip_results._solver_model = mainopt._solver_model
            main_mip_results._pyomo_var_to_solver_var_map = mainopt._pyomo_var_to_solver_var_map
        if main_mip_results.solver.termination_condition is tc.optimal:
            if config.single_tree and not config.add_no_good_cuts and not regularization_problem:
                self.update_suboptimal_dual_bound(main_mip_results)
            if regularization_problem:
                config.logger.info(self.log_formatter.format(self.mip_iter, 'Reg '+self.regularization_mip_type,
                                                                value(self.mip.MindtPy_utils.roa_proj_mip_obj),
                                                                self.primal_bound, self.dual_bound, self.rel_gap,
                                                                get_main_elapsed_time(self.timing)))

        elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            main_mip_results, _ = distinguish_mip_infeasible_or_unbounded(
                self.mip, config)
            return self.mip, main_mip_results

        if regularization_problem:
            self.mip.MindtPy_utils.objective_constr.deactivate()
            self.mip.MindtPy_utils.del_component('roa_proj_mip_obj')
            self.mip.MindtPy_utils.cuts.del_component('obj_reg_estimate')
            if config.add_regularization == 'level_L1':
                self.mip.MindtPy_utils.del_component('L1_obj')
            elif config.add_regularization == 'level_L_infinity':
                self.mip.MindtPy_utils.del_component(
                    'L_infinity_obj')

        return self.mip, main_mip_results


    def set_up_mip_solver(self, config, regularization_problem):
        """Set up the MIP solver.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        regularization_problem : bool
            Whether it is solving a regularization problem.

        Returns
        -------
        mainopt : SolverFactory
            The customized MIP solver.
        """
        # Deactivate extraneous IMPORT/EXPORT suffixes
        if config.nlp_solver == 'ipopt':
            getattr(self.mip, 'ipopt_zL_out', _DoNothing()).deactivate()
            getattr(self.mip, 'ipopt_zU_out', _DoNothing()).deactivate()
        if regularization_problem:
            mainopt = SolverFactory(config.mip_regularization_solver)
        else:
            if config.mip_solver == 'gurobi_persistent' and config.single_tree:
                mainopt = GurobiPersistent4MindtPy()
                self.solve_data = MindtPySolveData()
                self.export_attributes()
                mainopt.solve_data = self.solve_data
                mainopt.config = config
            else:
                mainopt = SolverFactory(config.mip_solver)

        # determine if persistent solver is called.
        if isinstance(mainopt, PersistentSolver):
            mainopt.set_instance(self.mip, symbolic_solver_labels=True)
        if config.single_tree and not regularization_problem:
            # Configuration of cplex lazy callback
            if config.mip_solver == 'cplex_persistent':
                lazyoa = mainopt._solver_model.register_callback(
                    single_tree.LazyOACallback_cplex)
                # pass necessary data and parameters to lazyoa
                lazyoa.main_mip = self.mip
                self.solve_data = MindtPySolveData()
                self.export_attributes()
                lazyoa.solve_data = self.solve_data
                lazyoa.config = config
                lazyoa.opt = mainopt
                mainopt._solver_model.set_warning_stream(None)
                mainopt._solver_model.set_log_stream(None)
                mainopt._solver_model.set_error_stream(None)
            if config.mip_solver == 'gurobi_persistent':
                mainopt.set_callback(single_tree.LazyOACallback_gurobi)
        if config.use_tabu_list:
            self.solve_data = MindtPySolveData()
            self.export_attributes()
            tabulist = mainopt._solver_model.register_callback(
                tabu_list.IncumbentCallback_cplex)
            tabulist.solve_data = self.solve_data
            tabulist.opt = mainopt
            tabulist.config = config
            mainopt._solver_model.parameters.preprocessing.reduce.set(1)
            # If the callback is used to reject incumbents, the user must set the
            # parameter c.parameters.preprocessing.reduce either to the value 1 (one)
            # to restrict presolve to primal reductions only or to 0 (zero) to disable all presolve reductions
            mainopt._solver_model.set_warning_stream(None)
            mainopt._solver_model.set_log_stream(None)
            mainopt._solver_model.set_error_stream(None)
        return mainopt


    # The following functions deal with handling the solution we get from the above MIP solver function


    def handle_main_optimal(self, main_mip, config, update_bound=True):
        """This function copies the results from 'solve_main' to the working model and updates
        the upper/lower bound. This function is called after an optimal solution is found for 
        the main problem.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        config : ConfigBlock
            The specific configurations for MindtPy.
        update_bound : bool, optional
            Whether to update the bound, by default True.
            Bound will not be updated when handling regularization problem.
        """
        # proceed. Just need integer values
        MindtPy = main_mip.MindtPy_utils
        # check if the value of binary variable is valid
        for var in MindtPy.discrete_variable_list:
            if var.value is None:
                config.logger.warning(
                    f"Integer variable {var.name} not initialized.  "
                    "Setting it to its lower bound")
                var.set_value(var.lb, skip_validation=True)  # nlp_var.bounds[0]
        # warm start for the nlp subproblem
        copy_var_list_values(
            main_mip.MindtPy_utils.variable_list,
            self.working_model.MindtPy_utils.variable_list,
            config)

        if update_bound:
            self.update_dual_bound(value(MindtPy.mip_obj.expr))
            config.logger.info(self.log_formatter.format(self.mip_iter, 'MILP', value(MindtPy.mip_obj.expr),
                                                            self.primal_bound, self.dual_bound, self.rel_gap,
                                                            get_main_elapsed_time(self.timing)))


    def handle_main_other_conditions(self, main_mip, main_mip_results, config):
        """This function handles the result of the latest iteration of solving the MIP problem (given any of a few
        edge conditions, such as if the solution is neither infeasible nor optimal).

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        main_mip_results : SolverResults
            Results from solving the MIP problem.
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MindtPy unable to handle MILP main termination condition.
        """
        if main_mip_results.solver.termination_condition is tc.infeasible:
            self.handle_main_infeasible(main_mip, config)
        elif main_mip_results.solver.termination_condition is tc.unbounded:
            temp_results = self.handle_main_unbounded(main_mip, config)
        elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
            temp_results = self.handle_main_unbounded(main_mip, config)
            if temp_results.solver.termination_condition is tc.infeasible:
                self.handle_main_infeasible(main_mip, config)
        elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
            self.handle_main_max_timelimit(
                main_mip, main_mip_results, config)
            self.results.solver.termination_condition = tc.maxTimeLimit
        elif main_mip_results.solver.termination_condition is tc.feasible or \
            (main_mip_results.solver.termination_condition is tc.other and
            main_mip_results.solution.status is SolutionStatus.feasible):
            # load the solution and suppress the warning message by setting
            # solver status to ok.
            MindtPy = main_mip.MindtPy_utils
            config.logger.info(
                'MILP solver reported feasible solution, '
                'but not guaranteed to be optimal.')
            copy_var_list_values(
                main_mip.MindtPy_utils.variable_list,
                self.working_model.MindtPy_utils.variable_list,
                config)
            self.update_suboptimal_dual_bound(main_mip_results)
            config.logger.info(self.log_formatter.format(self.mip_iter, 'MILP', value(MindtPy.mip_obj.expr),
                                                            self.primal_bound, self.dual_bound, self.rel_gap,
                                                            get_main_elapsed_time(self.timing)))
        else:
            raise ValueError(
                'MindtPy unable to handle MILP main termination condition '
                'of %s. Solver message: %s' %
                (main_mip_results.solver.termination_condition, main_mip_results.solver.message))


    def handle_main_infeasible(self, main_mip, config):
        """This function handles the result of the latest iteration of solving
        the MIP problem given an infeasible solution.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        config : ConfigBlock
            The specific configurations for MindtPy.
        """
        config.logger.info(
            'MILP main problem is infeasible. '
            'Problem may have no more feasible '
            'binary configurations.')
        if self.mip_iter == 1:
            config.logger.warning(
                'MindtPy initialization may have generated poor '
                'quality cuts.')
        # TODO no-good cuts for single tree case
        # set optimistic bound to infinity
        # TODO: can we remove the following line?
        # self.dual_bound_progress.append(self.dual_bound)
        config.logger.info(
            'MindtPy exiting due to MILP main problem infeasibility.')
        if self.results.solver.termination_condition is None:
            if (self.primal_bound == float('inf') and self.objective_sense == minimize) or (self.primal_bound == float('-inf') and self.objective_sense == maximize):
            # if self.mip_iter == 0:
                self.results.solver.termination_condition = tc.infeasible
            else:
                self.results.solver.termination_condition = tc.feasible


    def handle_main_max_timelimit(self, main_mip, main_mip_results, config):
        """This function handles the result of the latest iteration of solving the MIP problem
        given that solving the MIP takes too long.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        main_mip_results : [type]
            Results from solving the MIP main subproblem.
        config : ConfigBlock
            The specific configurations for MindtPy.
        """
        # TODO if we have found a valid feasible solution, we take that, if not, we can at least use the dual bound
        MindtPy = main_mip.MindtPy_utils
        config.logger.info(
            'Unable to optimize MILP main problem '
            'within time limit. '
            'Using current solver feasible solution.')
        copy_var_list_values(
            main_mip.MindtPy_utils.variable_list,
            self.working_model.MindtPy_utils.variable_list,
            config)
        self.update_suboptimal_dual_bound(main_mip_results)
        config.logger.info(self.log_formatter.format(self.mip_iter, 'MILP', value(MindtPy.mip_obj.expr),
                                                        self.primal_bound, self.dual_bound, self.rel_gap,
                                                        get_main_elapsed_time(self.timing)))


    def handle_main_unbounded(self, main_mip, config):
        """This function handles the result of the latest iteration of solving the MIP 
        problem given an unbounded solution due to the relaxation.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        config : ConfigBlock
            The specific configurations for MindtPy.

        Returns
        -------
        main_mip_results : SolverResults
            The results of the bounded main problem.
        """
        # Solution is unbounded. Add an arbitrary bound to the objective and resolve.
        # This occurs when the objective is nonlinear. The nonlinear objective is moved
        # to the constraints, and deactivated for the linear main problem.
        MindtPy = main_mip.MindtPy_utils
        config.logger.warning(
            'main MILP was unbounded. '
            'Resolving with arbitrary bound values of (-{0:.10g}, {0:.10g}) on the objective. '
            'You can change this bound with the option obj_bound.'.format(config.obj_bound))
        MindtPy.objective_bound = Constraint(
            expr=(-config.obj_bound, MindtPy.mip_obj.expr, config.obj_bound))
        mainopt = SolverFactory(config.mip_solver)
        if isinstance(mainopt, PersistentSolver):
            mainopt.set_instance(main_mip)
        set_solver_options(mainopt, self.timing, config, solver_type='mip')
        with SuppressInfeasibleWarning():
            main_mip_results = mainopt.solve(main_mip,
                                            tee=config.mip_solver_tee,
                                            load_solutions=False,
                                            **config.mip_solver_args)
            if len(main_mip_results.solution) > 0:
                    self.mip.solutions.load_from(main_mip_results)
        return main_mip_results


    def handle_regularization_main_tc(self, main_mip, main_mip_results, config):
        """Handles the result of the latest FP iteration of solving the regularization main problem.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        main_mip_results : SolverResults
            Results from solving the regularization main subproblem.
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MindtPy unable to handle the regularization problem termination condition.
        """
        if main_mip_results is None:
            config.logger.info(
                'Failed to solve the regularization problem.'
                'The solution of the OA main problem will be adopted.')
        elif main_mip_results.solver.termination_condition in {tc.optimal, tc.feasible}:
            self.handle_main_optimal(
                main_mip, config, update_bound=False)
        elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
            config.logger.info(
                'Regularization problem failed to converge within the time limit.')
            self.results.solver.termination_condition = tc.maxTimeLimit
            # break
        elif main_mip_results.solver.termination_condition is tc.infeasible:
            config.logger.info(
                'Regularization problem infeasible.')
        elif main_mip_results.solver.termination_condition is tc.unbounded:
            config.logger.info(
                'Regularization problem ubounded.'
                'Sometimes solving MIQP in cplex, unbounded means infeasible.')
        elif main_mip_results.solver.termination_condition is tc.unknown:
            config.logger.info(
                'Termination condition of the regularization problem is unknown.')
            if main_mip_results.problem.lower_bound != float('-inf'):
                config.logger.info('Solution limit has been reached.')
                self.handle_main_optimal(main_mip, config, update_bound=False)
            else:
                config.logger.info('No solution obtained from the regularization subproblem.'
                                'Please set mip_solver_tee to True for more informations.'
                                'The solution of the OA main problem will be adopted.')
        else:
            raise ValueError(
                'MindtPy unable to handle regularization problem termination condition '
                'of %s. Solver message: %s' %
                (main_mip_results.solver.termination_condition, main_mip_results.solver.message))


    def setup_main(self, config, fp, regularization_problem):
        """Set up main problem/main regularization problem for OA, ECP, Feasibility Pump and ROA methods.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        fp : bool
            Whether it is in the loop of feasibility pump.
        regularization_problem : bool
            Whether it is solving a regularization problem.
        """
        MindtPy = self.mip.MindtPy_utils

        for c in MindtPy.constraint_list:
            if c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree:
                c.deactivate()

        MindtPy.cuts.activate()

        sign_adjust = 1 if self.objective_sense == minimize else - 1
        MindtPy.del_component('mip_obj')
        if regularization_problem and config.single_tree:
            MindtPy.del_component('roa_proj_mip_obj')
            MindtPy.cuts.del_component('obj_reg_estimate')
        if config.add_regularization is not None and config.add_no_good_cuts:
            if regularization_problem:
                MindtPy.cuts.no_good_cuts.activate()
            else:
                MindtPy.cuts.no_good_cuts.deactivate()

        if fp:
            MindtPy.del_component('fp_mip_obj')
            if config.fp_main_norm == 'L1':
                MindtPy.fp_mip_obj = generate_norm1_objective_function(
                    self.mip,
                    self.working_model,
                    discrete_only=config.fp_discrete_only)
            elif config.fp_main_norm == 'L2':
                MindtPy.fp_mip_obj = generate_norm2sq_objective_function(
                    self.mip,
                    self.working_model,
                    discrete_only=config.fp_discrete_only)
            elif config.fp_main_norm == 'L_infinity':
                MindtPy.fp_mip_obj = generate_norm_inf_objective_function(
                    self.mip,
                    self.working_model,
                    discrete_only=config.fp_discrete_only)
        elif regularization_problem:
            # The epigraph constraint is very "flat" for branching rules.
            # In ROA, if the objective function is linear(or quadratic when quadratic_strategy = 1 or 2), the original objective function is used in the MIP problem.
            # In the MIP projection problem, we need to reactivate the epigraph constraint(objective_constr).
            if MindtPy.objective_list[0].expr.polynomial_degree() in self.mip_objective_polynomial_degree:
                MindtPy.objective_constr.activate()
            if config.add_regularization == 'level_L1':
                MindtPy.roa_proj_mip_obj = generate_norm1_objective_function(self.mip,
                                                                            self.best_solution_found,
                                                                            discrete_only=False)
            elif config.add_regularization == 'level_L2':
                MindtPy.roa_proj_mip_obj = generate_norm2sq_objective_function(self.mip,
                                                                            self.best_solution_found,
                                                                            discrete_only=False)
            elif config.add_regularization == 'level_L_infinity':
                MindtPy.roa_proj_mip_obj = generate_norm_inf_objective_function(self.mip,
                                                                                self.best_solution_found,
                                                                                discrete_only=False)
            elif config.add_regularization in {'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag'}:
                MindtPy.roa_proj_mip_obj = generate_lag_objective_function(self.mip,
                                                                        self.best_solution_found,
                                                                        config,
                                                                        self.timing,
                                                                        discrete_only=False)
            if self.objective_sense == minimize:
                MindtPy.cuts.obj_reg_estimate = Constraint(
                    expr=sum(MindtPy.objective_value[:]) <= (1 - config.level_coef) * self.primal_bound + config.level_coef * self.dual_bound)
            else:
                MindtPy.cuts.obj_reg_estimate = Constraint(
                    expr=sum(MindtPy.objective_value[:]) >= (1 - config.level_coef) * self.primal_bound + config.level_coef * self.dual_bound)
        else:
            if config.add_slack:
                MindtPy.del_component('aug_penalty_expr')

                MindtPy.aug_penalty_expr = Expression(
                    expr=sign_adjust * config.OA_penalty_factor * sum(
                        v for v in MindtPy.cuts.slack_vars[...]))
            main_objective = MindtPy.objective_list[-1]
            MindtPy.mip_obj = Objective(
                expr=main_objective.expr +
                (MindtPy.aug_penalty_expr if config.add_slack else 0),
                sense=self.objective_sense)

            if config.use_dual_bound:
                # Delete previously added dual bound constraint
                MindtPy.cuts.del_component('dual_bound')
                if self.dual_bound not in {float('inf'), float('-inf')}:
                    if self.objective_sense == minimize:
                        MindtPy.cuts.dual_bound = Constraint(
                            expr=main_objective.expr +
                            (MindtPy.aug_penalty_expr if config.add_slack else 0) >= self.dual_bound,
                            doc='Objective function expression should improve on the best found dual bound')
                    else:
                        MindtPy.cuts.dual_bound = Constraint(
                            expr=main_objective.expr +
                            (MindtPy.aug_penalty_expr if config.add_slack else 0) <= self.dual_bound,
                            doc='Objective function expression should improve on the best found dual bound')


    def export_attributes(self):
        for name, val in self.__dict__.items():
            setattr(self.solve_data, name, val) 

    def update_attributes(self):
        for name, val in self.solve_data.__dict__.items():
            self.__dict__[name] = val

    def update_result(self):
        if self.objective_sense == minimize:
            self.results.problem.lower_bound = self.dual_bound
            self.results.problem.upper_bound = self.primal_bound
        else:
            self.results.problem.lower_bound = self.primal_bound
            self.results.problem.upper_bound = self.dual_bound

        self.results.solver.timing = self.timing
        self.results.solver.user_time = self.timing.total
        self.results.solver.wallclock_time = self.timing.total
        self.results.solver.iterations = self.mip_iter
        self.results.solver.num_infeasible_nlp_subproblem = self.nlp_infeasible_counter
        self.results.solver.best_solution_found_time = self.best_solution_found_time
        self.results.solver.primal_integral = self.get_primal_integral()
        self.results.solver.dual_integral = self.get_dual_integral()
        self.results.solver.primal_dual_gap_integral = self.results.solver.primal_integral + \
            self.results.solver.dual_integral


    def load_solution(self):
        # Update values in original model
        config = self.config
        MindtPy = self.working_model.MindtPy_utils
        copy_var_list_values(
            from_list=self.best_solution_found.MindtPy_utils.variable_list,
            to_list=MindtPy.variable_list,
            config=config)
        # The original does not have variable list. 
        # Use get_vars_from_components() should be used for both working_model and original_model to exclude the unused variables.
        self.working_model.MindtPy_utils.deactivate()
        # The original objective should be activated to make sure the variable list is in the same order (get_vars_from_components).
        self.working_model.MindtPy_utils.objective_list[0].activate()
        if self.working_model.find_component("_int_to_binary_reform") is not None:
            self.working_model._int_to_binary_reform.deactivate()
        # exclude fixed variables here. This is consistent with the definition of variable_list in GDPopt.util
        working_model_variable_list = list(get_vars_from_components(block=self.working_model, 
                                    ctype=(Constraint, Objective), 
                                    include_fixed=False, 
                                    active=True,
                                    sort=True, 
                                    descend_into=True,
                                    descent_order=None))
        original_model_variable_list = list(get_vars_from_components(block=self.original_model, 
                                    ctype=(Constraint, Objective), 
                                    include_fixed=False, 
                                    active=True,
                                    sort=True, 
                                    descend_into=True,
                                    descent_order=None))
        for v_from, v_to in zip(working_model_variable_list, original_model_variable_list):
            if v_from.name != v_to.name:
                raise ValueError('The name of the two variables is not the same. Loading final solution')
        copy_var_list_values(working_model_variable_list,
                            original_model_variable_list,
                            config=config)


    def check_config(self):
        """Checks if the configuration options make sense.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        """
        config = self.config
        # configuration confirmation
        if config.init_strategy == 'FP':
            config.add_no_good_cuts = True
            config.use_tabu_list = False

        if config.nlp_solver == 'baron':
            config.equality_relaxation = False
        if config.nlp_solver == 'gams' and config.nlp_solver.__contains__('solver'):
            if config.nlp_solver_args['solver'] == 'baron':
                config.equality_relaxation = False

        if config.solver_tee:
            config.mip_solver_tee = True
            config.nlp_solver_tee = True
        if config.add_no_good_cuts:
            config.integer_to_binary = True
        if config.use_tabu_list:
            config.mip_solver = 'cplex_persistent'
            if config.threads > 1:
                config.threads = 1
                config.logger.info(
                    'The threads parameter is corrected to 1 since incumbent callback conflicts with multi-threads mode.')
        if config.solution_pool:
            if config.mip_solver not in {'cplex_persistent', 'gurobi_persistent'}:
                if config.mip_solver in {'appsi_cplex', 'appsi_gurobi'}:
                    config.logger.info("Solution pool does not support APPSI solver.")
                config.mip_solver = 'cplex_persistent'
        if config.calculate_dual_at_solution:
            if config.mip_solver == 'appsi_cplex':
                config.logger.info("APPSI-Cplex cannot get duals for mixed-integer problems"
                                "mip_solver will be changed to Cplex.")
                config.mip_solver = 'cplex'
            if config.mip_regularization_solver == 'appsi_cplex':
                config.logger.info("APPSI-Cplex cannot get duals for mixed-integer problems"
                                "mip_solver will be changed to Cplex.")
                config.mip_regularization_solver = 'cplex'
            if config.mip_solver in {'gurobi', 'appsi_gurobi'} or \
                config.mip_regularization_solver in {'gurobi', 'appsi_gurobi'}:
                raise ValueError("GUROBI can not provide duals for mixed-integer problems.")

        if config.init_strategy == 'initial_binary' and config.strategy == 'ECP':
            raise ValueError("ECP method do not support 'initial_binary' as the initialization strategy.")
