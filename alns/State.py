from typing import Protocol, Union
import pyscipopt as scip


class State(Protocol):
    """
    Protocol for a solution state. Solutions should define an ``objective()``
    member function for evaluation.
    """

    def objective(self) -> float:
        """
        Computes the state's associated objective value.
        """


class SCIPState(State):
    """
    Implementation of the State protocol for SCIP
    """
    def __init__(self, problem_instance: Union[str, scip.Model]):
        self.problem_instance = problem_instance
        self.model = None
        if isinstance(problem_instance, str):
            self.problem_instance_file = problem_instance
            self.create_model()
        elif isinstance(problem_instance, scip.Model):
            self.model = problem_instance

        else:
            raise ValueError("Invalid type for problem instance")

        self.model.hideOutput()

    def create_model(self):
        self.model = scip.Model()
        self.model.readProblem(self.problem_instance_file)

    def objective(self) -> float:
        if self.model is None:
            self.create_model()
        self.model.optimize()
        return self.model.getObjVal()

    def get_model(self):
        if self.model is None:
            self.create_model()
        return self.model

    def get_solution(self):
        if self.model is None:
            self.objective()
        # Ensure we have solution before trying to retrieve it.
        if self.model.getNSols() > 0:
            return self.model.getSols()
        else:
            return None

    def copy(self):
        new_state = SCIPState(self.problem_instance)
        return new_state
