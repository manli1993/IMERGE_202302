

from model.scenario import PollutionScenario
from model.model import PollutionModel
from model.simulator import PollutionSimulator
from config import config


if __name__ == "__main__":

    simulator = PollutionSimulator()
    simulator.run(
        config=config,
        scenario_class=PollutionScenario,
        model_class=PollutionModel,
    )

