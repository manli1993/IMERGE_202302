
from melodie import Model
from .agent import PollutionAgent
from .data_collector import PollutionDataCollector
from .environment import PollutionEnvironment


class PollutionModel(Model):

    def setup(self):
        self.agent_list = self.create_agent_container(PollutionAgent,
                                                      self.scenario.agent_num,
                                                      self.scenario.get_registered_dataframe('agent_params_scenario_independent'))

        with self.define_basic_components():
            self.environment = PollutionEnvironment()
            self.data_collector = PollutionDataCollector()

    def run(self):
#        self.environment.setup_random_scenario_parameters()
#        self.environment.setup_agents_random_parameters(self.agent_list)
        self.environment.setup_agents_initial_pollution_allow_amount(self.agent_list)
        for t in range(0, self.scenario.periods):
#            print("t = " + str(t))
            self.environment.agents_calc_result_variables(self.agent_list, t)
            self.environment.agents_update_state_variables(self.agent_list)
            self.environment.remove_agent_for_successtive_negative_profit(self.agent_list)
            self.environment.calc_total_pollution_allow_amount_current_year(self.agent_list)
            self.environment.check_accumulated_allowance_and_if_enough_add_agent(self.agent_list)
            self.data_collector.collect(t)
        self.data_collector.save()
