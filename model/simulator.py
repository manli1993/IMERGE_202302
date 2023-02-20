
import sqlalchemy
from melodie import Simulator
from .scenario import PollutionScenario


class PollutionSimulator(Simulator):

    def register_scenario_dataframe(self):
        scenarios_dict = {}
        self.load_dataframe('scenarios', 'C1_scenarios_COD_Papermaking_50家造纸_增产0.2.xlsx', scenarios_dict)

    def register_static_dataframes(self):
        agent_params_scenario_independent_dict = {'id':sqlalchemy.Integer(),
                                                  'sector': sqlalchemy.Integer(),
                                                  'plant_age': sqlalchemy.Integer(),
                                                  'scale_type': sqlalchemy.Integer(),
                                                  'production': sqlalchemy.Float(),
                                                  'price': sqlalchemy.Float(),
                                                  'production_value_threshold1': sqlalchemy.Float(),
                                                  'production_value_threshold2': sqlalchemy.Float(),
                                                  'labor_elasticity': sqlalchemy.Float(),
                                                  'capital_elasticity': sqlalchemy.Float(),
                                                  'total_factor_productivity': sqlalchemy.Float(),
                                                  'intermediate_input_param': sqlalchemy.Float(),
                                                  'pollution_type': sqlalchemy.Integer(),
                                                  'pollution_limit_direct_old': sqlalchemy.Float(),
                                                  'pollution_limit_direct_new': sqlalchemy.Float(),
                                                  'pollution_limit_indirect': sqlalchemy.Float(),
                                                  'waste_water_amount_param1': sqlalchemy.Float(),
                                                  'waste_water_amount_param2': sqlalchemy.Float(),
                                                  'pollution_generation_intensity': sqlalchemy.Float(),
                                                  'pollution_generation_intensity_limit': sqlalchemy.Float(),
                                                  'end_removal_rate': sqlalchemy.Float(),
                                                  'end_removal_rate_limit': sqlalchemy.Float(),
                                                  'clean_production_invest_cost_param': sqlalchemy.Float(),
                                                  'clean_production_benefit_param': sqlalchemy.Float(),
                                                  'end_removal_invest_cost_param1': sqlalchemy.Float(),
                                                  'end_removal_invest_cost_param2': sqlalchemy.Float(),
                                                  'end_removal_operate_cost_param1': sqlalchemy.Float(),
                                                  'end_removal_operate_cost_param2': sqlalchemy.Float(),
                                                  'end_removal_operate_cost_param3': sqlalchemy.Float(),
                                                  'income_tax_rate': sqlalchemy.Float(),
                                                  'environment_tax_rate': sqlalchemy.Float(),
                                                  'waste_water_fee': sqlalchemy.Float(),
                                                  'excess_pollution_fixed_punishment': sqlalchemy.Float(),
                                                  'excess_pollution_variable_punishment': sqlalchemy.Float(),
                                                  'annual_abatement_ratio': sqlalchemy.Float(),
                                                 # 'initial_permit': sqlalchemy.Float()
                                                  'initial_pollution_allow_amount':sqlalchemy.Float()}   #0530新增加initial_permit
        self.load_dataframe('agent_params_scenario_independent',
                            '2015Papermaking.xlsx',
                            agent_params_scenario_independent_dict)






