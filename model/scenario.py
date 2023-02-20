
import numpy as np
from melodie import Scenario


class PollutionScenario(Scenario):

    def setup(self):
        self.periods = 0
        self.agent_num = 0
        self.risk_preference_param = 0.0
        self.utility_function_param = 0.0
        self.with_learning = 0               #如果启动学习机制 scenario中with_learning变为1  否则是0
        self.learning_success_only = 0
        self.observation_ratio = 0.0
        self.clean_production_adjustment_step1 = 0.0
        self.clean_production_adjustment_step2 = 0.0
        self.clean_production_adjustment_step3 = 0.0
        self.production_adjustment_step1_increase = 0.0
        self.production_adjustment_step1_decrease = 0.0
        self.production_adjustment_step2_increase = 0.0
        self.production_adjustment_step2_decrease = 0.0
        self.production_adjustment_step3_increase = 0.0
        self.production_adjustment_step3_decrease = 0.0
        self.end_removal_adjustment_step1 = 0.0
        self.end_removal_adjustment_step2 = 0.0
        self.end_removal_adjustment_step3 = 0.0
        self.clean_production_lifetime = 0
        self.return_rate = 0.0
        self.income_tax_rebate_rate = 0.0
        self.government_check_probability = 0.0
        self.government_check_probability_increase_step = 0.0
        self.learning_threshold_percentage = 0.0
        self.gov_pre_totalemission = 0.0









