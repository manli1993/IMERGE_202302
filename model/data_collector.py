
from melodie import DataCollector


class PollutionDataCollector(DataCollector):
    def setup(self):
        self.add_agent_property("agent_list", 'production')
        self.add_agent_property("agent_list", 'sector')
        self.add_agent_property("agent_list", 'plant_age')
        self.add_agent_property("agent_list", 'scale_type')
        self.add_agent_property("agent_list", 'pollution_generation_intensity')
        self.add_agent_property("agent_list", 'end_removal_rate')
        #self.add_agent_property("agent_list", "production_adjustment_action")
        self.add_agent_property("agent_list", 'production_increase_boolean')
        self.add_agent_property("agent_list", 'production_decrease_boolean')
        self.add_agent_property("agent_list", 'production_maintain_boolean')
        self.add_agent_property("agent_list", 'clean_production_improvement_boolean')

        self.add_agent_property("agent_list", 'end_removal_improvement_boolean')
        self.add_agent_property("agent_list", 'clean_production_benefit')
        self.add_agent_property("agent_list", 'end_removal_cost')
        self.add_agent_property("agent_list", 'pollution_allow_amount_surplus')
        self.add_agent_property("agent_list", 'waste_water_amount')
        self.add_agent_property("agent_list", 'pollution_allow_amount')
        self.add_agent_property("agent_list", 'pollution_generation_amount')
        self.add_agent_property("agent_list", 'pollution_final_amount')
        self.add_agent_property("agent_list", 'labor_cost')
        self.add_agent_property("agent_list", 'capital_cost')
        self.add_agent_property("agent_list", 'intermediate_input_cost')
        self.add_agent_property("agent_list", 'income_tax')
        self.add_agent_property("agent_list", 'environment_tax_or_waster_water_fee')
        self.add_agent_property("agent_list", 'clean_production_benefit')
        self.add_agent_property("agent_list", 'end_removal_cost')
        self.add_agent_property("agent_list", 'excess_pollution_punishment')
        self.add_agent_property("agent_list", 'profit')
        self.add_agent_property("agent_list", 'profit_increase_boolean')
        self.add_agent_property("agent_list", 'profit_successtive_negative_count')
        self.add_agent_property("agent_list", 'firm_quit_boolean')
        self.add_agent_property("agent_list", 'initial_pollution_allow_amount')
        self.add_agent_property("agent_list", 'initial_permit')      #0530
        self.add_environment_property('government_accumulated_allowance')
        self.add_environment_property('government_check_probability')
        self.add_environment_property('total_number_of_agents')
        self.add_environment_property('new_agent_id_count')
        
