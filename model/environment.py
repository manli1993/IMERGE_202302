'''
20220502 ysm 为什么有的def有return 有的没有：
#27 def setup_strategy_counter_dict :因为有self. 就是有对象的属性，只要这个对象存在在系统里，这个dictionary就一直存在，不断被修改，记录所有的变化；
#46 new_strategy_counter_dict 没有self， 可以理解为一个模板 每次要用这个模板 生成新的  return给要用的地方  用完 就释放了 这个东西就不在了 

20220602  行为把S6 S11 S12 去掉 ，即不考虑增产又治污
'''



import numpy as np
import copy

from melodie import AgentList, Environment
from .agent import PollutionAgent             #gml  从agent.py中import PollutionAgent


class PollutionEnvironment(Environment):

    def setup(self):
        self.scenario = self.current_scenario()
        self.government_accumulated_allowance = self.scenario.gov_pre_totalemission # 这里，只有初始化的时候加一次   没用到 Scenario表中改为0了
        self.government_check_probability = self.scenario.government_check_probability
        self.total_number_of_agents = 0
        self.caught_excess_pollution_of_current_period = 0.0
        self.total_pollution_allow_amount_current_year = 0.0
        self.total_pollution_emission_amount_current_year = 0.0
        self.new_agent_id_count = 0
        self.setup_strategy_counter_dict()                                        # 向周围人模仿  周围人采取行为的数量   越多人采取某行为  该行为的效用越大 = 自主效用*周围人数

    def setup_strategy_counter_dict(self):
        self.strategy_counter_dict = {
                                      "S2": 1,
                                      "S3": 1,
                                      "S4": 1,
                                      "S5": 1,
                                      #"S6": 1,
                                      "S7": 1,
                                      "S8": 1,
                                      "S9": 1,
                                      "S10": 1,
                                     # "S11": 1,
                                     # "S12": 1,
                                      "S1": 1
                                      }                                         

    def get_new_strategy_counter_dict(self):                                     #0617  不是新建企业   而是 相当于 strategy_counter_dict 见#128
        new_strategy_counter_dict = {
                                      "S2": 1,
                                      "S3": 1,
                                      "S4": 1,
                                      "S5": 1,
                                     # "S6": 1,
                                      "S7": 1,
                                      "S8": 1,
                                      "S9": 1,
                                      "S10": 1,
                                     # "S11": 1,
                                     # "S12": 1,
                                      "S1": 1
                                      }
        return new_strategy_counter_dict

    def setup_agents_random_parameters(self, agent_list: AgentList):
        for agent in agent_list:
           agent.setup_random_parameters()

#    def setup_random_scenario_parameters(self):
#        self.scenario.government_check_probability = self.scenario.government_check_probability * np.random.uniform(0.8, 1.2)


    def setup_agents_initial_pollution_allow_amount(self, agent_list: AgentList):
        for agent in agent_list:
            agent.setup_initial_pollution_allow_amount()

    def agents_calc_result_variables(self, agent_list: AgentList, t):
        for agent in agent_list:
            agent.calc_result_variables(self.government_check_probability, t)
            self.caught_excess_pollution_of_current_period += agent.excess_pollution_caught
        if self.caught_excess_pollution_of_current_period > 0 and self.government_check_probability < 1:
            self.government_check_probability += self.scenario.government_check_probability_increase_step

    def agents_update_state_variables(self, agent_list: AgentList):
        agent_list_sorted_by_pollution_intensity = []
        for agent in agent_list:
            agent_list_sorted_by_pollution_intensity.append(agent)                                          #学习 清洁生产 产污强度；后续  可学习 末端治理 去除率
        agent_list_sorted_by_pollution_intensity.sort(key=lambda x: x.pollution_generation_intensity)
        threshold_position = int(len(agent_list) * (1 - self.scenario.learning_threshold_percentage))        #即产污强度 在所有企业中的排序 倒数第多少家
        threshold_intensity = agent_list_sorted_by_pollution_intensity[threshold_position].pollution_generation_intensity

   #     pollution_intensity_list = [agent.pollution_generation_intensity for agent in agent_list]
    #    average_pollution_intensity = np.array(pollution_intensity_list).mean()                               # 向行业平均水平学   后续可改为向最低产污强度的学min   传到agent.py learn_intensity  没问题！！！
    #    average_pollution_intensity = np.array(pollution_intensity_list).min()                  
    #    average_pollution_intensity  =0.027    #0604GML  学最先进  要改  等同于输入表的limit+一个小小的数   造纸0.026（+0.01=0.027）  纺织 0.147（+0.001=0.148）  食品 0.0045（+0.0001=0.0046)
        average_pollution_intensity  =0.279               #学行业平均 造纸0.279 纺织2.36 食品0.184
    #    average_pollution_intensity = 0.04             #学造纸中位数（以50%排序），可假设 当做平均水平（其实是中等水平）
#向先进的 比我好的企业 随机取值    
    #    average_pollution_intensity = 0.04* np.random.uniform(1,1.48)    #0608 gml 造纸行业  产污强度 中位数0.04  在标杆-中位数 随机取  0.04/0.027=1.48
      #  print(average_pollution_intensity)
        new_strategy_counter_dict = self.get_new_strategy_counter_dict()                           
        for agent in agent_list:
            if agent.pollution_generation_intensity <= threshold_intensity:
                clean_production_learn_boolean = False
            else:
                clean_production_learn_boolean = True                                                          #如果大于产污强度阈值 （即排序在后20%）那就要向先进的去学
            strategy_chosen_name = agent.update_state_variables(clean_production_learn_boolean,
                                                                average_pollution_intensity,  
                                                                self.government_check_probability,
                                                                self.strategy_counter_dict,
                                                                copy.deepcopy(self.strategy_counter_dict)        #20220502 增加 self.strategy_counter_dict,
                                                                )    #学习周围人  strategy_counter_dict
            self.government_accumulated_allowance += agent.pollution_allow_amount_surplus                     # ？？政府的手里的许可量= 企业富余的 （即每年许可量-现有排放量），对应agent.py #296  #340  按道理  应该是 达标后 再减排 得到的量 可用于新建； 在此就假设比许可量少的 富余出来的量的一半 可用于新建
#            self.government_accumulated_allowance += agent.pollution_allow_amount_legally       #0415 新增
            rand = np.random.uniform(0, 1)
            if self.scenario.learning_success_only == 1:
                if agent.profit_increase_boolean > 0:
                    if rand <= self.scenario.observation_ratio:
                        new_strategy_counter_dict[strategy_chosen_name] += 1
                    else:
                        pass
                else:
                    pass
            else:
                if rand <= self.scenario.observation_ratio:
                    new_strategy_counter_dict[strategy_chosen_name] += 1
                else:
                    pass
        self.strategy_counter_dict = new_strategy_counter_dict

    def remove_agent_for_successtive_negative_profit(self, agent_list: AgentList):
        for agent in agent_list:
            if agent.profit_successtive_negative_count == 3:         #利润连续5年小于0  退出市场                 #对应agent.py #203行    没错！
                self.government_accumulated_allowance += agent.pollution_allow_amount                        #政府本身的许可量 增加 了倒闭退出企业的许可量
                agent_list.remove(agent)               

    def calc_total_pollution_allow_amount_current_year(self, agent_list: AgentList):
        for agent in agent_list:
            self.total_pollution_allow_amount_current_year += agent.pollution_allow_amount                 #当年的总许可量=所有企业的允许排放量之和   也包括新建企业
            
    def calc_total_pollution_emission_amount_current_year(self, agent_list: AgentList):                       #0415 新增  
        for agent in agent_list:
            self.total_pollution_emission_amount_current_year += agent.pollution_final_amount_after_update    #0415 新增    # 当年的所有排放量 等于 排放量之和
            

    def check_accumulated_allowance_and_if_enough_add_agent(self, agent_list: AgentList):
        check_boolean = True
        while check_boolean and self.caught_excess_pollution_of_current_period < 0:                                # 之前设计的是 ：当 监管查不到超排企业时， 增加 新建的 排放强度低的企业
            # 之前跑到t=6或7的时候，就会卡在这个while循环里跑很久，因为政府的allowance超级多，就不停地增加agent，需要调整相关参数 （详见上面45行）
            check_boolean = self.add_agent_with_low_pollution_intensity(agent_list)
        self.total_number_of_agents = len(agent_list)
        self.caught_excess_pollution_of_current_period = 0
        # 每个period，新增完一次agent之后要让这个变量归0，下一个period还是先看当期caught_excess_pollution_of_current_period是否为负，
        # 然后决定要不要增加agent。如果上一期没归0，下一期可能一上来就是负的，就不增加agent了。

    def add_agent_with_low_pollution_intensity(self, agent_list: AgentList):
        rand_id = np.random.randint(0, self.scenario.agent_num)
        agent_params = self.scenario.get_registered_dataframe('agent_params_scenario_independent')
        agent_params_dict = agent_params.iloc[rand_id].to_dict()
        new_agent_id = self.scenario.agent_num + self.new_agent_id_count
        new_agent = PollutionAgent(new_agent_id)
        new_agent.setup()
        new_agent.set_params(agent_params_dict)
        new_agent.id = new_agent_id
        new_agent.scenario = self.scenario
        new_agent.setup_initial_pollution_allow_amount()
# 注意 如果使用 行业绩效：废水量（产值*废水系数）*浓度，新建的许可量与排放量差距太大，因为都是命令 新企业的产污强度最小；如果 复制原企业的话 产污 强度 可不用写 下面这3行

        '''
#20220504 由于初始许可量等都是复制别的企业的  而产污强度取最小的  导致 排放量非常小  ,surplus 变多了，带来更多新企业，使得总许可量增加，总排放量增加 
        pollution_intensity_list = [agent.pollution_generation_intensity for agent in agent_list]              #20220405 由于初始许可量等都是复制别的企业的  而产污强度取最小的  导致 排放量非常小  导致新建企业后续不停增产  不合理 ；因此 #144-146去掉  （这个根源是
                                                                                                                 #根源是 之前许可量的计算 是按 废水量*系数  而排放量是按 产污强度*产值，导致差距很大；； 现在不存在这个情况  因为我设置了 初始许可量=排放量*0.9，这样 新建企业也会有许可量富余，但不多，合理
        intensity_min = min(pollution_intensity_list)
        new_agent.pollution_generation_intensity = intensity_min
        '''        
        
        new_agent.plant_age = 2 # plant_age = 2 means "new plant"

        pollution_allow_amount = new_agent.initial_pollution_allow_amount
#        total_pollution_allow_amount_to_allocate = 0.5 * (self.government_accumulated_allowance) + \
#                                                   100*(self.total_pollution_allow_amount_current_year)   #分别表示倍量替代许可量+政府储备库中为初始排污总量的5%  可用于新建企业
#        total_pollution_allow_amount_to_allocate = 0.5* (self.government_accumulated_allowance)            #gml20220502  倍量替代   accumulate为排放量-初始许可量 富余的量 加 倒闭企业退出的许可量
        total_pollution_allow_amount_to_allocate = 0.5* (self.government_accumulated_allowance)*0           #先不让新建企业进来了； 存量企业 达标后 再增产增排  就相当于新建进入吧
           
# total_pollution_allow_amount_to_allocate = 1.0 * (self.government_accumulated_allowance)                 #0405    run出来竟然没有新建企业
       # total_pollution_allow_amount_to_allocate = 0.5 * (self.government_accumulated_allowance)                 #0405  新存替代 （存量企业减排的量 用来倍量替代新企业，此时存量企业的许可量不变哦，见环保部官网）本身这个量就来自于储备库 因此不要current     
#        total_pollution_allow_amount_to_allocate = 0.5 * (self.government_accumulated_allowance)    #0415总新增许可量= 总许可-总排放 （即达标时多出来的  才能用于新增） pollution_allow_amount_legally
#        total_pollution_allow_amount_to_allocate = self.total_pollution_allow_amount_current_year - self.total_pollution_emission_amount_current_year 
#        total_pollution_allow_amount_to_allocate =  self.total_pollution_emission_amount_current_year - self.total_pollution_allow_amount_current_year                                         
    

        if total_pollution_allow_amount_to_allocate >= pollution_allow_amount:
            agent_list.add(new_agent)
            self.new_agent_id_count += 1
            self.government_accumulated_allowance -= pollution_allow_amount
            
            check_boolean = True
#            print("new agent added")
#            print("government_accumulated_allowance = " + str(self.government_accumulated_allowance))
            # 从这里的print可以看到，一直在增加agent，government_accumulated_allowance也一直在下降
            # 看完了可以把这两行print删掉了。
        else:
            check_boolean = False
        
        return check_boolean

























