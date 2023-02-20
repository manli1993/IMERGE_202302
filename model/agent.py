'''
20220227 于松民师兄帮修改  许可量分配机制（初始排污权、政府储备库新建企业）、学习机制with_learning
20220328  荣易帮改  #411-415  如果达标 用利润最大化决策；如果超标，用风险最小化决策  （可作为一种决策机制情景讨论）
20220329 王璐瑶帮改 增加#475-481   表示只选择概率最大（效用最大）的行为，而不是随机选择
20220407 璐瑶帮改  #471 达标时 如果排放量*（1+step）小于等于许可量，选择S5；如果 排放量<许可量<排放量*（1+step),选择什么也不做
20200501  GML 增产 减产 清洁生产 step 随机采样3%-8%   初始许可量=排放量*0.9  def中增加t
20200503 修改效用函数 精简版
20220504 效用函数（利润 、风险、组合）  参数均匀与正态分布 随机采 最正确版


20220531 在0505的基础上 copy0523的行为 如 #123 self.firm_quit_boolean = False；   同时，在输入表中增加一列initial_permit,初始排放权直接在输入表中修改！！！ 成功   可用于 基准情景在输入表中直接输入许可量；可用于比较不同决策机制 、不同分配原则
         这个代码 没改企业达标后腾出许可量用于新建（璐瑶帮改surplus  #300行左右），可见0530新存替代”那个代码

0531 13:19 利润最大化

0602 生产治污行为学习模仿   #627代码 两种：1、行为选择概率 只考虑周围人；1、行为选择概率 为P自我决策*P周围人

0603 技术学习   落后的a%向行业平均或最先进水平去学 average  记得改env.py 的average  赋予它 输入表中 最先进 和行业平均；注意最先进 不能等于limit 要稍大于  否则没法选clean_boolean

0608 技术学习（刘老师建议）随机向比自己好的学，RUN N次 得到的平均产污强度与 真实值比较，进而说明 向先进学 向平均学 随机学  哪种最符合真实情况！！！   我 假设排序后50%的向行业平均-标杆的产污强度随机取值  改average

0908  输入表中行为变化系数step用论文中的数据，以排污许可平台上爬取的现实数据 作为许可量约束initial
'''

import numpy as np

from melodie import Agent


class PollutionAgent(Agent):

    def setup(self):

        # strategy positions:
        # 1. pollution_generation_intensity_adjustment
        # 2. production_adjustment
        # 3. end_removal_rate_adjustment
        self.strategy_dict = {
                              "S2": (0, 0, 1),
                              "S3": (0, -1, 0),
                              "S4": (0, -1, 1),
                              "S5": (0, 1, 0),
                              #"S6": (0, 1, 1),
                              "S7": (-1, 0, 0),
                              "S8": (-1, 0, 1),
                              "S9": (-1, -1, 0),
                              "S10": (-1, -1, 1),
                              #"S11": (-1, 1, 0),
                              #"S12": (-1, 1, 1),
                              "S1": (0, 0, 0)
                             }

        """
        scenario independent attributes
        """
        self.sector = 0
        self.plant_age = 0
        self.scale_type = 0
        self.price = 0.0
        self.production_value_threshold1 = 0.0
        self.production_value_threshold2 = 0.0
        self.labor_elasticity = 0.0
        self.capital_elasticity = 0.0
        self.total_factor_productivity = 0.0
        self.intermediate_input_param = 0.0

        self.waste_water_amount_param1 = 0.0
        self.waste_water_amount_param2 = 0.0
        self.pollution_generation_intensity_limit = 0.0
        self.end_removal_rate_limit = 0.0
        self.pollution_type = 0
        self.pollution_limit_direct_old = 0.0
        self.pollution_limit_direct_new = 0.0
        self.pollution_limit_indirect = 0.0

        self.clean_production_invest_cost_param = 0.0
        self.clean_production_benefit_param = 0.0
        self.end_removal_invest_cost_param1 = 0.0
        self.end_removal_invest_cost_param2 = 0.0
        self.end_removal_operate_cost_param1 = 0.0
        self.end_removal_operate_cost_param2 = 0.0
        self.end_removal_operate_cost_param3 = 0.0

        self.income_tax_rate = 0.0
        self.environment_tax_rate = 0.0
        self.waste_water_fee = 0.0
        self.excess_pollution_fixed_punishment = 0.0
        self.excess_pollution_variable_punishment = 0.0

        self.annual_abatement_ratio = 0.0
        self.initial_pollution_allow_amount = 0.0
        self.initial_permit = 0.0    #20220530补充  初始排污权直接用输入表中的  #143

        """
        variables that change every period
        
        each period includes two steps:
         - first, the agent calculates the "result variables" (based on "state variables") for the current period
         - second, the agent updates the "state variables" for the next period
        """
        # state variables that will be strategically updated every period          #每期会更新的量
        self.production = 0.0
        self.pollution_generation_intensity = 0.0
        self.end_removal_rate = 0.0

        self.production_adjustment_action = 0 # no_action = 0, decrease = 1, increase = 2
        self.production_increase_boolean = False               #gml 0511增产行为个数
        self.production_decrease_boolean = False               #gml 0511减产行为个数
        self.production_maintain_boolean = False               #gml 0512
        self.clean_production_improvement_boolean = False
        self.clean_production_benefit = 0.0
        self.end_removal_improvement_boolean = False
        self.end_removal_cost = 0.0
        self.pollution_allow_amount_surplus = 0.0
   #     self.pollution_allow_amount_legally = 0.0               #0415gml新增

        # result variables that will be calculated for each period                  #每期会计算的量
        self.waste_water_amount = 0.0
        self.pollution_allow_amount = 0.0
        self.pollution_generation_amount = 0.0
        self.pollution_final_amount = 0.0
        self.excess_pollution_caught = 0.0

        self.labor_cost = 0.0
        self.capital_cost = 0.0
        self.intermediate_input_cost = 0.0
        self.income_tax = 0.0
        self.environment_tax_or_waster_water_fee = 0.0
        self.clean_production_benefit = 0.0
        self.end_removal_cost = 0.0
        self.excess_pollution_punishment = 0.0
        self.profit = 0.0
        self.profit_increase_boolean = 0
        self.profit_successtive_negative_count = 0
        self.firm_quit_boolean = False
        
        
        

    """
    randomize parameters   #参数随机采样用  #20220501 把增产 减产 清洁生产 的减排变化幅度在0.6*0.9 ~ 0.6*1.3随机均匀采  去除率还是按照6%
    """
#    def setup_random_parameters(self):
        
        
        
    """
    setup initial pollution allow amount 
    """
    
    def setup_initial_pollution_allow_amount(self):
        
   #     self.initial_pollution_allow_amount = self.waste_water_amount_param1
      #  self.initial_pollution_allow_amount = self.initial_permit   #Gml 0530这行意味着 在输入表中新增加一列 初始许可量initial_permit 这样的话 可以提前在输入表里把初始许可量搞出来
        '''
        waste_water_amount = self.calc_waste_water_amount(self.production)
        if self.pollution_type == 0: # value = 0: direct pollution
            if self.plant_age == 1: # plant_age = 1 means "old plant"
                pollution_limit = self.pollution_limit_direct_old
            else:
                pollution_limit = self.pollution_limit_direct_new                  #新建企业的许可量 =废水量*严格的新建企业浓度
        else: # value = 1: indirect pollution
            pollution_limit = self.pollution_limit_indirect
        self.initial_pollution_allow_amount = waste_water_amount * pollution_limit * 10**(-6)   
        print(self.initial_pollution_allow_amount)
        '''
        '''
        #以上是按废水量*浓度限值   即按行业绩效的许可量；     间排 直排 新建 的浓度限值都不一样    ；后续讨论不同分配原则时可用
         gml20220501   以下是按 许可量==产值*产污强度*（1-去除率）=初始排放量*a   a=0.9 即等比例减排10%
         由于我专门依据减排10%  把废水量的系数给调整了，现在这两种方式算的结果一样  ；后续讨论减排目标，调许可量的时候 还是第二种方便一些 直接改0.9 0.8...；新建初始许可量也等于 =排放量*0.9
        '''
#20220503 修改上面的  final排放量 加上self  
        
        self.pollution_generation_amount = self.calc_pollution_generation_amount(self.production,
                                                                                 self.pollution_generation_intensity)
        self.pollution_final_amount = self.calc_pollution_final_amount(self.pollution_generation_amount,
                                                                       self.end_removal_rate)
 #       self.initial_pollution_allow_amount = self.pollution_final_amount*1.1       #注意：排污量不变  初始许可量为固定值   不同于#224 return的排放量

#        if self.id == 9:
#            print(self.initial_pollution_allow_amount)   #id=9这家企业 初始许可量为7.92；
#            print(self.pollution_final_amount)           #id=9这家企业 初始排放量为8.80；  8.8*0.9=7.92 对！
        
        self.pollution_allow_amount = self.initial_pollution_allow_amount   #0522 璐瑶补充此行代码  不然 许可量初始为0.0
       # print(self.initial_pollution_allow_amount)   #gml0501 成功！
        #self.pollution_allow_amount = self.initial_permit 
     #   if self.id == 9:
     #       print(self.initial_permit) 
    """
    calculate the result variables for the current period
    """
    def calc_result_variables(self, government_check_probability, t):
        self.waste_water_amount = self.calc_waste_water_amount(self.production)
        self.pollution_allow_amount = self.calc_pollution_allow_amount(t)
        self.pollution_generation_amount = self.calc_pollution_generation_amount(self.production,
                                                                                 self.pollution_generation_intensity)
        self.pollution_final_amount = self.calc_pollution_final_amount(self.pollution_generation_amount,
                                                                       self.end_removal_rate)
        self.production_value = self.production * self.price
        self.labor_cost = self.calc_labor_cost(self.production)
        self.capital_cost = self.calc_capital_cost(self.production)
        self.intermediate_input_cost = self.calc_intermediate_input_cost(self.production)
        self.income_tax = self.calc_income_tax(self.production)
        self.environment_tax_or_waster_water_fee = self.calc_environment_tax_or_waster_water_fee(self.waste_water_amount,
                                                                                                 self.pollution_final_amount)
        if np.random.uniform(0, 1) < government_check_probability:
            self.excess_pollution_punishment = self.calc_excess_pollution_punishment(self.pollution_final_amount,
                                                                                     self.pollution_allow_amount)
            self.excess_pollution_caught = self.pollution_final_amount - self.pollution_allow_amount
        else:
            self.excess_pollution_punishment = 0
            self.excess_pollution_caught = 0

        profit = self.production_value - self.labor_cost - self.capital_cost - self.intermediate_input_cost - \
                  self.income_tax - self.environment_tax_or_waster_water_fee + self.clean_production_benefit - \
                  self.end_removal_cost - self.excess_pollution_punishment                                                 #注意 清洁生产回报 用加法

        if profit > self.profit:                              #20220503  Ithink self.profit  应该是是当年利润（行动前）；profit 应该是行动后的利润    不同于#211 
            self.profit_increase_boolean = 1                #盈利的话 企业数量记为1   （I think也可像亏本一样 改为+=1  连续几年盈利）
        else:
            self.profit_increase_boolean = 0

        self.profit = profit                                  #利润更新了 self利润 是行动后的利润了
        if self.profit < 0:                                   #如果当年利润小于0，那么亏本企业数量就+1 
            self.profit_successtive_negative_count += 1     #对应Environment.py #122  
        else:
            self.profit_successtive_negative_count = 0

#        if self.id == 19:
#            print(profit)
#            print(self.profit_increase_boolean)

        return profit                                       #gml 20220503 新添加这行

    def calc_waste_water_amount(self, production):               #如果初始排放权不用废水量*浓度的计算方式，就不需要 废水量 变量；废水量 这个变量用在 排污费fee 间排#283行
        production_value = production * self.price
#        waste_water_amount = np.e ** (self.waste_water_amount_param1 +
#                                     self.waste_water_amount_param2 * np.log(production_value))
        waste_water_amount = production_value*self.waste_water_amount_param1   #  gml 0310 gml0227 改废水量公式  造纸 每万元产废水43.69吨  （2016环统）别的行业；用43.69算的许可量远大于实际排放量，因此改为11试试
        return waste_water_amount

    def calc_pollution_allow_amount(self, t):
        power = int(t/1)                              #ysm20220228   表示许可量 每隔5年 变一次； 讨论动态许可量时  可把5变成3、 1  ，如果变成1 表示许可量逐年递减！  #gml0310 t/1
        pollution_allow_amount = self.initial_pollution_allow_amount * (1 - self.annual_abatement_ratio) ** power    
#        pollution_allow_amount = self.initial_pollution_allow_amount*0.9**int(t/1)      也可写成这种形式，更方便些 不用改输入表annual了
#        if self.id == 0:
#            print(pollution_allow_amount)        
#        print(pollution_allow_amount)          #0503 目前看每年许可量没有变化   #为什么每家企业有24个一样的数？！！！不知道
        return pollution_allow_amount
 
            
            
    def calc_pollution_generation_amount(self, production, pollution_generation_intensity):
        production_value = production * self.price
        pollution_generation_amount = production_value * pollution_generation_intensity
        return pollution_generation_amount

    def calc_pollution_final_amount(self, pollution_generation_amount, end_removal_rate):
        pollution_final_amount = pollution_generation_amount * (1 - end_removal_rate)
        return pollution_final_amount

    def calc_labor_cost(self, production):
        production_value = production * self.price
        n = 0.05
        part_1 = (production_value / np.e ** self.total_factor_productivity)
        part_2 = (self.labor_elasticity / (self.capital_elasticity * n)) ** self.capital_elasticity
        labor_cost = (part_1 * part_2) ** (1 / (self.labor_elasticity + self.capital_elasticity))    #0302 gml让L和K乘0 ，以免出现利润为负的情况，但还是会出现负利润，说明清洁生产等太贵了
        return labor_cost

    def calc_capital_cost(self, production):
        production_value = production * self.price
        n = 0.05
        part_1 = (production_value / np.e ** self.total_factor_productivity)
        part_2 = ((self.capital_elasticity * n) / self.labor_elasticity) ** self.labor_elasticity
        capital_cost = (part_1 * part_2) ** (1 / (self.labor_elasticity + self.capital_elasticity))
        return capital_cost

    def calc_intermediate_input_cost(self, production):
        production_value = production * self.price
        intermediate_input_cost = production_value * self.intermediate_input_param
        return intermediate_input_cost

    def calc_income_tax(self, production):
        production_value = production * self.price
        income_tax = production_value * self.income_tax_rate
        if self.clean_production_improvement_boolean:
            return income_tax * (1 - self.income_tax_rate)
        else:
            return income_tax

    def calc_environment_tax_or_waster_water_fee(self, waste_water_amount, pollution_final_amount):
        if self.pollution_type == 0: # value = 0: direct pollution
            fee = pollution_final_amount * self.environment_tax_rate # check if the unit is right
        else: # value = 1: indirect pollution
            fee = waste_water_amount * self.waste_water_fee                    #排污费 用到废水量 
        return fee

    def calc_excess_pollution_punishment(self, pollution_final_amount, pollution_allow_amount):
        punishment = 0
        if pollution_final_amount > pollution_allow_amount:
            excess_pollution = pollution_final_amount - pollution_allow_amount
            fix_punishment = self.excess_pollution_fixed_punishment
            variable_punishment = self.excess_pollution_variable_punishment * excess_pollution   #excess_pollution_variable_punishment是agent输入表中 1.4万元
            punishment = fix_punishment + variable_punishment
        else:
            pass
        return punishment


    """
    update the state variables for the next period
    """
    def update_state_variables(self,
                               clean_production_learn_boolean,
                               learned_pollution_generation_intensity,
                               t,                                      #0502 新增t 否则会报错 positional argument
                               government_check_probability,
                               strategy_counter_dict):   #0415新增pollution_allow_amount 对应 #308  pollution_allow_amount,initial_pollution_allow_amount
        '''
        pollution_generation_amount_before_update = self.calc_pollution_generation_amount(
            self.production,
            self.pollution_generation_intensity
        )
        pollution_final_amount_before_update = self.calc_pollution_final_amount(
            pollution_generation_amount_before_update,
            self.end_removal_rate
        )
        '''
#        pollution_final_amount_before_update = self.initial_pollution_allow_amount   #0415!!! before=初始许可量  
        pollution_final_amount_before_update = self.pollution_allow_amount   #20220502  排放量 等于许可量  ，对应于#340 surplus

###较重要！行为决策
        strategy_utility_dict = self.calc_strategy_utility_dict(clean_production_learn_boolean,
                                                                learned_pollution_generation_intensity,
                                                                t,
                                                                government_check_probability
                                                                )                    #   calc_strategy_utility_dict  在#537行定义
        strategy_chosen_name = self.strategy_choice(strategy_utility_dict, strategy_counter_dict)               # strategy_utility_dict, strategy_counter_dict   在#574行定义
        strategy_chosen_touple = self.strategy_dict[strategy_chosen_name]

        [strategy_chosen_pollution_generation_intensity,
         self.production,
         strategy_chosen_end_removal_rate] = self.generate_strategy_state_variables(strategy_chosen_touple)   #gml0502  touple只在这里和上行出现，意思是选择 某个行为的名称strategy_chosen_name 
#0605gml 让learned =具体数  不用写 ；定义env.py #94行average= 行业最低产污强度+0.01（很小的数）即可！因为#351  要让这个数大于输入表的limit 才能技术学习！！！！！！！
       # learned_pollution_generation_intensity=0.027
        if clean_production_learn_boolean:                     #技术学习
            strategy_chosen_pollution_generation_intensity = learned_pollution_generation_intensity
            
#0603 learned_intensity 是多少呢 应该是等于平均产污强度 或最小产污强度      没问题，learned==environment.py 中 average_pollution_intensity = np.array(pollution_intensity_list).min()  或mean()       
           ### learned_pollution_generation_intensity = 0.01
     #   if self.id ==66:
     #       print(strategy_chosen_pollution_generation_intensity)
      #  print(strategy_chosen_pollution_generation_intensity)

#如果采取行为后的产污强度 在下限与当前产污强度之间，那么 会采取清洁生产行为；否则就不会。这几行代码的意思 就是  有技术天花板的约束
        if self.pollution_generation_intensity_limit < strategy_chosen_pollution_generation_intensity < self.pollution_generation_intensity:
            self.clean_production_benefit = self.calc_strategy_clean_production_benefit(strategy_chosen_pollution_generation_intensity)
            self.pollution_generation_intensity = strategy_chosen_pollution_generation_intensity
            self.clean_production_improvement_boolean = True
            
      #      print(self.pollution_generation_intensity)
        else:
            self.clean_production_improvement_boolean = False
            
#如果采取行为后的去除率  在当前和上限之间，那么会采取末端治理 ；
        if self.end_removal_rate < strategy_chosen_end_removal_rate < self.end_removal_rate_limit:
            self.end_removal_rate = strategy_chosen_end_removal_rate
            self.end_removal_improvement_boolean = True
            self.end_removal_cost = self.calc_strategy_end_removal_cost(strategy_chosen_end_removal_rate)
        else:
            self.end_removal_improvement_boolean = False

#更新后的产污量与排放量  after

        pollution_generation_amount_after_update = self.calc_pollution_generation_amount(
            self.production,
            self.pollution_generation_intensity
        )
        pollution_final_amount_after_update = self.calc_pollution_final_amount(
            pollution_generation_amount_after_update,
            self.end_removal_rate
        )

#许可量盈余 用于后期政府给新建企业的  before 为 许可量（每期）
        self.pollution_allow_amount_surplus = pollution_final_amount_before_update - pollution_final_amount_after_update    #富余许可量 为采取清洁生产或末端治理 前后的差值    #0415  许可量-排放量    如果为正 就是达标后 富余的量 
        #print(strategy_chosen_name)        
        return strategy_chosen_name           #某行为的名称 S1.。。S12

    def generate_strategy_state_variables(self, strategy_tuple):               #tuple[0] 表示行为 比如S1（0，0,1）中的第一列
        production_value = self.production * self.price
# 不同产值 减产幅度 在输入表里 有区分（根据决策树）。目前考虑成统一的了0.03-0.08
        if production_value <= self.production_value_threshold1:
            if strategy_tuple[1] < 0:
                production_adjustment_step = self.scenario.production_adjustment_step1_decrease* np.random.uniform(0.9,1.3)
            else:
                production_adjustment_step = self.scenario.production_adjustment_step1_increase* np.random.uniform(0.9,1.3)
            clean_production_adjust_step = self.scenario.clean_production_adjustment_step1* np.random.uniform(0.9,1.3)
            end_removal_adjustment_step = self.scenario.end_removal_adjustment_step1

        elif self.production_value_threshold1 < production_value < self.production_value_threshold2:
            if strategy_tuple[1] < 0:
                production_adjustment_step = self.scenario.production_adjustment_step2_decrease* np.random.uniform(0.9,1.3)
            else:
                production_adjustment_step = self.scenario.production_adjustment_step2_increase* np.random.uniform(0.9,1.3)
            clean_production_adjust_step = self.scenario.clean_production_adjustment_step2* np.random.uniform(0.9,1.3)
            end_removal_adjustment_step = self.scenario.end_removal_adjustment_step2

        else:
            if strategy_tuple[1] < 0:
                production_adjustment_step = self.scenario.production_adjustment_step3_decrease* np.random.uniform(0.9,1.3)   #赋值成0.06*0.9--0.06*1.3 之间的随机数  step输入表是0.06
            else:
                production_adjustment_step = self.scenario.production_adjustment_step3_increase* np.random.uniform(0.9,1.3)
            clean_production_adjust_step = self.scenario.clean_production_adjustment_step3* np.random.uniform(0.9,1.3)
            end_removal_adjustment_step = self.scenario.end_removal_adjustment_step3



        '''  

        else:
            if strategy_tuple[1] < 0:
                production_adjustment_step = self.scenario.production_adjustment_step3_decrease* np.random.uniform(0.9,1.3)   #赋值成0.06*0.9--0.06*1.3 之间的随机数  step输入表是0.06
            else:
                production_adjustment_step = self.scenario.production_adjustment_step3_increase* np.random.uniform(0.9,1.3)
            clean_production_adjust_step = self.scenario.clean_production_adjustment_step3* np.random.uniform(0.9,1.3)
            end_removal_adjustment_step = self.scenario.end_removal_adjustment_step3
        '''


#产污强度 产值和 去除率调整      例如 采取末端治理后 排放量下降step=0.06  或0.03-0.08的随机数  ；由于所有行为step一样  意味着 减产 清洁 和末端的 减排量是一样的  但对应的利润 减排成本不同 
        pollution_generation_intensity = self.pollution_generation_intensity * (1 + strategy_tuple[0] * clean_production_adjust_step)
        if strategy_tuple[1] == 0:
            self.production_adjustment_action = 0
        elif strategy_tuple[1] < 0:
            self.production_adjustment_action = 1        #1  减产
        else:
            self.production_adjustment_action = 2        #2  增产
        production = self.production * (1 + strategy_tuple[1] * production_adjustment_step)
#        end_removal_rate = self.end_removal_rate * (1 + strategy_tuple[2] * end_removal_adjustment_step)
        end_removal_rate = self.end_removal_rate * (1 - strategy_tuple[2] * end_removal_adjustment_step)+ strategy_tuple[2]* end_removal_adjustment_step   #  处理后 剩余的排放量 是原来的95%   #0328 荣易指出问题（去除率不能直接成（1+step）
        if production < self.production:                  #gml0511添加 可在输出表中显示 增产 减产 不变的企业个数
            self.production_decrease_boolean = True
        else:
            self.production_decrease_boolean = False
        
        if production > self.production:
            self.production_increase_boolean = True
        else:
            self.production_increase_boolean = False

        
        if production == self.production:
            self.production_maintain_boolean = True
        else:
            self.production_maintain_boolean = False
#        print(self.production_maintain_boolean)    
        
#        print(clean_production_adjust_step)  #0503   是随机数啦 见#390
#        print(self.scenario.clean_production_adjustment_step3)
        return [pollution_generation_intensity, production, end_removal_rate]


#清洁生产 回报   #0504 单位投资 和单位收益的正态分布 随机采
    def calc_strategy_clean_production_benefit(self, strategy_pollution_generation_intensity):
        intensity_diff = self.pollution_generation_intensity - strategy_pollution_generation_intensity    #gml0401  这个self.pollution应该是上一期的吧？结合#342  应该是;   strategy_pollution_generation_intensity为什么可以表示采取行为后的产污强度？（问）
        
        #total_invest_cost = intensity_diff * self.clean_production_invest_cost_param
        total_invest_cost = intensity_diff * self.production * self.clean_production_invest_cost_param * np.random.normal(1, 0.3)    #gml0401 改为 *产值 表示 产污量的变化  #0504  加了正态区间 随机采
        #annual_benefit = intensity_diff * self.clean_production_benefit_param
        annual_benefit = intensity_diff * self.production * self.clean_production_benefit_param * np.random.normal(1, 0.3)          #gml0502 改为 *产值 表示 产污量的变化  即每减少1吨 得到的收益   #117行写参数 呈正态分布随机采样
        annual_factor = ((1 + self.scenario.return_rate) ** self.scenario.clean_production_lifetime - 1)/\
                        (self.scenario.return_rate * (1 + self.scenario.return_rate) ** self.scenario.clean_production_lifetime)
        clean_production_benefit = annual_benefit * annual_factor - total_invest_cost
                                                                               #  ！没问题 要考虑贴现的 贴现完 是正的几百万 ，说明清洁生产既 挣钱又减污；可以与“增产+末端”、“增产+清洁”等行为 比较风险与利润
#        clean_production_benefit = annual_benefit  - total_invest_cost        #若不考虑收益贴现  收益在  负几百-2000万左右   
#        print(clean_production_benefit)
        return clean_production_benefit

#末端治理成本
    def calc_strategy_end_removal_cost(self, strategy_end_removal_rate):   #为什么这里直接出来一个strategy_end_removal_rate 代表采取末端治理行为后的去除率  前文没有这个变量啊！0502对 定义了这个变量
        '''
        0502 代码计算不考虑invest成本了 输入表两个参数都变成0 
        '''  
        #运行成本 参考超然师兄 P44 和 P66  不同行业的截距项  造纸为-0.3 纺织为 -0.72 食品为-1.5

        operation_cost = np.e ** (-0.3+ 3.33 + 0.44 *(np.log(self.pollution_generation_amount)) -0.51* (1 - strategy_end_removal_rate))         #以此为准 不同行业的截距数要变  不同行业的截距项  造纸为-0.3 纺织为 -0.72 食品为-1.5
    
#        end_removal_cost = invest_cost  + operation_cost
        end_removal_cost =  operation_cost
#        print(invest_cost) 
#        print(operation_cost)   #200 万左右
        
#        print(end_removal_cost)                     #几百万        
        return end_removal_cost

    def evaluate_strategy(self, strategy_state_variables, t, government_check_probability):       #20220502 GML 增加了t 对应许可量 pollution_allow
        pollution_generation_intensity = strategy_state_variables[0]
        production = strategy_state_variables[1]
        end_removal_rate = strategy_state_variables[2]
        clean_production_benefit = self.calc_strategy_clean_production_benefit(pollution_generation_intensity)
        end_removal_cost = self.calc_strategy_end_removal_cost(end_removal_rate)

        production_value = production * self.price
        labor_cost = self.calc_labor_cost(production)
        capital_cost = self.calc_capital_cost(production)
        intermediate_input_cost = self.calc_intermediate_input_cost(production)
        income_tax = self.calc_income_tax(production)

        waste_water_amount = self.calc_waste_water_amount(production)
        pollution_generation_amount = self.calc_pollution_generation_amount(production, pollution_generation_intensity)
#        pollution_allow_amount = self.calc_pollution_allow_amount(waste_water_amount)
        pollution_allow_amount = self.calc_pollution_allow_amount(t)             #20220502 GML  改允许排放量
        pollution_final_amount = self.calc_pollution_final_amount(pollution_generation_amount, end_removal_rate)
        environment_tax_or_waster_water_fee = self.calc_environment_tax_or_waster_water_fee(waste_water_amount,
                                                                                            pollution_final_amount)
 #       excess_pollution_punishment = self.calc_excess_pollution_punishment(pollution_final_amount,
 #                                                                          pollution_allow_amount) * government_check_probability
        excess_pollution_punishment = self.calc_excess_pollution_punishment(pollution_final_amount,
                                                                            pollution_allow_amount)      #20220502 GML改 如果随机数大于概率就是要罚款的
        
        strategy_profit = production_value - labor_cost - capital_cost - intermediate_input_cost - \
                          income_tax - environment_tax_or_waster_water_fee + clean_production_benefit - \
                          end_removal_cost                           #0906 去掉excess_punishment
#gml 20220504  利润 也变为0-1之间的数 Sigmoid形式（下面错： 由于profit很大 导致Sigmoid趋近于1  所有 用profit/profitmax代替  体现在def calc_strategy_utility 后面
#        strategy_profit = 1/(1 + np.e**((-1)*strategy_profit))      #gml 20220503  利润 也变为0-1之间的数   为了 让下面#515 strategy_utility ln后面为正值
#        strategy_risk = np.e ** (self.scenario.risk_preference_param * (pollution_final_amount - pollution_allow_amount))    #gml20220328   风险函数
#gml0504   风险函数 sigmoid形式0~ 1  (在def calc_strategy_utility 定义 例如效用函数 = 利润最大 or 风险最小 or组合)
        strategy_risk = (pollution_final_amount - pollution_allow_amount)/pollution_allow_amount   #0504gml  风险=（排放-许可）/许可
        #strategy_risk = excess_pollution_punishment* self.scenario.government_check_probability
        if pollution_allow_amount > pollution_final_amount:   #20220328 荣易帮改  许可量大于排放量 达标   选利润最大化#415    #这个在讨论 达标 超标 选择不同决策规则时用 
 #       if pollution_allow_amount >= pollution_final_amount*(1+ self.scenario.production_adjustment_step1_increase):   #20220405 gml修改  如果比排放量*（1+step）大，那一定比Emission大，肯定达标，用利润最大化，意思是 达标后 要是选择增产超出了许可量 那就不能选利润最大化（即增产）
            larger = 1

        else:
            larger = 0

        return (strategy_profit, strategy_risk, larger)
        

    def calc_strategy_utility(self,strategy_state_variables,t, government_check_probability, strategy_profit, profit_max, profit_min, strategy_risk, risk_min, risk_max, larger,strategy_name):        #gml20220203   #0407  璐瑶补充strategy_name    #20220502 为了对应allow（t），增加了t
#20220504 增加  利润和风险 变为0-1之间的数 
        strategy_profit = 1/(1 + np.e**((-1)*strategy_profit/profit_max))  

        strategy_risk   = 1/(1 + np.e**((-1)*risk_min/strategy_risk))   

        '''
20220503 以下 区分达标 不达标 讨论各种决策组合情景
《一》《二》《三》 不论达标超标 ，都用一样的函数 （利润最大、风险最小、效用最大） ：让 IF和ELSE 后面一致即可 ，用哪个 决策机制 ，就把哪个#去掉  。注意：利润、风险、效用的函数表达不一样 ；beta只体现效用 ；
《四》 超标时，风险最小；达标时，利润最大
《五》 超标时，风险最小；达标时，效用最大
《六》 超标时，效用最大；达标时，利润最大
不存在：超标时，利润最大，达标时风险最小的情况
        '''        
        pollution_generation_intensity = strategy_state_variables[0]
        production = strategy_state_variables[1]
        end_removal_rate = strategy_state_variables[2]
      
        pollution_generation_amount = self.calc_pollution_generation_amount(production, pollution_generation_intensity)
        pollution_allow_amount = self.pollution_allow_amount  #20220522  璐瑶说 下面哪行带(t)  的 就表示许可量=初始许可量*(1-anuual)^t; 而这一行这样写 意味着 当许可量等于 当前更新的许可量（即surplus大于0后，许可量=before_amount.
        #pollution_allow_amount = self.calc_pollution_allow_amount(t)             #20220502 GML  改允许排放量  意味跟initial有关
        pollution_final_amount = self.calc_pollution_final_amount(pollution_generation_amount, end_removal_rate)
        
        #utility_beta = self.scenario.utility_function_param +np.random.normal(0,0.1)   #20220829 gml 为了beta 在区间取值 用utility_beta 替换 self.scenario.utility_function_param
        
        if not larger:  # 不达标  
        
            strategy_utility = np.e ** ((np.log(strategy_profit) )* (self.scenario.utility_function_param +np.random.normal(0,0.1)) +\
                                        ( np.log(strategy_risk))    * (1 - (self.scenario.utility_function_param +np.random.normal(0,0.1))))     #lnx X要大于等于0 
                                                                                                                    #为了让β取区间 比如在情景表中贝塔为0.5，那就是在0.5-0.6中间随机取值

        #    strategy_utility = strategy_profit      #print出都在0.65-0.73之间 没问题
         #   strategy_utility = (strategy_profit-profit_min)/(profit_max-profit_min)
            
          #  strategy_utility = strategy_risk         #print出都在0-1之间 有的0.02 有的0.97没问题

                                    
        else:   # 达标
          #  strategy_utility = strategy_profit
          #  strategy_utility = (strategy_profit-profit_min)/(profit_max-profit_min)
          #  strategy_utility = strategy_risk
        #    print(strategy_utility)
            
            strategy_utility = 0               #gml0506 以下四行 为新添加
            if pollution_allow_amount >= pollution_final_amount*(1+self.scenario.production_adjustment_step1_increase):     #gml20220407 increase_step 要调成大于0.05的  如 0.06 为啥呢
                if strategy_name == "S5":
                    strategy_utility = 1   #增产S5 的效用是 1  其他行为是0  选效用最大的 
            if pollution_final_amount < pollution_allow_amount < pollution_final_amount*(1+ self.scenario.production_adjustment_step1_increase):
                if strategy_name == "S1":
                    strategy_utility = 1  
            

 
# 组合效用函数 表达  道格拉斯函数 取对数  u=e^ (beta*ln(profit归一化)+(1-beta)*ln(r归一化))  但归一化 后 有的值为0 因此参考贺舟 π=π/π_max   r=r_min/r  （上面已经处理 利润和风险）
#        strategy_utility = np.e ** ((np.log(strategy_profit) )* self.scenario.utility_function_param +\
#                                        ( np.log(strategy_risk))    * (1 - self.scenario.utility_function_param))     #lnx X要大于等于0
#        print(strategy_utility)
#        return strategy_utility
        return max(strategy_utility,0.00000001)       #20220530  璐瑶0522帮改
    
    
    
    
    def calc_strategy_utility_dict(self, clean_production_learn_boolean,
                                   learned_pollution_generation_intensity,
                                   t,
                                   government_check_probability):                     #20220502gml  增加了t 对应初始许可量 逐年减
        strategy_evaluation_dict = {}
        strategy_profit_list = []
        strategy_risk_list = []
        for strategy_name, strategy_tuple in self.strategy_dict.items():
            if clean_production_learn_boolean and (strategy_name in ["S1", "S2", "S3", "S4", "S5", "S6"]):             #如果在 S1-S6  产污强度不下降的情况下  技术学习  那没问题
                pass
            else:                                                                                                     #否则  S7-S12  已采取清洁生产的情况下 再技术学习
                strategy_state_variables = self.generate_strategy_state_variables(strategy_tuple)
                if clean_production_learn_boolean:                                                                    #如果技术学习    
                    strategy_state_variables[0] = learned_pollution_generation_intensity                              # 行为中第一列即【0】-1 就变成了学习后的产污强度 ？
                else:
                    pass
                strategy_evaluation = self.evaluate_strategy(strategy_state_variables,t,government_check_probability)   #增加t

                strategy_evaluation_dict[strategy_name] = strategy_evaluation
                strategy_profit_list.append(strategy_evaluation[0])
                strategy_risk_list.append(strategy_evaluation[1])               # strategy_evaluation_dict = {} 中 有两列吗 分别是 利润 和 风险  对应#563  没错

        profit_max = max(strategy_profit_list)
        profit_min = min(strategy_profit_list)  #gml20220203
        risk_min = min(strategy_risk_list)
        risk_max = max(strategy_risk_list)  #gml20220203
        strategy_utility_dict = {}
        for strategy_name, strategy_evaluation_tuple in strategy_evaluation_dict.items():
            strategy_profit = strategy_evaluation_tuple[0]
            strategy_risk = strategy_evaluation_tuple[1]
            #strategy_utility = self.calc_strategy_utility(strategy_profit, profit_max, strategy_risk, risk_min)
#            strategy_utility = self.calc_strategy_utility(strategy_profit, profit_max, profit_min, strategy_risk, risk_min, risk_max)
#468行 定义了 calc_strategy_utility
#            strategy_utility = self.calc_strategy_utility(strategy_state_variables, government_check_probability,strategy_profit, profit_max, profit_min, strategy_risk, risk_min, risk_max, strategy_evaluation[2],strategy_name)  #gml20220203    #20220328荣易帮改  strategy_evaluation[2]表示larger  
#                                                                                                                                                                                                                                    #20220407 璐瑶帮做  补充strategy_state_variables, government_check_probability，strategy_name
            strategy_utility = self.calc_strategy_utility(strategy_state_variables, t, government_check_probability,strategy_profit, profit_max, profit_min, strategy_risk, risk_min, risk_max, strategy_evaluation[2],strategy_name) 



            strategy_utility_dict[strategy_name] = strategy_utility                #strategy_name 就是S1。。。S12  确定

        return strategy_utility_dict                                               #strategy_utility_dict[strategy_name]即 S1等每个行为组合的效用值  不同于#592 

    def strategy_choice(self, strategy_utility_dict, strategy_counter_dict):         # 定义了strategy_utility_dict 和 strategy_counter_dict   对应#300行         strategy_counter_dict 见environment.py

        strategy_utility_sum = 0
        # print(strategy_utility_dict.values())
##启动模仿机制（自我决策概率*周围人）
        if self.scenario.with_learning == 1:                     #在scenario表在该1 或 0
            
            for strategy_name in strategy_utility_dict.keys():
                strategy_utility_dict[strategy_name] = strategy_utility_dict[strategy_name] * \
                                                       strategy_counter_dict[strategy_name]     #每个行为的效用=  自我strategy_utility(#570行)  乘以  counter 
            '''
#20220602GML下面代码表示 假设模仿概率之余周围人有关  ；上面代码表示受自我决策和周围人影响;0603  这样做不合理 不知道最初始的行为 得到的结果是 不同行业都采取的末端和维持不变   放弃这个 ！关注上面的代码 在自我决策的基础上 乘周围人影响  observation=1
            for strategy_name in strategy_utility_dict.keys():
                strategy_utility_dict[strategy_name] = strategy_counter_dict[strategy_name]
            '''

                                                       
            for strategy_name in strategy_utility_dict.keys():
                strategy_utility_sum += strategy_utility_dict[strategy_name]
            for strategy_name in strategy_utility_dict.keys():
                strategy_utility_dict[strategy_name] = strategy_utility_dict[strategy_name] / strategy_utility_sum            #此处的strategy_utility_dict[strategy_name] 为概率  每个行为的效用（自我*周围）/总效用
#不启动模仿机制（自我决策）
        else:                                                    #不启动学习机制
            for strategy_name in strategy_utility_dict.keys():
                strategy_utility_sum += strategy_utility_dict[strategy_name]
            for strategy_name in strategy_utility_dict.keys():
                strategy_utility_dict[strategy_name] = strategy_utility_dict[strategy_name] / strategy_utility_sum             #strategy_utility_dict[strategy_name] 更新为 概率 即 每个行为的效用/12个行为的总效用
#        if self.id == 50:                         #0328 荣易帮写 为了看企业id=1  运行20年的概率
#            print(strategy_utility_dict)
#            input()
#        rand = np.random.uniform(0, 1)
        
        strategy_chosen = ''                                  #20220329 璐瑶帮写  每家企业选择 概率最大的那一个行为   即无论run多少次 都选那个行为  “百度：python找出字典中value最大值的几种方法”
        max_strategy_utility_p = 0
        for si in strategy_utility_dict.keys():
            this_strategy_utility = strategy_utility_dict[si]
            if max_strategy_utility_p<this_strategy_utility:     #如果得到的概率  比最大概率大  那就替换这个最大概率
                max_strategy_utility_p = this_strategy_utility
                strategy_chosen = si
        '''
        
        rand = np.random.uniform(0, 1)
        strategy_utility_accumulated = 0
        strategy_chosen = ''
        for strategy_name in strategy_utility_dict.keys():
            strategy_utility_accumulated += strategy_utility_dict[strategy_name]                   #此处的 strategy_utility_accumulated 和 strategy_utility_dict[strategy_name]   都是概率 所有行为的概率加和为1  蒙特卡洛
            if strategy_utility_accumulated >= rand:
                strategy_chosen = strategy_name
                break
        
#        if self.id == 7:
             print(strategy_chosen)     #0328 荣易帮写 为了看企业id=1  每年选择的行为   看所有行为的选择的话 就把if隐掉 print与上面对齐
        '''
        return strategy_chosen              #表示行为编号 S1。。。S12  #613  与strategy_name 一个意思











