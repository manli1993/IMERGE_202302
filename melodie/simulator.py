
import abc
import os.path
import threading
import time
from multiprocessing import Pool
from typing import ClassVar, TYPE_CHECKING, Optional, List, Dict, Tuple, Callable, Union, Type
import logging

import pandas as pd

import melodie.visualization
from . import DB
from .agent import Agent

from .agent_list import AgentList

from .table_generator import TableGenerator
from .basic.exceptions import MelodieExceptions

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .environment import Environment
    from .model import Model
    from .scenario_manager import Scenario
    from .data_collector import DataCollector
    from .config import Config
    from .visualization import Visualizer
else:
    from .scenario_manager import Scenario
    from .config import Config
    from .db import create_db_conn


class Simulator(metaclass=abc.ABCMeta):
    def __init__(self):
        self.config: Optional[Config] = None
        self.server_thread: threading.Thread = None
        self.scenario_class: Optional[ClassVar['Scenario']] = None
        self.scenarios_dataframe: Optional[pd.DataFrame] = None
        self.agent_params_dataframe: Optional[pd.DataFrame] = None
        self.registered_dataframes: Optional[Dict[str, pd.DataFrame]] = {}
        self.scenarios: Optional[List['Scenario']] = None

    @abc.abstractmethod
    def register_scenario_dataframe(self) -> None:

        pass

    def register_static_dataframes(self) -> None:

        pass

    def register_generated_dataframes(self) -> None:

        pass

    def register_dataframe(self, table_name: str, data_frame: pd.DataFrame, data_types: dict = None) -> None:

        if data_types is None:
            data_types = {}
        DB.register_dtypes(table_name, data_types)
        create_db_conn(self.config).write_dataframe(table_name, data_frame, data_types=data_types,
                                                    if_exists="replace")
        self.registered_dataframes[table_name] = create_db_conn(self.config).read_dataframe(table_name)

    def load_dataframe(self, table_name: str, file_name: str, data_types: dict) -> None:

        _, ext = os.path.splitext(file_name)
        table: Optional[pd.DataFrame]
        assert table_name.isidentifier(), f"table_name `{table_name}` was not an identifier!"
        if ext in {'.xls', '.xlsx'}:
            file_path_abs = os.path.join(self.config.excel_source_folder, file_name)
            table = pd.read_excel(file_path_abs)
        else:
            raise NotImplemented(file_name)

        DB.register_dtypes(table_name, data_types)
        create_db_conn(self.config).write_dataframe(table_name, table, data_types=data_types,
                                                    if_exists="replace", )
        # self.registered_dataframes[table_name] = table
        self.registered_dataframes[table_name] = create_db_conn(self.config).read_dataframe(table_name)

    def get_registered_dataframe(self, table_name) -> pd.DataFrame:

        if table_name not in self.registered_dataframes:
            raise MelodieExceptions.Data.StaticTableNotRegistered(table_name, list(self.registered_dataframes.keys()))

        return self.registered_dataframes[table_name]

    def generate_scenarios_from_dataframe(self, df_name: str) -> List['Scenario']:

        self.scenarios_dataframe = self.get_registered_dataframe(df_name)
        assert self.scenarios_dataframe is not None
        assert self.scenario_class is not None
        table = self.scenarios_dataframe
        cols = [col for col in table.columns]
        scenarios: List[Scenario] = []
        for i in range(table.shape[0]):
            scenario = self.scenario_class()
            scenario.manager = self
            for col_name in cols:
                assert col_name in scenario.__dict__.keys(), f"col_name: '{col_name}', scenario: {scenario}"
                scenario.__dict__[col_name] = table.loc[i, col_name]
            scenarios.append(scenario)
        assert len(scenarios) != 0
        return scenarios

    def new_table_generator(self, table_name: str, rows_in_scenario: Union[int, Callable[[Scenario], int]]):

        return TableGenerator(self, table_name, rows_in_scenario)

    def generate_scenarios(self) -> List['Scenario']:

        return self.generate_scenarios_from_dataframe('scenarios')

    def pre_run(self):

        create_db_conn(self.config).clear_database()
        self.register_scenario_dataframe()
        self.register_static_dataframes()
        self.register_generated_dataframes()

        self.scenarios = self.generate_scenarios()
        assert self.scenarios is not None

    def run_model(self, config, scenario, run_id, model_class: ClassVar['Model'], visualizer=None):
        """

        :return: 
        """
        logger.info(f'Running {run_id + 1} times in scenario {scenario.id}.')
        t0 = time.time()
        model = model_class(config,
                            scenario,
                            run_id_in_scenario=run_id,
                            visualizer=visualizer)

        model.setup()
        t1 = time.time()
        model.run()
        t2 = time.time()

        model_setup_time = t1 - t0
        model_run_time = t2 - t1
        if model.data_collector is not None:
            data_collect_time = model.data_collector._time_elapsed
        else:
            data_collect_time = 0.0
        model_run_time -= data_collect_time
        info = (f'Running {run_id + 1} in scenario {scenario.id} completed with time elapsed(seconds):\n'
                f'    model-setup   \t {round(model_setup_time, 6)}\n'
                f'    model-run     \t {round(model_run_time, 6)}\n'
                f'    data-collect  \t {round(data_collect_time, 6)}\n')
        logger.info(info)

    def run(self,
            config: 'Config',
            model_class: ClassVar['Model'],
            scenario_class: ClassVar['Scenario'] = None
            ):
        """
        Main function for running model!
        """
        t0 = time.time()
        self.config = config
        self.scenario_class = scenario_class if scenario_class is not None else Scenario
        self.pre_run()

        logger.info('Loading scenarios and static tables...')
        t1 = time.time()
        for scenario_index, scenario in enumerate(self.scenarios):
            for run_id in range(scenario.number_of_run):
                self.run_model(config, scenario, run_id, model_class, )

            logger.info(f'{scenario_index + 1} of {len(self.scenarios)} scenarios has completed.')

        t2 = time.time()
        logger.info(f'Melodie completed all runs, time elapsed totally {t2 - t0}s, and {t2 - t1}s for running.')
