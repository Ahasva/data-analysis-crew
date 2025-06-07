import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import CodeInterpreterTool, CSVSearchTool, DirectoryReadTool, FileReadTool
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini/gemini-2.0-flash")

AGENT_MODEL = LLM(model=OPENAI_MODEL_NAME)
PLANNING_MODEL = LLM(model=OPENAI_MODEL_NAME)
MANAGER_MODEL = LLM(model="gpt-4o", temperature=0.1)

print(f"Using Models:\n\tAgent Model:\t{AGENT_MODEL.model}\
      \n\tPlanning Model:\t{PLANNING_MODEL.model}\
      \n\tManager Model:\t{MANAGER_MODEL.model}")


# Pydantic output from select_features task
class FeatureSelectionOutput(BaseModel):
    problem_type: str = Field(description="classification or regression")
    top_features: List[str] = Field(description="List of selected top features")
    reasoning: str = Field(description="Why these features and why this problem type")


@CrewBase
class DataAnalysisCrew():
    """DataAnalysisCrew crew"""

    agents_config: Dict[str, Dict[str, Any]]
    tasks_config: Dict[str, Dict[str, Any]]
    agents: List[BaseAgent]
    tasks: List[Task]

    code_interpreter = CodeInterpreterTool()
    csv_search = CSVSearchTool()
    directory_reader = DirectoryReadTool()
    file_reader = FileReadTool()

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_engineer'],
            llm=AGENT_MODEL,
            tools=[
                self.directory_reader,
                self.csv_search,
                self.code_interpreter,
                self.file_reader,
            ],
            max_execution_time=600,
            memory=True,
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            reasoning=True,
            max_reasoning_attempts=3,
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'],
            llm=AGENT_MODEL,
            tools=[self.code_interpreter],
            max_execution_time=600,
            memory=True,
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            reasoning=True,
            max_reasoning_attempts=3,
            allow_delegation=True
        )

    @agent
    def model_builder(self) -> Agent:
        return Agent(
            config=self.agents_config['model_builder'],
            llm=AGENT_MODEL,
            tools=[self.code_interpreter, self.csv_search],
            allow_code_execution=True,
            code_execution_mode="safe",
            memory=True,
            verbose=True,
            reasoning=True,
            max_execution_time=600,
            allow_delegation=True
        )

    @agent
    def insight_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['insight_reporter'],
            llm=AGENT_MODEL,
            tools=[self.code_interpreter],
            verbose=True
        )

    @agent
    def data_project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["data_project_manager"],
            llm=MANAGER_MODEL,
            verbose=True,
            allow_code_execution=True
        )

    #_______TASKS_______
    @task
    def load_data(self) -> Task:
        return Task(config=self.tasks_config['load_data'])

    @task
    def clean_data(self) -> Task:
        return Task(
            config=self.tasks_config['clean_data'],
            context=[self.load_data]
        )

    @task
    def explore_data(self) -> Task:
        return Task(
            config=self.tasks_config['explore_data'],
            context=[self.clean_data]
        )

    @task
    def select_features(self) -> Task:
        return Task(
            config=self.tasks_config['select_features'],
            context=[self.explore_data],
            output_pydantic=FeatureSelectionOutput
        )

    @task
    def build_predictive_model(self) -> Task:
        return Task(
            config=self.tasks_config['build_predictive_model'],
            context=[self.clean_data, self.select_features]
        )

    @task
    def summarize_findings(self) -> Task:
        return Task(
            config=self.tasks_config['summarize_findings'],
            context=[self.build_predictive_model]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DataAnalysisCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            planning=True,
            manager_agent=self.data_project_manager,
            #planning_llm=PLANNING_MODEL,
            verbose=True,
            memory=True
        )
