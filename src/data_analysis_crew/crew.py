import os
from typing import Any, Dict, List, Literal, Optional, Tuple
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai_tools import CodeInterpreterTool, CSVSearchTool, DirectoryReadTool, FileReadTool, FileWriterTool

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


csv_source = CSVKnowledgeSource(
    file_paths=["data.csv"]
)

#_______PYDANTIC OUTPUT_______
class LoadDataOutput(BaseModel):
    dataset_path: str
    shape: Tuple[int, int]
    columns: List[str]
    dtype_map: Dict[str, str]
    missing_values: Dict[str, int]


class CleanedDataOutput(BaseModel):
    cleaned_path: str
    final_features: List[str]
    categorical_features: List[str]
    numeric_features: List[str]
    dropped_columns: List[str]
    imputation_summary: Dict[str, str]


class ExplorationOutput(BaseModel):
    plot_paths: List[str]
    top_correlations: List[Tuple[str, float]]
    anomalies: List[str]
    statistical_notes: str


class FeatureSelectionOutput(BaseModel):
    problem_type: Literal["classification", "regression"] = Field(
        description="classification or regression"
    )
    top_features: List[str] = Field(
        description="List of selected top features"
    )
    reasoning: str = Field(
        description="Why these features and why this problem type"
    )


class ModelOutput(BaseModel):
    model_type: str = Field(description="The type of model used, e.g., 'RandomForestClassifier'")
    target: str = Field(description="The target variable the model is predicting")
    feature_importance_path: str = Field(description="Path to the feature importance plot image file")
    metrics: Dict[str, float] = Field(description="Model performance metrics such as accuracy, F1, R², or MSE")
    confusion_matrix_path: Optional[str] = Field(
        default=None,
        description="Path to the confusion matrix plot (if classification task)"
    )
    plain_summary: str = Field(
        description="Plain-language executive summary: 2–3 sentences summarizing model results and insights"
    )

#_______CREW_______
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
    file_writer = FileWriterTool()

    ##___AGENTS___
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
            knowledge_sources=csv_source
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
            tools=[
                self.code_interpreter,
                self.csv_search
            ],
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
            tools=[
                self.code_interpreter,
                self.file_writer
            ],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe"
        )

    @agent
    def data_project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["data_project_manager"],
            llm=MANAGER_MODEL,
            verbose=True,
            allow_code_execution=True
        )

    ##___TASKS___
    @task
    def load_data(self) -> Task:
        return Task(
            config=self.tasks_config["load_data"],
            output_pydantic=LoadDataOutput
        )

    @task
    def clean_data(self) -> Task:
        return Task(
            config=self.tasks_config["clean_data"],
            context=[self.load_data],
            output_pydantic=CleanedDataOutput
        )

    @task
    def explore_data(self) -> Task:
        return Task(
            config=self.tasks_config["explore_data"],
            context=[self.clean_data],
            output_pydantic=ExplorationOutput
        )

    @task
    def select_features(self) -> Task:
        return Task(
            config=self.tasks_config["select_features"],
            context=[self.explore_data],
            output_pydantic=FeatureSelectionOutput
        )

    @task
    def build_predictive_model(self) -> Task:
        return Task(
            config=self.tasks_config["build_predictive_model"],
            context=[self.clean_data, self.select_features],
            output_pydantic=ModelOutput
        )

    @task
    def summarize_findings(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_findings"],
            context=[self.build_predictive_model]
        )

    ##___CREW___
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
