"""
Orchestration of agentic AI crew
"""
import os
from typing import Any, Dict, List, Literal, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai_tools import (
    CodeInterpreterTool,
    CSVSearchTool,
    DirectoryReadTool,
    FileReadTool,
    FileWriterTool
)
from pydantic import BaseModel, Field

from data_analysis_crew.tools import build_predictive_model, launch_dashboard, load_or_clean

# Load environment variables
load_dotenv()

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "anthropic/claude-3-haiku-20240307")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini/gemini-2.0-flash")

AGENT_MODEL = LLM(model=OPENAI_MODEL_NAME)
PLANNING_MODEL = LLM(model=OPENAI_MODEL_NAME)
MANAGER_MODEL = LLM(model="gpt-4o-mini", temperature=0.1)


# === LLM CONFIGURATIONS ===
AGENT_LLMS = {
    "data_engineer": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.2
        ),
    "data_analyst": LLM(
        model=ANTHROPIC_MODEL_NAME,
        temperature=0.4
        ),
    "model_builder": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.3
        ),
    "insight_reporter": LLM(
        model=ANTHROPIC_MODEL_NAME,
        temperature=0.6
        ),
    "data_project_manager": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.1
        ),
}

print("\nUsing Models:")
for role, llm in AGENT_LLMS.items():
    print(f"\t{role}:\n\t\t{llm.model}\t(temp={llm.temperature})")

# === GLOBAL PATHS & TOOLS ===
DATA_FOLDER = "knowledge"
FILE_NAME = "diabetes.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / DATA_FOLDER / FILE_NAME

# pass *relative* path to the crew / tools
RELATIVE_PATH = DATA_PATH.relative_to(PROJECT_ROOT)

#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#DATA_PATH = os.path.join(PROJECT_ROOT, DATA_FOLDER, FILE_NAME)

csv_source = CSVKnowledgeSource(file_paths=["diabetes.csv"])
csv_search = CSVSearchTool(file_path=RELATIVE_PATH)
code_interpreter = CodeInterpreterTool()
directory_reader = DirectoryReadTool()
file_reader = FileReadTool()
file_writer = FileWriterTool()

# === OUTPUT SCHEMAS ===
class LoadDataOutput(BaseModel):
    dataset_path: str = Field(description="Path to the loaded dataset")
    # ⚠️  Fix: add `items` so OpenAI function schema is valid
    shape: Tuple[int, int] = Field(
        description="Shape of the dataset (rows, columns)",
        json_schema_extra={
            "items": {"type": "integer"},  # <— required by OpenAI
            "minItems": 2,
            "maxItems": 2,
        },
    )
    columns: List[str] = Field(description="List of dataset columns")
    dtype_map: Optional[Dict[str, str]] = Field(default=None, description="Data type for each column")
    missing_values: Optional[Dict[str, int]] = Field(default=None, description="Count of missing values per column")

class CleanedDataOutput(BaseModel):
    cleaned_path: str = Field(description="Path to the cleaned dataset file")
    final_features: List[str] = Field(description="List of features retained after cleaning")
    categorical_features: List[str] = Field(description="List of identified categorical features")
    numeric_features: List[str] = Field(description="List of identified numerical features")
    dropped_columns: List[str] = Field(description="List of columns dropped during cleaning")
    imputation_summary: Optional[Dict[str, str]] = Field(default=None, description="Summary of how missing values were handled")

class FeatureCorrelation(BaseModel):
    feature: str = Field(description="Feature name")
    correlation: float = Field(description="Correlation coefficient with the target")

class ExplorationOutput(BaseModel):
    plot_paths: List[str] = Field(description="Paths to saved plots from data exploration")
    top_correlations: List[FeatureCorrelation] = Field(description="Top correlated features with the target")
    anomalies: List[str] = Field(description="List of potential data anomalies")
    statistical_notes: str = Field(description="Narrative summary of statistical insights")

class FeatureSelectionOutput(BaseModel):
    problem_type: Literal["classification", "regression"] = Field(description="Inferred ML problem type")
    top_features: List[str] = Field(description="List of selected top features")
    reasoning: str = Field(description="Explanation for selected features and problem type")

class ModelOutput(BaseModel):
    # ── core info ───────────────────────────────────────────────────────
    model_type: str = Field(
        description="Sklearn class name of the trained model (e.g. RandomForestClassifier, SVR)."
    )
    problem_type: Literal["classification", "regression"] = Field(
        description="Problem formulation inferred by the pipeline."
    )
    target: str = Field(
        description="Name of the target column that was predicted."
    )

    # ── evaluation ──────────────────────────────────────────────────────
    metrics: Dict[str, float] = Field(
        description="Primary evaluation metrics. "
                    "For classification: {'accuracy','f1'}; "
                    "for regression: {'r2','mse'}."
    )
    plain_summary: str = Field(
        description="Short one-liner summarising the metrics (shown on the dashboard card)."
    )

    # ── artefacts (optional because some models lack importances) ───────
    feature_importance_path: Optional[str] = Field(
        default=None,
        description="Relative path to feature-importance PNG "
                    "(may be None if not supported)."
    )
    secondary_plot_path: Optional[str] = Field(
        default=None,
        description="Relative path to the secondary plot: "
                    "confusion_matrix.png (classification) or residuals.png (regression)."
    )

    # ── legacy alias for confusion matrix ───────────────────────────────
    confusion_matrix_path: Optional[str] = Field(
        default=None,
        description="(DEPRECATED) alias of `secondary_plot_path` when "
                    "`problem_type=='classification'`."
    )

# === CREW ===
@CrewBase
class DataAnalysisCrew():
    agents_config: Dict[str, Dict[str, Any]]
    tasks_config: Dict[str, Dict[str, Any]]
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_engineer"],
            llm=AGENT_LLMS["data_engineer"],
            tools=[
                code_interpreter,
                file_reader,
                csv_search,
                load_or_clean
            ],
            max_execution_time=600,
            memory=True,
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            reasoning=True,
            max_reasoning_attempts=3,
            knowledge_sources=[csv_source]
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst"],
            llm=AGENT_LLMS["data_analyst"],
            tools=[code_interpreter],
            max_execution_time=600,
            memory=True,
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            reasoning=True,
            max_reasoning_attempts=3,
            allow_delegation=True,
            knowledge_sources=[csv_source]
        )

    @agent
    def model_builder(self) -> Agent:
        return Agent(
            config=self.agents_config["model_builder"],
            llm=AGENT_LLMS["model_builder"],
            tools=[
                code_interpreter,
                csv_search,
                build_predictive_model,
            ],
            allow_code_execution=True,
            code_execution_mode="safe",
            memory=True,
            verbose=True,
            reasoning=True,
            max_execution_time=600,
            allow_delegation=True,
            knowledge_sources=[csv_source]
        )

    @agent
    def insight_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["insight_reporter"],
            llm=AGENT_LLMS["insight_reporter"],
            tools=[
                code_interpreter,
                file_writer,
                launch_dashboard
            ],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe"
        )

    @agent
    def data_project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["data_project_manager"],
            llm=AGENT_LLMS["data_project_manager"],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_reasoning_attempts=1
        )

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
            context=[self.load_data()],
            output_pydantic=CleanedDataOutput
        )

    @task
    def explore_data(self) -> Task:
        return Task(
            config=self.tasks_config["explore_data"],
            context=[self.clean_data()],
            output_pydantic=ExplorationOutput
        )

    @task
    def select_features(self) -> Task:
        return Task(
            config=self.tasks_config["select_features"],
            context=[self.explore_data()],
            output_pydantic=FeatureSelectionOutput
        )

    @task
    def build_predictive_model(self) -> Task:
        return Task(
            config=self.tasks_config["build_predictive_model"],
            context=[self.clean_data(), self.select_features()],
            output_json=ModelOutput
        )

    @task
    def summarize_findings(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_findings"],
            context=[self.build_predictive_model()]
        )

    @task
    def launch_dashboard(self) -> Task:
        return Task(
            config=self.tasks_config["launch_dashboard"],
            context=[self.summarize_findings()]
        )
    @crew
    def crew(self) -> Crew:
        non_manager_agents = [
            agent for agent in self.agents if agent != self.data_project_manager()
        ]
        return Crew(
            agents=non_manager_agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            planning=True,
            manager_agent=self.data_project_manager(),
            verbose=True,
            memory=True
        )
