"""
Orchestration of agentic AI crew
"""
import os
import subprocess
from typing import Any, Dict, List, Literal, Optional, Tuple
from pathlib import Path
import hashlib
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
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

from data_analysis_crew.tools import (
    build_predictive_model,
    explore_data,
    launch_dashboard,
    load_or_clean,
    install_dependency
)

# ======= LOAD ENVIRONMENT VARIABLES =======
load_dotenv()

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "anthropic/claude-3-haiku-20240307")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini/gemini-2.0-flash")

# Currently not used
AGENT_MODEL = LLM(model=OPENAI_MODEL_NAME)
PLANNING_MODEL = LLM(model=OPENAI_MODEL_NAME)
MANAGER_MODEL = LLM(model="gpt-4o-mini", temperature=0.1)

default_tag = os.getenv("CREWAI_DOCKER_IMAGE_TAG", "code-interpreter:latest")

# ======= LLM CONFIGURATIONS =======
AGENT_LLMS = {
    "data_engineer": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.2
        ),
    "data_analyst": LLM(
        model=ANTHROPIC_MODEL_NAME,
        temperature=0.3
        ),
    "model_builder": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.2
        ),
    "insight_reporter": LLM(
        model=ANTHROPIC_MODEL_NAME,
        temperature=0.3
        ),
    "data_project_manager": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.1
        ),
}

print("\nUsing Models:")
for role, llm in AGENT_LLMS.items():
    print(f"\t{role}:\n\t\t{llm.model}\t(temp={llm.temperature})")

# ======= GLOBAL PATHS & TOOLS =======
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
directory_reader = DirectoryReadTool()
file_reader = FileReadTool()
file_writer = FileWriterTool()

# ======= OUTPUT SCHEMAS =======
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
    secondary_plot_paths: Optional[List[str]] = Field(
    default=None,
    description="List of paths to additional plots like ROC, PR, residuals, etc."
)

    # ── legacy alias for confusion matrix ───────────────────────────────
    confusion_matrix_path: Optional[str] = Field(
        default=None,
        description="(DEPRECATED) alias of `secondary_plot_path` when "
                    "`problem_type=='classification'`."
    )

# ======= CREW =======
@CrewBase
class DataAnalysisCrew():
    agents_config: Dict[str, Dict[str, Any]]
    tasks_config: Dict[str, Dict[str, Any]]
    agents: List[BaseAgent]
    tasks: List[Task]
    # Dynamically assigned during `@before_kickoff` after Docker image resolution
    code_interpreter: Optional[CodeInterpreterTool] = None
    # Will be assigned in @before_kickoff from hash(requirements_txt)
    image_tag: Optional[str] = None  

    # ── Docker environment ───────────────────────────────
    @before_kickoff
    def setup_docker_environment(self, inputs):
        print("🔧 Setting up Docker environment with required libraries...")

        BUILD_DIR = Path("build")
        BUILD_DIR.mkdir(parents=True, exist_ok=True)

        if "available_libraries" in inputs:
            requirements_txt = "\n".join(inputs["available_libraries"])
            (BUILD_DIR / "requirements.txt").write_text(requirements_txt, encoding="utf-8")
            print(f"📦 Created build/requirements.txt with: {inputs['available_libraries']}")

            dockerfile = """
FROM python:3.12-slim

COPY requirements.txt .

RUN pip install --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt
"""
            (BUILD_DIR / "Dockerfile").write_text(dockerfile, encoding="utf-8")
            print("📝 Created build/Dockerfile")

            # ✅ Tag with requirements hash
            requirements_hash = hashlib.md5(requirements_txt.encode()).hexdigest()
            self.image_tag = f"crewai-env:{requirements_hash}"
            self.code_interpreter = CodeInterpreterTool(default_image_tag=self.image_tag)
            print(f"🧩 Using Docker image tag: {self.image_tag}")

            # 🟡 Optional: Save hash to disk for future reference/debugging
            (BUILD_DIR / "requirements.hash").write_text(self.image_tag, encoding="utf-8")

            # ✅ Only build if needed
            existing_images = subprocess.run(
                ["docker", "images", "-q", self.image_tag],
                capture_output=True,
                text=True,
                check=True
            )

            if existing_images.stdout.strip():
                print(f"✅ Docker image {self.image_tag} already exists. Skipping build.")
                return
            else:
                print(f"🔨 Docker image '{self.image_tag}' not found. Building now...")
                try:
                    subprocess.run(["docker", "build", "-t",
                                    self.image_tag, str(BUILD_DIR)],
                                    check=True)
                    print("✅ Docker image built successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Docker build failed: {e}")
        else:
            print("⚠️ No 'available_libraries' input provided — skipping Docker setup.")

    # ── Agent Configuration ───────────────────────────────
    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_engineer"],
            llm=AGENT_LLMS["data_engineer"],
            tools=[
                tool for tool in [
                    self.code_interpreter,
                    install_dependency,
                    file_reader,
                    csv_search,
                    load_or_clean
                ] if tool is not None
            ],
            max_execution_time=180,
            memory=True,
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            reasoning=True,
            max_reasoning_attempts=2,
            knowledge_sources=[csv_source]
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst"],
            llm=AGENT_LLMS["data_analyst"],
            tools=[
                tool for tool in [
                    self.code_interpreter,
                    install_dependency,
                    explore_data
                ] if tool is not None
            ],
            max_execution_time=180,
            memory=True,
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            reasoning=True,
            max_reasoning_attempts=2,
            allow_delegation=False,
            knowledge_sources=[csv_source]
        )

    @agent
    def model_builder(self) -> Agent:
        return Agent(
            config=self.agents_config["model_builder"],
            llm=AGENT_LLMS["model_builder"],
            tools=[
                tool for tool in [
                    self.code_interpreter,
                    install_dependency,
                    csv_search,
                    build_predictive_model
                ] if tool is not None
            ],
            allow_code_execution=True,
            code_execution_mode="safe",
            memory=True,
            verbose=True,
            reasoning=True,
            max_execution_time=360,
            allow_delegation=False,
            knowledge_sources=[csv_source]
        )

    @agent
    def insight_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["insight_reporter"],
            llm=AGENT_LLMS["insight_reporter"],
            tools=[
                tool for tool in [
                    self.code_interpreter,
                    install_dependency,
                    file_writer,
                    launch_dashboard
                ] if tool is not None
            ],
            allow_code_execution=True,
            code_execution_mode="safe",
            memory=True,
            verbose=True,
            max_execution_time=180,
            max_reasoning_attempts=2
        )

    @agent
    def data_project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["data_project_manager"],
            llm=AGENT_LLMS["data_project_manager"],
            tools=[],
            allow_code_execution=False,
            code_execution_mode="safe",
            memory=True,
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=1,
            allow_delegation=True
        )

    # ── Task Configuration ───────────────────────────────
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
            context=[self.clean_data()], # alternative: self.clean_data()
            output_pydantic=ExplorationOutput,
            async_execution=False, # depends on `context=`-param
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
            output_json=ModelOutput,
            output_file="output/model-report.json"
        )

    @task
    def summarize_findings(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_findings"],
            context=[self.build_predictive_model()],
            output_file="output/final-insight-summary.md"
        )

    @task
    def launch_dashboard(self) -> Task:
        return Task(
            config=self.tasks_config["launch_dashboard"],
            context=[self.summarize_findings()],
            output_file=None
        )

    # ── Crew Configuration ───────────────────────────────
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
            memory=True,
            output_log_file="output/crew_log.json"
        )
