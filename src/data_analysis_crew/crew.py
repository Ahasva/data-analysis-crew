"""
Orchestration of agentic AI crew
"""
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional
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
from data_analysis_crew.tools import (
    build_predictive_model,
    explore_data,
    launch_dashboard,
    clean_data_tool,
    #load_or_clean,
    install_dependency
)
from data_analysis_crew.schemas import (
    LoadDataOutput,
    CleanedDataOutput,
    ExplorationOutput,
    FeatureSelectionOutput,
    ModelOutput,
    SummaryReportOutput
)
# ─── import centralized paths ─────────────────────────────────────────────────
from data_analysis_crew.settings import FILE_NAME, REL_PATH_DATA


# ======= LOAD ENVIRONMENT VARIABLES =======
load_dotenv()

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "anthropic/claude-3-haiku-20240307")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini/gemini-2.0-flash")

default_tag = os.getenv("CREWAI_DOCKER_IMAGE_TAG", "code-interpreter:latest")


# ======= LLM CONFIGURATIONS =======
AGENT_LLMS = {
    "data_engineer": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.2
    ),
    "data_analyst": LLM(
        model=ANTHROPIC_MODEL_NAME,
        temperature=0.25
    ),
    "model_builder": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.2
    ),
    "insight_reporter": LLM(
        model=ANTHROPIC_MODEL_NAME,
        temperature=0.3
    ),
    "quality_checker": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.1
    ),
    "data_project_manager": LLM(
        model=OPENAI_MODEL_NAME,
        temperature=0.2
    ),
}

FUNCTION_CALLING_LLM = LLM(
    model=OPENAI_MODEL_NAME,
    temperature=0.0
)

print("\nUsing Models:")
for role, llm in AGENT_LLMS.items():
    print(f"\t{role}:\n\t\t{llm.model}\t(temp={llm.temperature})")
print(f"\t{FUNCTION_CALLING_LLM}:\n\t\t{FUNCTION_CALLING_LLM.model}\t(temp={FUNCTION_CALLING_LLM.temperature})")


# ======= GLOBAL PATHS & TOOLS =======
print(f"\n💾\tUsed data:\t{FILE_NAME}\n🧭\tRelative path:\t{str(REL_PATH_DATA)}\n")

# ─── Knowledge Source ────────────────────────────────────────────────────────────────────────
csv_source = CSVKnowledgeSource(file_paths=[FILE_NAME])

# ─── Tools ───────────────────────────────────────────────────────────────────────────────────
csv_search = CSVSearchTool(csv=str(REL_PATH_DATA))
directory_reader = DirectoryReadTool()
file_reader = FileReadTool()
file_writer = FileWriterTool()


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
    _external_step_callback: Optional[Callable[[Any], None]] = None

    # ─── @before_kickoff ──────────────────────────────────────────────────────────────────────
    @before_kickoff
    def prepare_inputs(self, inputs):
        """Set up Docker + ensure all runtime variables like `file_name` are clean"""

        # ─── Docker Setup ─────────────────────────────────────────────────────────────────────
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

            hash_source = (BUILD_DIR / "requirements.txt").read_bytes()
            requirements_hash = hashlib.md5(hash_source).hexdigest()
            self.image_tag = f"crewai-env:{requirements_hash}"
            self.code_interpreter = CodeInterpreterTool(default_image_tag=self.image_tag)
            print(f"🧩 Using Docker image tag: {self.image_tag}")

            existing_images = subprocess.run(
                ["docker", "images", "-q", self.image_tag],
                capture_output=True,
                text=True,
                check=True
            )

            if existing_images.stdout.strip():
                print(f"✅ Docker image {self.image_tag} already exists. Skipping build.")
            else:
                print(f"🔨 Docker image '{self.image_tag}' not found. Building now...")
                subprocess.run(["docker", "build", "-t",
                                self.image_tag, str(BUILD_DIR)],
                                check=True)
                print("✅ Docker image built successfully.")

        else:
            print("⚠️ No 'available_libraries' input provided — skipping Docker setup.")

        # ─── Input Cleaning ───────────────────────────────────────────────────────────────────
        if "file_name" not in inputs and "raw_path" in inputs:
            inputs["file_name"] = Path(inputs["raw_path"]).stem
            print(f"🧠 Inferred file_name from raw_path: {inputs['file_name']}")

        if "file_name" in inputs:
            inputs["file_name"] = Path(inputs["file_name"]).stem
            print(f"🧼 Cleaned file_name: {inputs['file_name']}")

        # ─── Debug ────────────────────────────────────────────────────────────────────────────
        print("\n🚀 Final Inputs for Crew:")
        for key, value in inputs.items():
            print(f"   {key}: {value}")

        return inputs

    # ─── Agent Configuration ───────────────────────────────
    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_engineer"],
            llm=AGENT_LLMS["data_engineer"],
            tools=[
                tool for tool in [
                    csv_search,
                    clean_data_tool,
                    directory_reader,
                    file_writer,
                    self.code_interpreter,
                    install_dependency,
                    explore_data,
                    file_reader,
                    #load_or_clean
                ] if tool is not None
            ],
            function_calling_llm=FUNCTION_CALLING_LLM,
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
                    csv_search,
                    clean_data_tool,
                    directory_reader,
                    file_writer,
                    self.code_interpreter,
                    install_dependency,
                    explore_data
                ] if tool is not None
            ],
            function_calling_llm=FUNCTION_CALLING_LLM,
            max_execution_time=300,
            memory=True,
            verbose=True,
            allow_delegation=False,
            allow_code_execution=True,
            code_execution_mode="safe",
            inject_date=True,
            reasoning=True,
            max_reasoning_attempts=2,
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
                    explore_data,
                    csv_search,
                    directory_reader,
                    build_predictive_model
                ] if tool is not None
            ],
            function_calling_llm=FUNCTION_CALLING_LLM,
            allow_code_execution=True,
            code_execution_mode="safe",
            memory=True,
            verbose=True,
            inject_date=True,
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
                    directory_reader,
                    self.code_interpreter,
                    install_dependency,
                    file_writer,
                    launch_dashboard
                ] if tool is not None
            ],
            function_calling_llm=FUNCTION_CALLING_LLM,
            allow_code_execution=True,
            code_execution_mode="safe",
            memory=True,
            verbose=True,
            inject_date=True,
            reasoning=True,
            max_execution_time=180,
            max_reasoning_attempts=2
        )

    @agent
    def quality_checker(self) -> Agent:
        return Agent(
            config=self.agents_config["quality_checker"],
            llm=AGENT_LLMS["quality_checker"],
            tools=[
                directory_reader,
                file_reader,
                file_writer
            ],
            memory=True,
            verbose=True,
            inject_date=True,
            reasoning=True,
            max_reasoning_attempts=1,
            allow_delegation=False
        )

    @agent
    def data_project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["data_project_manager"],
            llm=AGENT_LLMS["data_project_manager"],
            #tools=[],
            allow_code_execution=False,
            code_execution_mode="safe",
            memory=True,
            verbose=True,
            inject_date=True,
            reasoning=True,
            max_reasoning_attempts=1,
            allow_delegation=True
        )

    # ─── Task Configuration ────────────────────────────────────────────────────────────────
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
            output_pydantic=ModelOutput,
            #output_json=ModelOutput,
            output_file="output/model-report.json"
        )

    @task
    def summarize_findings(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_findings"],
            context=[self.build_predictive_model()],
            output_pydantic=SummaryReportOutput
        )
    
    @task
    def validate_summary(self) -> Task:
        return Task(
            config=self.tasks_config["validate_summary"],
            context=[self.summarize_findings()]
        )

    @task
    def launch_dashboard(self) -> Task:
        return Task(
            config=self.tasks_config["launch_dashboard"],
            context=[self.validate_summary()]
        )

    # ─── Delegation Activity ───────────────────────────────────────────────────────────────
    def track_collaboration(self, output):
        task_description = getattr(output, "task_description", "Unknown task")
        agent_name = getattr(output, "agent_name", "Unknown agent")

        print(f"\n📡 Step Callback → Task: {task_description[:80]}... | Agent: {agent_name}")
        print(f"🔍 Raw Output:\n{str(output)[:1000]}")

        if "Delegate work to coworker" in str(output):
            print("🤝 Delegation occurred!")

        if "Ask question to coworker" in str(output):
            print("❓ Question asked!")

        if self._external_step_callback:
            self._external_step_callback(output)

    # ─── Crew Configuration ───────────────────────────────
    @crew
    def crew(self, **kwargs) -> Crew:
        # External callback
        self._external_step_callback = kwargs.pop("on_step_callback", None)

        non_manager_agents = [
            agent for agent in self.agents if agent != self.data_project_manager()
        ]
        return Crew(
            agents=non_manager_agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            memory=True,
            function_calling_llm=FUNCTION_CALLING_LLM,
            manager_agent=self.data_project_manager(),
            planning=True,
            output_log_file="output/crew_log.json",
            step_callback=self.track_collaboration,
            **kwargs
        )
