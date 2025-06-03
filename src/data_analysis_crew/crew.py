import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileReadTool, CodeInterpreterTool, DirectoryReadTool

load_dotenv()

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini/gemini-2.0-flash")
print(f"Loaded models:\n\tOpenAI:\t{OPENAI_MODEL_NAME}\n\tGoogle:\t{GOOGLE_MODEL_NAME}")

AGENT_MODEL = LLM(model=OPENAI_MODEL_NAME)
PLANNING_MODEL = LLM(model=OPENAI_MODEL_NAME)
print(f"\nUsed models:\n\tAGENT_MODEL:\t{AGENT_MODEL.model}\n\tPLANNING_MODEL:\t{PLANNING_MODEL.model}")

@CrewBase
class DataAnalysisCrew():
    """DataAnalysisCrew crew"""

    agents_config: Dict[str, Dict[str, Any]]
    tasks_config: Dict[str, Dict[str, Any]]
    agents: List[BaseAgent]
    tasks: List[Task]

    file_reader = FileReadTool()
    code_interpreter = CodeInterpreterTool()
    directory_reader = DirectoryReadTool()

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_engineer'],  # type: ignore[index]
            llm=AGENT_MODEL,
            tools=[self.directory_reader, self.file_reader],
            max_execution_time=600,
            memory=True,
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'],  # type: ignore[index]
            llm=AGENT_MODEL,
            tools=[self.code_interpreter],
            max_execution_time=600,
            memory=True,
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            reasoning=True,
            max_reasoning_attempts=3
        )

    @agent
    def model_builder(self) -> Agent:
        return Agent(
            config=self.agents_config['model_builder'],  # type: ignore[index]
            llm=AGENT_MODEL,
            tools=[self.code_interpreter],
            allow_code_execution=True,
            code_execution_mode="safe",
            memory=True,
            reasoning=True,
            max_execution_time=600,
            verbose=True,
        )

    @agent
    def insight_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['insight_reporter'],  # type: ignore[index]
            llm=AGENT_MODEL,
            tools=[self.code_interpreter],
            verbose=True
        )

    @task
    def collect_and_clean_data(self) -> Task:
        return Task(
            config=self.tasks_config['collect_and_clean_data'],  # type: ignore[index]
        )

    @task
    def analyze_data(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_data'],  # type: ignore[index]
        )

    @task
    def build_predictive_model(self) -> Task:
        return Task(
            config=self.tasks_config['build_predictive_model'],  # type: ignore[index]
        )

    @task
    def summarize_findings(self) -> Task:
        return Task(
            config=self.tasks_config['summarize_findings'],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DataAnalysisCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            planning=True,
            planning_llm=PLANNING_MODEL,
            verbose=True,
            memory=True
        )
