from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from .agents import DataExtractionAgent, LanguageToQueryAgent, ValidationAgent
from .data_access import RetailDataStore


@dataclass
class RetailInsightsOrchestrator:
    """Coordinates agents for summarization and conversational Q&A."""

    llm_client: Any
    datastore: RetailDataStore

    def __post_init__(self) -> None:
        self.lang_to_sql_agent = LanguageToQueryAgent(self.llm_client, self.datastore)
        self.data_agent = DataExtractionAgent(self.datastore)
        self.validation_agent = ValidationAgent(self.llm_client)

    # Data loading
    def load_data(self, uploaded_file: Any):
        return self.datastore.load_file(uploaded_file)

    # Summarization mode
    def run_summarization(self) -> str:
        profile = self.datastore.basic_profile()
        aggregates = self.datastore.get_aggregates_for_summary()

        system = SystemMessage(
            content=(
                "You are a senior retail analytics consultant. You are given "
                "a brief profile and some aggregates from a sales dataset. "
                "Write a concise, business-friendly summary of performance. "
                "Highlight growth/decline, best/worst performing regions or products "
                "if visible, and any anomalies. Limit to 3-6 bullet points."
            )
        )

        import json

        human = HumanMessage(
            content=json.dumps(
                {
                    "profile": profile,
                    "aggregates": aggregates,
                },
                indent=2,
            )
        )

        resp = self.llm_client.invoke([system, human])
        return resp.content

    # Conversational Q&A
    def run_qa(self, question: str, chat_history: List[Dict[str, str]]):
        debug: Dict[str, Any] = {"steps": []}

        profile = self.datastore.basic_profile()

        # Step 1: language-to-SQL
        l2q_output = self.lang_to_sql_agent.run(question, profile)
        debug["steps"].append({"agent": "language_to_query", "output": l2q_output})

        sql = l2q_output.get("sql", "SELECT * FROM sales LIMIT 0")

        # Step 2: execute SQL
        data_output = self.data_agent.run(sql)
        debug["steps"].append({"agent": "data_extraction", "output": data_output})

        # Step 3: validate and formulate answer
        validation_output = self.validation_agent.run(question, sql, data_output, profile)
        debug["steps"].append({"agent": "validation", "output": validation_output})

        answer = validation_output.get("user_facing_answer", "I was not able to derive a confident answer.")

        return answer, debug
