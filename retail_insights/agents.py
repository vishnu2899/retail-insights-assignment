from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from .data_access import RetailDataStore


@dataclass
class LanguageToQueryAgent:
    """Converts a natural language question into an analytical SQL query."""

    llm_client: Any
    datastore: RetailDataStore

    def run(self, question: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        schema_desc_lines = []
        for col in profile.get("columns", []):
            schema_desc_lines.append(f"- {col['name']} ({col['type']})")
        schema_desc = "\n".join(schema_desc_lines) or "(schema unknown)"

        system = SystemMessage(
            content=(
                "You are a senior analytics engineer. Given a business question "
                "and the table schema, write a single DuckDB-compatible SQL query "
                "against the table named `sales` that best answers the question. "
                "RULES:\n"
                "- Use the column names EXACTLY as they appear in the schema list.\n"
                "- If a column name contains spaces or special characters (for example "
                '  "SKU Code"), wrap it in double quotes in SQL (e.g. SELECT "SKU Code").\n'
                "- Do NOT invent new column names. If the question needs a field that "
                "is not present in the schema (for example dates or sales amounts), "
                "then return a query that selects zero rows and clearly explain in "
                "the reasoning which columns are missing.\n"
                "Respond in strict JSON with keys: sql, reasoning, confidence (0-1)."
            )
        )
        human = HumanMessage(
            content=(
                f"Table schema for `sales`:\n{schema_desc}\n\n"
                f"Business question: {question}"
            )
        )

        resp = self.llm_client.invoke([system, human])
        import json

        raw = str(resp.content)

        # Some models wrap JSON in markdown code fences like ```json ... ```.
        # Strip common fence patterns before parsing.
        if raw.strip().startswith("```"):
            # Remove leading ```... and trailing ``` if present
            stripped = raw.strip().lstrip("`")
            # After lstrip, first line may be 'json' or similar
            parts = stripped.split("\n", 1)
            if len(parts) == 2:
                raw_json = parts[1]
            else:
                raw_json = parts[0]
            # Remove trailing ``` if present
            raw_json = raw_json.rsplit("```", 1)[0]
        else:
            raw_json = raw

        try:
            parsed = json.loads(raw_json)
        except Exception:
            parsed = {"sql": "SELECT * FROM sales LIMIT 0", "reasoning": raw, "confidence": 0.3}

        return parsed


@dataclass
class DataExtractionAgent:
    """Executes SQL and prepares a compact result summary for the LLM."""

    datastore: RetailDataStore

    def run(self, sql: str) -> Dict[str, Any]:
        from pandas import DataFrame

        result: DataFrame = self.datastore.run_query(sql)
        preview = result.head(20)
        return {
            "row_count": int(result.shape[0]),
            "column_count": int(result.shape[1]),
            "columns": list(result.columns),
            "preview_rows": preview.to_dict(orient="records"),
        }


@dataclass
class ValidationAgent:
    """Validates whether the result sufficiently answers the question."""

    llm_client: Any

    def run(self, question: str, sql: str, result_summary: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if the result can answer the question, given the dataset profile.

        The profile (columns, row_count, etc.) lets the validator explain when the
        dataset itself is insufficient (e.g. missing date/sales columns for YoY)."""

        system = SystemMessage(
            content=(
                "You are a careful analytics reviewer. You will be given: a business "
                "question, the SQL that was executed, a compact result summary, and "
                "a high-level dataset profile (schema + row count). "
                "First, check whether the TABLE SCHEMA even contains the fields "
                "needed to answer the question (for example, dates for YoY or sales "
                "amounts for revenue questions). If the schema is missing required "
                "columns, clearly say that the dataset cannot support this question "
                "and name which columns are missing. Do NOT pretend to answer. "
                "Otherwise, assess whether the SQL result directly answers the "
                "question; if the result is empty or incomplete, explain that. "
                "Respond in strict JSON with keys: is_valid (true/false), "
                "confidence (0-1), critique, user_facing_answer."
            )
        )

        import json

        human = HumanMessage(
            content=json.dumps(
                {
                    "question": question,
                    "sql": sql,
                    "result_summary": result_summary,
                    "profile": profile,
                },
                indent=2,
            )
        )

        resp = self.llm_client.invoke([system, human])

        raw = str(resp.content)

        # Handle possible markdown code fences around the JSON (```json ... ```)
        if raw.strip().startswith("````") or raw.strip().startswith("```"):
            stripped = raw.strip().lstrip("`")
            parts = stripped.split("\n", 1)
            if len(parts) == 2:
                raw_json = parts[1]
            else:
                raw_json = parts[0]
            raw_json = raw_json.rsplit("```", 1)[0]
        else:
            raw_json = raw

        try:
            parsed = json.loads(raw_json)
        except Exception:
            parsed = {
                "is_valid": False,
                "confidence": 0.0,
                "critique": f"Could not parse validation response: {raw}",
                "user_facing_answer": "I'm not fully confident in this answer.",
            }
        return parsed
