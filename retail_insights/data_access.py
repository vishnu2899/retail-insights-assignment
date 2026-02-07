from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, Optional

import duckdb
import pandas as pd


@dataclass
class RetailDataStore:
    """In-memory analytical store built on top of DuckDB.

    For the assignment scale, we load the uploaded dataset into DuckDB in-memory.
    In the scalable design (100GB+), this is replaced by external tables / cloud
    data warehouse connections, but the query surface remains similar.
    """

    conn: duckdb.DuckDBPyConnection | None = None
    table_name: str = "sales"

    def _ensure_connection(self) -> duckdb.DuckDBPyConnection:
        if self.conn is None:
            self.conn = duckdb.connect(database=":memory:")
        return self.conn

    def load_file(self, uploaded_file: Any) -> pd.DataFrame:
        """Load uploaded file (CSV, XLSX, JSON) into DuckDB and return DataFrame."""

        conn = self._ensure_connection()

        filename = getattr(uploaded_file, "name", "uploaded")
        name_lower = filename.lower()

        if name_lower.endswith(".csv"):
            data = uploaded_file.read()
            df = pd.read_csv(io.BytesIO(data))
        elif name_lower.endswith(".xlsx") or name_lower.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        elif name_lower.endswith(".json"):
            data = uploaded_file.read()
            df = pd.read_json(io.BytesIO(data))
        else:
            raise ValueError("Unsupported file type. Please upload CSV, XLSX, or JSON.")

        # Register DataFrame as a DuckDB table for SQL querying
        conn.unregister(self.table_name) if self.table_name in conn.execute("SHOW TABLES").fetchdf()["name"].values else None
        conn.register(self.table_name, df)

        return df

    def run_query(self, sql: str) -> pd.DataFrame:
        conn = self._ensure_connection()
        return conn.execute(sql).fetchdf()

    def basic_profile(self) -> Dict[str, Any]:
        """Return lightweight profile stats useful for summarization prompts."""

        conn = self._ensure_connection()
        profile: Dict[str, Any] = {}

        try:
            profile["row_count"] = int(conn.execute(f"SELECT COUNT(*) AS c FROM {self.table_name}").fetchone()[0])
        except Exception:
            profile["row_count"] = None

        try:
            cols_df = conn.execute(f"PRAGMA table_info('{self.table_name}')").fetchdf()
            profile["columns"] = cols_df[["name", "type"]].to_dict(orient="records")
        except Exception:
            profile["columns"] = []

        return profile

    def get_aggregates_for_summary(self) -> Dict[str, Any]:
        """Compute a few generic aggregates to ground the LLM summary.

        Uses best-effort heuristics: tries to infer amount column names.
        """

        conn = self._ensure_connection()
        cols_df = conn.execute(f"PRAGMA table_info('{self.table_name}')").fetchdf()
        col_names = [c.lower() for c in cols_df["name"].tolist()]

        amount_col: Optional[str] = None
        for candidate in ["sales", "amount", "revenue", "gmv", "total"]:
            for original in cols_df["name"]:
                if original.lower() == candidate or candidate in original.lower():
                    amount_col = original
                    break
            if amount_col:
                break

        aggregates: Dict[str, Any] = {"amount_column": amount_col}

        if amount_col:
            try:
                total_df = conn.execute(f"SELECT SUM({amount_col}) AS total_sales FROM {self.table_name}").fetchdf()
                aggregates["total_sales"] = float(total_df["total_sales"].iloc[0])
            except Exception:
                aggregates["total_sales"] = None

        # Attempt to detect a date column
        date_col: Optional[str] = None
        for original in cols_df["name"]:
            lname = original.lower()
            if "date" in lname or lname in ("order_date", "invoice_date"):
                date_col = original
                break

        aggregates["date_column"] = date_col

        if date_col and amount_col:
            try:
                trend_df = conn.execute(
                    f"""
                    SELECT
                        DATE_TRUNC('month', {date_col}) AS month,
                        SUM({amount_col}) AS monthly_sales
                    FROM {self.table_name}
                    GROUP BY 1
                    ORDER BY 1
                    LIMIT 24
                    """
                ).fetchdf()
                aggregates["monthly_trend"] = trend_df.to_dict(orient="records")
            except Exception:
                aggregates["monthly_trend"] = []

        return aggregates
