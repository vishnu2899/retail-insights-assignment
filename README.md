# Retail Insights Assistant

GenAI-powered Retail Insights Assistant that analyzes sales data, generates summaries, and answers ad-hoc business questions.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Setup & Execution Guide

- **Python version**: 3.10+ recommended.  
- **Create and activate a virtual environment** (optional but recommended):  
  - Windows (Command Prompt):  
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

1. **Install dependencies**  
   From the project root:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your LLM API key**  
   The app expects an OpenAI-compatible API key exposed as an environment variable. For example (PowerShell):
   ```powershell
   [System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "YOUR_KEY_HERE", "User")
   ```

3. **Run the Streamlit app**  
   ```bash
   streamlit run app.py
   ```

4. **Use the UI**  
   - Open the local URL Streamlit prints (usually http://localhost:8501).  
   - Upload a CSV / Excel / JSON file with retail / sales data.  
   - Choose either **Summarization** (high-level overview) or **Conversational Q&A** (ask a free-form question).  
   - Review the answer and, if needed, inspect the debug trace of the multi-agent pipeline.

## Technical Notes

### Assumptions

- The uploaded dataset can be loaded into memory (assignment-scale data).  
- A single logical table named `sales` is sufficient for most analyses.  
- Column names and basic data types are enough for the Language-to-Query agent to infer reasonable SQL.  
- The LLM has access to an OpenAI-compatible Chat API via `langchain-openai`.

### Limitations

- **Single-table focus**: the current implementation assumes one main `sales` table; complex star schemas are described in the architecture but not fully implemented in code.  
- **Schema dependency**: if the dataset is missing required fields (e.g., dates for YoY or amounts for revenue), the system correctly refuses to answer rather than hallucinating.  
- **In-memory analytics**: DuckDB runs in-memory, which is appropriate for the assignment but not for 100GB-scale data.  
- **No full RAG pipeline yet**: the architecture describes RAG over documentation and aggregates, but the production-grade vector store and retrieval layer are not wired up in this codebase.

### Possible Improvements

- **Scale-out data platform**: move from local DuckDB to a cloud warehouse (BigQuery / Snowflake / Databricks) backed by Parquet/Delta in object storage.  
- **Richer schema understanding**: add a retrieval step over schema docs and metric definitions to help the Language-to-Query agent generate more reliable SQL.  
- **RAG over business docs**: index PDFs / wikis (e.g., pricing policies, audit procedures) and add a document-QA mode alongside dataset-QA.  
- **More advanced orchestration**: migrate from a linear Python pipeline to LangGraph or a similar framework with explicit branches and fallbacks.  
- **Evaluation harness**: create a small benchmark of questions + expected answers / SQL to systematically track quality over time.
