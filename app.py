import os
import streamlit as st
import pandas as pd

from retail_insights.llm_config import get_llm_client
from retail_insights.data_access import RetailDataStore
from retail_insights.orchestration import RetailInsightsOrchestrator


st.set_page_config(page_title="Retail Insights Assistant", layout="wide")


@st.cache_resource(show_spinner=False)
def get_orchestrator():
    llm = get_llm_client()
    datastore = RetailDataStore()
    return RetailInsightsOrchestrator(llm_client=llm, datastore=datastore)


def main():
    st.title("Retail Insights Assistant")
    st.markdown("Analyze retail sales data, generate summaries, and ask business questions in natural language.")

    orchestrator = get_orchestrator()

    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload sales CSV or summarized report (CSV, XLSX, or JSON)",
        type=["csv", "xlsx", "json"],
    )

    mode = st.sidebar.radio("2. Mode", ["Summarization", "Conversational Q&A"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            try:
                df = orchestrator.load_data(uploaded_file)
                st.session_state["data_loaded"] = True
                st.session_state["data_shape"] = df.shape
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

        st.success(f"Data loaded with shape {st.session_state['data_shape']}")
        st.write("Preview of data:")
        st.dataframe(df.head())

        if mode == "Summarization":
            st.subheader("Automated Performance Summary")
            if st.button("Generate summary"):
                with st.spinner("Generating summary using multi-agent pipeline..."):
                    summary = orchestrator.run_summarization()
                st.markdown("### Summary")
                st.write(summary)

        else:
            st.subheader("Conversational Q&A")
            user_query = st.text_input("Ask a question about the sales performance")

            if user_query:
                with st.spinner("Thinking..."):
                    answer, debug_info = orchestrator.run_qa(user_query, st.session_state.chat_history)

                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                for msg in st.session_state.chat_history:
                    speaker = "You" if msg["role"] == "user" else "Assistant"
                    st.markdown(f"**{speaker}:** {msg['content']}")

                with st.expander("Debug info (agents & intermediate steps)"):
                    st.json(debug_info)

    else:
        st.info("Upload a dataset in the sidebar to get started.")


if __name__ == "__main__":
    main()
