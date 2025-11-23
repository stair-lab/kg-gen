#!/usr/bin/env python3
"""
Simple Streamlit dashboard to view knowledge graph evaluation results in a table.
"""

import streamlit as st
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

st.set_page_config(page_title="KG Evaluation Results", layout="wide")


@st.cache_data
def load_hf_dataset():
    """Load the HuggingFace dataset."""
    dataset = load_dataset("josancamon/kg-gen-MINE-evaluation-dataset")["train"]
    return dataset.to_list()


@st.cache_data
def discover_result_directories(results_folder="experiments/MINE/results"):
    """Discover all directories in the results folder."""
    results_path = Path(results_folder)
    if not results_path.exists():
        return []

    directories = [d for d in results_path.iterdir() if d.is_dir()]
    directories.sort()
    return [d.name for d in directories]


@st.cache_data
def load_all_results(results_folder="experiments/MINE/results"):
    """Load all results from all directories."""
    results_path = Path(results_folder)
    all_results = {}

    directories = [d for d in results_path.iterdir() if d.is_dir()]

    for directory in directories:
        model_name = directory.name
        model_results = {}

        json_files = sorted(directory.glob("results_*.json"))
        for json_file in json_files:
            try:
                idx = int(json_file.stem.split("_")[1])
                with open(json_file, "r") as f:
                    data = json.load(f)
                    # Remove accuracy summary if present
                    if isinstance(data, list) and len(data) > 0:
                        if (
                            isinstance(data[-1], dict)
                            and "accuracy" in data[-1]
                            and len(data[-1]) == 1
                        ):
                            data = data[:-1]
                    model_results[idx] = data
            except Exception as e:
                pass

        if model_results:
            all_results[model_name] = model_results

    return all_results


def main():
    st.title("Knowledge Graph Evaluation Results")

    # Load data
    with st.spinner("Loading data..."):
        dataset = load_hf_dataset()
        all_results = load_all_results()

    if not all_results:
        st.error("No results found in experiments/MINE/results/")
        return

    model_names = list(all_results.keys())

    # Essay selector
    essay_idx = st.selectbox(
        "Select Essay",
        range(len(dataset)),
        format_func=lambda x: f"Essay {x}: {dataset[x].get('essay_topic', 'Unknown')}",
    )

    essay_data = dataset[essay_idx]
    queries = essay_data.get("generated_queries", [])

    st.subheader(f"Essay Topic: {essay_data.get('essay_topic', 'Unknown')}")

    # Build table data
    table_data = []

    for query_idx, query in enumerate(queries):
        row = {
            "Query #": query_idx + 1,
            "Query": query,
        }

        # Add columns for each model's retrieved context and evaluation
        for model_name in model_names:
            model_results = all_results.get(model_name, {}).get(essay_idx, [])

            if query_idx < len(model_results):
                result = model_results[query_idx]
                context = result.get("retrieved_context", "N/A")
                evaluation = result.get("evaluation", 0)

                row[f"{model_name} - Context"] = context
                row[f"{model_name} - Evaluation"] = "✅" if evaluation == 1 else "❌"
            else:
                row[f"{model_name} - Context"] = "N/A"
                row[f"{model_name} - Evaluation"] = "N/A"

        table_data.append(row)

    # Display as dataframe
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.warning("No queries found for this essay.")

    # Show essay content in toggleable expander (default open)
    essay_content = essay_data.get("essay_content", "N/A")
    with st.expander("Essay Content", expanded=True):
        st.text(essay_content.replace("```", ""))


if __name__ == "__main__":
    main()
