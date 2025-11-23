#!/usr/bin/env python3
"""
Simple Streamlit dashboard to view knowledge graph evaluation results in a table.
"""

import streamlit as st
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from kg_gen import KGGen
from kg_gen.models import Graph

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


def load_kg_file(model_name: str, essay_idx: int, results_folder="experiments/MINE/results"):
    """Load knowledge graph file for a specific model and essay."""
    kg_path = Path(results_folder) / model_name / f"kg_{essay_idx}.json"
    if kg_path.exists():
        with open(kg_path, "r") as f:
            kg_data = json.load(f)
        return Graph(**kg_data)
    return None


def visualize_kg_in_browser(graph: Graph, model_name: str, essay_idx: int):
    """Generate and open KG visualization in a new browser window."""
    if graph is None or not graph.entities:
        st.warning(f"No knowledge graph available for {model_name}")
        return
    
    # Generate HTML visualization with a descriptive filename
    output_dir = Path("experiments/MINE/results") / model_name
    output_path = output_dir / f"kg_{essay_idx}_visualization.html"
    
    KGGen.visualize(graph, str(output_path), open_in_browser=True)
    st.success("✅ Knowledge graph visualization opened in a new browser window!")
    st.caption(f"File saved to: {output_path}")


def main():
    st.title("Knowledge Graph Evaluation Results")

    # Load data
    with st.spinner("Loading data..."):
        dataset = load_hf_dataset()
        all_results = load_all_results()

    if not all_results:
        st.error("No results found in experiments/MINE/results/")
        return

    model_names = sorted(list(all_results.keys()))

    # Model selector
    st.subheader("Select Models to Compare")
    selected_models = st.multiselect(
        "Models",
        model_names,
        default=model_names,
        label_visibility="collapsed",
    )
    
    if not selected_models:
        st.warning("Please select at least one model to view results.")
        return

    # Find essays that have results for at least one selected model
    available_essays = set()
    for model_name in selected_models:
        if model_name in all_results:
            available_essays.update(all_results[model_name].keys())
    
    available_essays = sorted(list(available_essays))
    
    if not available_essays:
        st.error("No essay results found for selected models")
        return

    # Essay selector (only show essays with results)
    essay_idx = st.selectbox(
        "Select Essay",
        available_essays,
        format_func=lambda x: f"Essay {x}: {dataset[x].get('essay_topic', 'Unknown')}",
    )

    essay_data = dataset[essay_idx]
    queries = essay_data.get("generated_queries", [])

    st.subheader(f"Essay Topic: {essay_data.get('essay_topic', 'Unknown')}")
    
    # Show which selected models have data for this essay
    models_without_data = [name for name in selected_models if essay_idx not in all_results.get(name, {})]
    
    if models_without_data:
        st.info(f"⚠️ {len(models_without_data)} model(s) missing results for this essay: {', '.join(models_without_data)}")

    # Build table data
    table_data = []

    for query_idx, query in enumerate(queries):
        row = {
            "Query #": query_idx + 1,
            "Query": query,
        }

        # Add columns for each selected model's retrieved context and evaluation
        for model_name in selected_models:
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

    # Knowledge Graph Visualization Section
    st.markdown("---")
    st.subheader("🔍 View Knowledge Graph")
    
    # Find which models have KG files for this essay
    models_with_kg = []
    for model_name in selected_models:
        kg_path = Path("experiments/MINE/results") / model_name / f"kg_{essay_idx}.json"
        if kg_path.exists():
            models_with_kg.append(model_name)
    
    if models_with_kg:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_kg_model = st.selectbox(
                "Select model to view knowledge graph",
                models_with_kg,
                key="kg_model_selector",
                label_visibility="collapsed"
            )
        
        with col2:
            if st.button("🔍 View Graph", type="primary", use_container_width=True):
                with st.spinner(f"Loading knowledge graph for {selected_kg_model}..."):
                    graph = load_kg_file(selected_kg_model, essay_idx)
                    if graph:
                        st.caption(f"Entities: {len(graph.entities)} | Relations: {len(graph.relations)}")
                        visualize_kg_in_browser(graph, selected_kg_model, essay_idx)
                    else:
                        st.error(f"Failed to load knowledge graph for {selected_kg_model}")
    else:
        st.info("No knowledge graph files available for the selected models and essay.")

    # Show essay content in toggleable expander (default open)
    st.markdown("---")
    essay_content = essay_data.get("essay_content", "N/A")
    with st.expander("📄 Essay Content", expanded=True):
        st.text(essay_content.replace("```", ""))


if __name__ == "__main__":
    main()
