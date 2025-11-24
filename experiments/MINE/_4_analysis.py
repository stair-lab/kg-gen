#!/usr/bin/env python3
"""
Analyze correlations between graph characteristics and model performance.
Extracts various graph statistics and correlates them with accuracy scores.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx
from scipy import stats
from collections import Counter


def load_kg_from_file(kg_file: Path) -> Dict:
    """Load a knowledge graph from a JSON file."""
    try:
        with open(kg_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {kg_file}: {e}")
        return None


def kg_to_networkx(kg_data: Dict) -> nx.DiGraph:
    """Convert KG data to NetworkX graph."""
    G = nx.DiGraph()
    
    # Add entities as nodes
    if "entities" in kg_data:
        for entity in kg_data["entities"]:
            G.add_node(entity)
    
    # Add relations as edges
    if "relations" in kg_data:
        for relation in kg_data["relations"]:
            if len(relation) == 3:
                source, rel_type, target = relation
                G.add_edge(source, target, relation_type=rel_type)
    
    return G


def extract_graph_statistics(kg_data: Dict) -> Dict:
    """Extract comprehensive statistics from a knowledge graph."""
    if kg_data is None:
        return None
    
    stats_dict = {}
    
    # Basic counts
    stats_dict["num_entities"] = len(kg_data.get("entities", []))
    stats_dict["num_relations"] = len(kg_data.get("relations", []))
    
    # Relation type diversity
    relation_types = [r[1] for r in kg_data.get("relations", []) if len(r) == 3]
    stats_dict["num_unique_relation_types"] = len(set(relation_types))
    
    # Relation type distribution
    relation_type_counts = Counter(relation_types)
    stats_dict["most_common_relation_count"] = relation_type_counts.most_common(1)[0][1] if relation_type_counts else 0
    stats_dict["relation_type_entropy"] = calculate_entropy(list(relation_type_counts.values()))
    
    # Entity/Relation ratio
    stats_dict["entities_to_relations_ratio"] = (
        stats_dict["num_entities"] / stats_dict["num_relations"]
        if stats_dict["num_relations"] > 0
        else 0
    )
    
    # NetworkX-based metrics
    G = kg_to_networkx(kg_data)
    
    if G.number_of_nodes() > 0:
        # Connectivity metrics
        stats_dict["num_connected_components"] = nx.number_weakly_connected_components(G)
        stats_dict["num_strongly_connected_components"] = nx.number_strongly_connected_components(G)
        
        # Largest component size
        if nx.number_weakly_connected_components(G) > 0:
            largest_wcc = max(nx.weakly_connected_components(G), key=len)
            stats_dict["largest_component_size"] = len(largest_wcc)
            stats_dict["largest_component_ratio"] = len(largest_wcc) / G.number_of_nodes()
        else:
            stats_dict["largest_component_size"] = 0
            stats_dict["largest_component_ratio"] = 0
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        stats_dict["avg_degree"] = np.mean(degrees) if degrees else 0
        stats_dict["max_degree"] = np.max(degrees) if degrees else 0
        stats_dict["avg_in_degree"] = np.mean(in_degrees) if in_degrees else 0
        stats_dict["avg_out_degree"] = np.mean(out_degrees) if out_degrees else 0
        stats_dict["degree_std"] = np.std(degrees) if degrees else 0
        
        # Graph density
        stats_dict["density"] = nx.density(G)
        
        # Isolated nodes
        stats_dict["num_isolated_nodes"] = len(list(nx.isolates(G)))
        stats_dict["isolated_nodes_ratio"] = (
            stats_dict["num_isolated_nodes"] / G.number_of_nodes()
            if G.number_of_nodes() > 0
            else 0
        )
        
        # Centrality measures (on largest component for efficiency)
        if stats_dict["largest_component_size"] > 1:
            largest_subgraph = G.subgraph(largest_wcc).copy()
            try:
                # PageRank
                pagerank_values = list(nx.pagerank(largest_subgraph).values())
                stats_dict["avg_pagerank"] = np.mean(pagerank_values)
                stats_dict["max_pagerank"] = np.max(pagerank_values)
                
                # In-degree centrality
                in_degree_centrality = list(nx.in_degree_centrality(largest_subgraph).values())
                stats_dict["avg_in_degree_centrality"] = np.mean(in_degree_centrality)
                
                # Betweenness centrality (sample for large graphs)
                if largest_subgraph.number_of_nodes() < 100:
                    betweenness = list(nx.betweenness_centrality(largest_subgraph).values())
                    stats_dict["avg_betweenness"] = np.mean(betweenness)
                    stats_dict["max_betweenness"] = np.max(betweenness)
                else:
                    stats_dict["avg_betweenness"] = 0
                    stats_dict["max_betweenness"] = 0
            except Exception as e:
                print(f"Warning: Could not compute centrality measures: {e}")
                stats_dict["avg_pagerank"] = 0
                stats_dict["max_pagerank"] = 0
                stats_dict["avg_in_degree_centrality"] = 0
                stats_dict["avg_betweenness"] = 0
                stats_dict["max_betweenness"] = 0
        else:
            stats_dict["avg_pagerank"] = 0
            stats_dict["max_pagerank"] = 0
            stats_dict["avg_in_degree_centrality"] = 0
            stats_dict["avg_betweenness"] = 0
            stats_dict["max_betweenness"] = 0
        
        # Reciprocity (for directed graphs)
        stats_dict["reciprocity"] = nx.reciprocity(G)
        
    else:
        # Empty graph
        stats_dict["num_connected_components"] = 0
        stats_dict["num_strongly_connected_components"] = 0
        stats_dict["largest_component_size"] = 0
        stats_dict["largest_component_ratio"] = 0
        stats_dict["avg_degree"] = 0
        stats_dict["max_degree"] = 0
        stats_dict["avg_in_degree"] = 0
        stats_dict["avg_out_degree"] = 0
        stats_dict["degree_std"] = 0
        stats_dict["density"] = 0
        stats_dict["num_isolated_nodes"] = 0
        stats_dict["isolated_nodes_ratio"] = 0
        stats_dict["avg_pagerank"] = 0
        stats_dict["max_pagerank"] = 0
        stats_dict["avg_in_degree_centrality"] = 0
        stats_dict["avg_betweenness"] = 0
        stats_dict["max_betweenness"] = 0
        stats_dict["reciprocity"] = 0
    
    return stats_dict


def calculate_entropy(counts: List[int]) -> float:
    """Calculate Shannon entropy of a distribution."""
    if not counts:
        return 0
    total = sum(counts)
    if total == 0:
        return 0
    probabilities = [c / total for c in counts]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)


def parse_accuracy(accuracy_str: str) -> float:
    """Convert accuracy string (e.g., '66.67%') to float."""
    return float(accuracy_str.replace("%", ""))


def load_accuracy_from_results(results_file: Path) -> float:
    """Load accuracy from a results JSON file."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                last_item = data[-1]
                if isinstance(last_item, dict) and "accuracy" in last_item:
                    return parse_accuracy(last_item["accuracy"])
    except Exception as e:
        print(f"Warning: Could not load accuracy from {results_file}: {e}")
    return None


def collect_data_for_model(model_dir: Path) -> Tuple[List[Dict], List[float]]:
    """Collect graph statistics and accuracies for all essays in a model directory."""
    graph_stats = []
    accuracies = []
    
    # Find all kg_*.json files
    kg_files = sorted(model_dir.glob("kg_*.json"))
    
    for kg_file in kg_files:
        # Extract essay index from filename
        essay_idx = int(kg_file.stem.split("_")[1])
        
        # Load corresponding results file
        results_file = model_dir / f"results_{essay_idx}.json"
        
        if not results_file.exists():
            continue
        
        # Load graph statistics
        kg_data = load_kg_from_file(kg_file)
        stats = extract_graph_statistics(kg_data)
        
        # Load accuracy
        accuracy = load_accuracy_from_results(results_file)
        
        if stats is not None and accuracy is not None:
            stats["essay_idx"] = essay_idx
            stats["model"] = model_dir.name
            graph_stats.append(stats)
            accuracies.append(accuracy)
    
    return graph_stats, accuracies


def collect_all_data(results_dir: Path) -> pd.DataFrame:
    """Collect data from all models."""
    all_stats = []
    all_accuracies = []
    
    # Get all model directories (exclude 'comparisons')
    model_dirs = [d for d in results_dir.iterdir() 
                  if d.is_dir() and d.name != "comparisons"]
    
    print(f"Found {len(model_dirs)} model directories")
    
    for model_dir in model_dirs:
        print(f"Processing {model_dir.name}...")
        stats, accuracies = collect_data_for_model(model_dir)
        
        if stats:
            all_stats.extend(stats)
            all_accuracies.extend(accuracies)
            print(f"  → Collected {len(stats)} data points")
    
    # Create DataFrame
    df = pd.DataFrame(all_stats)
    df["accuracy"] = all_accuracies
    
    return df


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between graph metrics and accuracy."""
    # Select numeric columns (excluding essay_idx)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["essay_idx", "accuracy"]]
    
    correlations = []
    
    for col in numeric_cols:
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(df[col], df["accuracy"])
        
        # Spearman correlation (rank-based, more robust)
        spearman_corr, spearman_p = stats.spearmanr(df[col], df["accuracy"])
        
        correlations.append({
            "metric": col,
            "pearson_r": pearson_corr,
            "pearson_p": pearson_p,
            "spearman_r": spearman_corr,
            "spearman_p": spearman_p,
            "abs_pearson": abs(pearson_corr),
            "abs_spearman": abs(spearman_corr)
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("abs_spearman", ascending=False)
    
    return corr_df


def plot_correlation_heatmap(df: pd.DataFrame, output_file: Path):
    """Plot correlation heatmap of all metrics."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["essay_idx"]]
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix: Graph Metrics vs Accuracy", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Correlation heatmap saved to {output_file}")


def plot_top_correlations(corr_df: pd.DataFrame, df: pd.DataFrame, output_file: Path, top_n=9):
    """Plot scatter plots for top correlated metrics."""
    top_metrics = corr_df.head(top_n)["metric"].tolist()
    
    # Create subplot grid
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, metric in enumerate(top_metrics):
        ax = axes[idx]
        
        # Scatter plot with regression line
        sns.regplot(data=df, x=metric, y="accuracy", ax=ax, 
                   scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
        
        # Get correlation values
        corr_info = corr_df[corr_df["metric"] == metric].iloc[0]
        
        # Title with correlation info
        ax.set_title(
            f"{metric}\nSpearman r={corr_info['spearman_r']:.3f} (p={corr_info['spearman_p']:.3e})",
            fontsize=10
        )
        ax.set_xlabel(metric, fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots if any
    for idx in range(len(top_metrics), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle("Top Graph Metrics Correlated with Accuracy", 
                fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Top correlations plot saved to {output_file}")


def plot_model_comparison(df: pd.DataFrame, output_file: Path):
    """Plot comparison of key metrics across models."""
    # Calculate mean statistics per model
    model_stats = df.groupby("model").agg({
        "accuracy": "mean",
        "num_entities": "mean",
        "num_relations": "mean",
        "num_unique_relation_types": "mean",
        "avg_degree": "mean",
        "density": "mean",
        "largest_component_ratio": "mean"
    }).reset_index()
    
    # Sort by accuracy
    model_stats = model_stats.sort_values("accuracy", ascending=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics = ["accuracy", "num_entities", "num_relations", 
               "num_unique_relation_types", "avg_degree", "density"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Bar plot
        bars = ax.bar(range(len(model_stats)), model_stats[metric], alpha=0.7)
        
        # Color bars by accuracy
        colors = plt.cm.RdYlGn(model_stats["accuracy"] / 100)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(range(len(model_stats)))
        ax.set_xticklabels(model_stats["model"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"Mean {metric} by Model", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("Model Comparison: Key Graph Metrics", 
                fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Model comparison plot saved to {output_file}")


def analyze_per_model_correlations(df: pd.DataFrame, output_file: Path):
    """Analyze correlations within each model separately."""
    models = df["model"].unique()
    
    results = []
    
    for model in models:
        model_df = df[df["model"] == model]
        
        if len(model_df) < 5:  # Need at least 5 data points
            continue
        
        # Compute correlations for this model
        corr_df = compute_correlations(model_df)
        
        # Get top 5 metrics
        top_5 = corr_df.head(5)
        
        results.append({
            "model": model,
            "n_samples": len(model_df),
            "mean_accuracy": model_df["accuracy"].mean(),
            "top_correlated_metrics": ", ".join(top_5["metric"].tolist()[:3]),
            "top_correlations": ", ".join([f"{r:.3f}" for r in top_5["spearman_r"].tolist()[:3]])
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_accuracy", ascending=False)
    
    # Save to file
    with open(output_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("PER-MODEL CORRELATION ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"Model: {row['model']}\n")
            f.write(f"  Samples: {row['n_samples']}\n")
            f.write(f"  Mean Accuracy: {row['mean_accuracy']:.2f}%\n")
            f.write(f"  Top Correlated Metrics: {row['top_correlated_metrics']}\n")
            f.write(f"  Correlations: {row['top_correlations']}\n\n")
    
    print(f"✓ Per-model analysis saved to {output_file}")


def write_comprehensive_report(df: pd.DataFrame, corr_df: pd.DataFrame, output_file: Path):
    """Write a comprehensive analysis report."""
    with open(output_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("GRAPH CHARACTERISTICS vs PERFORMANCE CORRELATION ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        # Overall statistics
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total data points: {len(df)}\n")
        f.write(f"Number of models: {df['model'].nunique()}\n")
        f.write(f"Models: {', '.join(df['model'].unique())}\n")
        f.write(f"Average accuracy across all: {df['accuracy'].mean():.2f}%\n")
        f.write(f"Accuracy range: {df['accuracy'].min():.2f}% - {df['accuracy'].max():.2f}%\n\n")
        
        # Top correlations
        f.write("=" * 100 + "\n")
        f.write("TOP CORRELATED METRICS (Spearman)\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Metric':<40} {'Spearman r':<15} {'P-value':<15} {'Significance':<15}\n")
        f.write("-" * 100 + "\n")
        
        for _, row in corr_df.head(15).iterrows():
            sig = "***" if row["spearman_p"] < 0.001 else "**" if row["spearman_p"] < 0.01 else "*" if row["spearman_p"] < 0.05 else ""
            f.write(f"{row['metric']:<40} {row['spearman_r']:<15.4f} {row['spearman_p']:<15.6f} {sig:<15}\n")
        
        f.write("\n")
        
        # Key insights
        f.write("=" * 100 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 100 + "\n\n")
        
        # Find most significant positive and negative correlations
        significant = corr_df[corr_df["spearman_p"] < 0.05]
        
        if len(significant) > 0:
            # Positive correlations
            positive = significant[significant["spearman_r"] > 0].head(5)
            if len(positive) > 0:
                f.write("1. POSITIVE CORRELATIONS (metrics that increase with better accuracy):\n")
                for _, row in positive.iterrows():
                    f.write(f"   • {row['metric']}: r={row['spearman_r']:.4f} (p={row['spearman_p']:.6f})\n")
                f.write("\n")
            
            # Negative correlations
            negative = significant[significant["spearman_r"] < 0].head(5)
            if len(negative) > 0:
                f.write("2. NEGATIVE CORRELATIONS (metrics that decrease with better accuracy):\n")
                for _, row in negative.iterrows():
                    f.write(f"   • {row['metric']}: r={row['spearman_r']:.4f} (p={row['spearman_p']:.6f})\n")
                f.write("\n")
        else:
            f.write("No statistically significant correlations found (p < 0.05)\n\n")
        
        # Model-specific insights
        f.write("=" * 100 + "\n")
        f.write("MODEL-SPECIFIC STATISTICS\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Model':<45} {'N':<6} {'Acc':<8} {'Entities':<10} {'Relations':<10} {'Degree':<10}\n")
        f.write("-" * 100 + "\n")
        
        model_stats = df.groupby("model").agg({
            "accuracy": "mean",
            "num_entities": "mean",
            "num_relations": "mean",
            "avg_degree": "mean"
        }).reset_index()
        model_stats["count"] = df.groupby("model").size().values
        model_stats = model_stats.sort_values("accuracy", ascending=False)
        
        for _, row in model_stats.iterrows():
            f.write(f"{row['model']:<45} {row['count']:<6.0f} {row['accuracy']:<8.2f} "
                   f"{row['num_entities']:<10.1f} {row['num_relations']:<10.1f} {row['avg_degree']:<10.2f}\n")
        
        f.write("\n")
        
        # Interpretation
        f.write("=" * 100 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 100 + "\n\n")
        
        top_metric = corr_df.iloc[0]
        
        f.write(f"The metric with the strongest correlation to accuracy is '{top_metric['metric']}' ")
        f.write(f"(Spearman r={top_metric['spearman_r']:.4f}, p={top_metric['spearman_p']:.6f}).\n\n")
        
        if abs(top_metric['spearman_r']) > 0.3 and top_metric['spearman_p'] < 0.05:
            direction = "positively" if top_metric['spearman_r'] > 0 else "negatively"
            f.write(f"This suggests that {top_metric['metric']} is {direction} associated with ")
            f.write("model performance. This correlation is statistically significant and may indicate ")
            f.write("that this graph characteristic plays an important role in retrieval quality.\n\n")
        elif top_metric['spearman_p'] >= 0.05:
            f.write("However, this correlation is not statistically significant (p >= 0.05), ")
            f.write("suggesting that graph characteristics may not strongly predict performance, ")
            f.write("or that other factors (e.g., prompt quality, model capabilities) dominate.\n\n")
        else:
            f.write("The correlation is weak, suggesting that simple graph statistics may not ")
            f.write("be sufficient to predict model performance on this task.\n\n")
        
        # Check for potential confounding factors
        f.write("POTENTIAL CONFOUNDING FACTORS:\n")
        f.write("- Different models may have inherent capabilities independent of graph structure\n")
        f.write("- Query difficulty may vary across essays\n")
        f.write("- Graph quality (semantic correctness) is not captured by structural metrics\n")
        f.write("- Embedding model quality affects retrieval regardless of graph structure\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("End of Analysis\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Comprehensive report saved to {output_file}")


def main():
    """Main analysis pipeline."""
    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    output_dir = script_dir / "results" / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 100)
    print("GRAPH CHARACTERISTICS vs PERFORMANCE CORRELATION ANALYSIS")
    print("=" * 100)
    print()
    
    # Step 1: Collect data
    print("Step 1: Collecting graph statistics and accuracy data...")
    df = collect_all_data(results_dir)
    print(f"✓ Collected {len(df)} data points from {df['model'].nunique()} models\n")
    
    # Save raw data
    data_file = output_dir / "graph_statistics_data.csv"
    df.to_csv(data_file, index=False)
    print(f"✓ Raw data saved to {data_file}\n")
    
    # Step 2: Compute correlations
    print("Step 2: Computing correlations...")
    corr_df = compute_correlations(df)
    
    # Save correlation data
    corr_file = output_dir / "correlations.csv"
    corr_df.to_csv(corr_file, index=False)
    print(f"✓ Correlations saved to {corr_file}\n")
    
    # Step 3: Generate visualizations
    print("Step 3: Generating visualizations...")
    
    # Correlation heatmap
    plot_correlation_heatmap(df, output_dir / "correlation_heatmap.png")
    
    # Top correlations scatter plots
    plot_top_correlations(corr_df, df, output_dir / "top_correlations.png")
    
    # Model comparison
    plot_model_comparison(df, output_dir / "model_comparison.png")
    
    print()
    
    # Step 4: Per-model analysis
    print("Step 4: Analyzing per-model correlations...")
    analyze_per_model_correlations(df, output_dir / "per_model_analysis.txt")
    print()
    
    # Step 5: Write comprehensive report
    print("Step 5: Writing comprehensive report...")
    write_comprehensive_report(df, corr_df, output_dir / "analysis_report.txt")
    print()
    
    # Print summary
    print("=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)
    print("\nTop 10 Correlated Metrics:")
    print("-" * 100)
    for idx, row in corr_df.head(10).iterrows():
        sig = "***" if row["spearman_p"] < 0.001 else "**" if row["spearman_p"] < 0.01 else "*" if row["spearman_p"] < 0.05 else ""
        print(f"{idx+1:2d}. {row['metric']:<40} r={row['spearman_r']:>7.4f} (p={row['spearman_p']:.6f}) {sig}")
    
    print("\n" + "=" * 100)
    print("Output files saved to:", output_dir)
    print("  • graph_statistics_data.csv - Raw data")
    print("  • correlations.csv - Correlation coefficients")
    print("  • correlation_heatmap.png - Full correlation matrix")
    print("  • top_correlations.png - Top correlated metrics")
    print("  • model_comparison.png - Cross-model comparison")
    print("  • per_model_analysis.txt - Per-model insights")
    print("  • analysis_report.txt - Comprehensive report")
    print("=" * 100)


if __name__ == "__main__":
    main()

