#!/usr/bin/env python3
"""
Reads all JSON files from result directories in experiments/MINE/results/,
extracts the accuracy field, and plots comprehensive comparisons.
Automatically discovers and compares all directories.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from itertools import combinations


def discover_result_directories(results_folder="results"):
    """Discover all directories in the results folder."""
    results_path = Path(results_folder)
    if not results_path.exists():
        raise FileNotFoundError(f"Results folder does not exist: {results_folder}")

    directories = [d for d in results_path.iterdir() if d.is_dir()]
    directories.sort()  # Sort for consistent ordering
    return directories


def parse_accuracy(accuracy_str):
    """Convert accuracy string (e.g., '66.67%') to float."""
    return float(accuracy_str.replace("%", ""))


def read_accuracies_from_folder(folder_path):
    """Read all JSON files from a folder and extract accuracy values."""
    accuracies = []
    file_paths = []

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    # Get all JSON files and sort them
    json_files = sorted(folder.glob("results_*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                # The accuracy is the last element in the array
                if isinstance(data, list) and len(data) > 0:
                    last_item = data[-1]
                    if isinstance(last_item, dict) and "accuracy" in last_item:
                        accuracy_value = parse_accuracy(last_item["accuracy"])
                        accuracies.append(accuracy_value)
                        file_paths.append(json_file.name)
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}")

    return accuracies, file_paths


def plot_comparison(
    accuracies1,
    file_paths1,
    folder1_name,
    accuracies2,
    file_paths2,
    folder2_name,
    output_file="comparison.png",
):
    """Plot a comparison of accuracy values from two folders."""

    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    indices = list(range(len(accuracies1)))

    # Line chart
    ax1.plot(
        indices,
        accuracies1,
        marker="o",
        label=folder1_name,
        alpha=0.7,
        linewidth=2,
        markersize=4,
    )
    ax1.plot(
        indices,
        accuracies2,
        marker="s",
        label=folder2_name,
        alpha=0.7,
        linewidth=2,
        markersize=4,
    )
    ax1.set_xlabel("File Index")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy Comparison (Line Chart)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2.scatter(indices, accuracies1, alpha=0.6, label=folder1_name, s=50)
    ax2.scatter(indices, accuracies2, alpha=0.6, label=folder2_name, s=50)
    ax2.set_xlabel("File Index")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Comparison (Scatter Plot)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Box plot
    data = [accuracies1, accuracies2]
    bp = ax3.boxplot(data, tick_labels=[folder1_name, folder2_name], patch_artist=True)
    colors = ["lightblue", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Accuracy Distribution")
    ax3.grid(True, alpha=0.3)

    # Histogram
    ax4.hist(accuracies1, bins=20, alpha=0.6, label=folder1_name, color="lightblue")
    ax4.hist(accuracies2, bins=20, alpha=0.6, label=folder2_name, color="lightcoral")
    ax4.set_xlabel("Accuracy (%)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Accuracy Distribution (Histogram)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"{folder1_name:40} {folder2_name}")
    print(f"Mean:      {np.mean(accuracies1):35.2f} {np.mean(accuracies2):.2f}")
    print(f"Median:    {np.median(accuracies1):35.2f} {np.median(accuracies2):.2f}")
    print(f"Std Dev:   {np.std(accuracies1):35.2f} {np.std(accuracies2):.2f}")
    print(f"Min:       {np.min(accuracies1):35.2f} {np.min(accuracies2):.2f}")
    print(f"Max:       {np.max(accuracies1):35.2f} {np.max(accuracies2):.2f}")
    print("=" * 60)

    # Calculate difference
    differences = [a2 - a1 for a1, a2 in zip(accuracies1, accuracies2)]
    print(
        f"\n{folder2_name} vs {folder1_name} - Average difference: {np.mean(differences):.2f}%"
    )
    print(f"Better cases in {folder2_name}: {sum(1 for d in differences if d > 0)}")
    print(f"Better cases in {folder1_name}: {sum(1 for d in differences if d < 0)}")
    print(f"Equal cases: {sum(1 for d in differences if d == 0)}")
    print("=" * 60)


def plot_all_directories_comparison(
    results_dict, output_file="comprehensive_comparison.png"
):
    """Plot a comprehensive comparison of all directories."""

    # Filter out empty results
    results_dict = {
        name: data
        for name, data in results_dict.items()
        if len(data["accuracies"]) > 0
    }
    
    if not results_dict:
        print("Warning: No valid results to plot")
        return

    # Set style for better-looking plots
    plt.style.use("seaborn-v0_8-darkgrid")

    # Create figure with multiple subplots (2x3 grid, removed violin and stats table)
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Prepare data
    folder_names = list(results_dict.keys())
    all_accuracies = [results_dict[name]["accuracies"] for name in folder_names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(folder_names)))

    # 1. Line plot with all directories
    ax1 = fig.add_subplot(gs[0, :])
    for i, (name, data) in enumerate(results_dict.items()):
        accuracies = data["accuracies"]
        indices = list(range(len(accuracies)))
        ax1.plot(
            indices,
            accuracies,
            marker="o",
            label=name,
            alpha=0.7,
            linewidth=2,
            markersize=3,
            color=colors[i],
        )
    ax1.set_xlabel("File Index", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title(
        "Accuracy Comparison Across All Experiments", fontsize=14, fontweight="bold"
    )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[1, 0])
    bp = ax2.boxplot(all_accuracies, tick_labels=folder_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy Distribution", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45, labelsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Histogram overlay
    ax3 = fig.add_subplot(gs[1, 1])
    for i, (name, data) in enumerate(results_dict.items()):
        accuracies = data["accuracies"]
        ax3.hist(accuracies, bins=15, alpha=0.5, label=name, color=colors[i])
    ax3.set_xlabel("Accuracy (%)", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.set_title("Accuracy Distribution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Mean comparison bar chart
    ax4 = fig.add_subplot(gs[1, 2])
    means = [np.mean(acc) for acc in all_accuracies]
    stds = [np.std(acc) for acc in all_accuracies]
    x_pos = np.arange(len(folder_names))
    bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(folder_names, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax4.set_title("Mean Accuracy with Std Dev", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Comprehensive comparison plot saved to {output_file}")

    # Print detailed statistics
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICS")
    print("=" * 80)
    print(
        f"{'Experiment':<40} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}"
    )
    print("-" * 80)
    for name, data in results_dict.items():
        accuracies = data["accuracies"]
        if len(accuracies) > 0:
            print(
                f"{name:<40} {np.mean(accuracies):<10.2f} {np.median(accuracies):<10.2f} "
                f"{np.std(accuracies):<10.2f} {np.min(accuracies):<10.2f} {np.max(accuracies):<10.2f}"
            )
        else:
            print(f"{name:<40} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    print("=" * 80)

    # Rank experiments by mean accuracy
    ranked = sorted(
        results_dict.items(), 
        key=lambda x: np.mean(x[1]["accuracies"]) if len(x[1]["accuracies"]) > 0 else -1, 
        reverse=True
    )
    print("\n" + "=" * 80)
    print("RANKING (by Mean Accuracy)")
    print("=" * 80)
    for rank, (name, data) in enumerate(ranked, 1):
        if len(data["accuracies"]) > 0:
            mean_acc = np.mean(data["accuracies"])
            print(f"{rank}. {name:<50} {mean_acc:.2f}%")
        else:
            print(f"{rank}. {name:<50} N/A (no data)")
    print("=" * 80)


def write_summary_file(results_dict, output_file="results/summary.txt"):
    """Write a detailed summary of all results to a text file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("KNOWLEDGE GRAPH EVALUATION - COMPREHENSIVE ANALYSIS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'Experiment':<45} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n"
        )
        f.write("-" * 100 + "\n")

        for name, data in results_dict.items():
            accuracies = data["accuracies"]
            if len(accuracies) > 0:
                f.write(
                    f"{name:<45} "
                    f"{np.mean(accuracies):>10.2f} "
                    f"{np.median(accuracies):>10.2f} "
                    f"{np.std(accuracies):>10.2f} "
                    f"{np.min(accuracies):>10.2f} "
                    f"{np.max(accuracies):>10.2f}\n"
                )
            else:
                f.write(
                    f"{name:<45} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10} "
                    f"{'N/A':>10}\n"
                )
        f.write("-" * 100 + "\n\n")

        # Ranking
        ranked = sorted(
            results_dict.items(),
            key=lambda x: np.mean(x[1]["accuracies"]) if len(x[1]["accuracies"]) > 0 else -1,
            reverse=True,
        )
        f.write("RANKING (by Mean Accuracy)\n")
        f.write("-" * 100 + "\n")
        for rank, (name, data) in enumerate(ranked, 1):
            if len(data["accuracies"]) > 0:
                mean_acc = np.mean(data["accuracies"])
                f.write(f"{rank}. {name:<60} {mean_acc:>10.2f}%\n")
            else:
                f.write(f"{rank}. {name:<60} {'N/A':>10}\n")
        f.write("-" * 100 + "\n\n")

        # Pairwise comparisons
        f.write("PAIRWISE COMPARISONS\n")
        f.write("=" * 100 + "\n\n")

        folder_names = list(results_dict.keys())
        pairs = list(combinations(folder_names, 2))

        for name1, name2 in pairs:
            data1 = results_dict[name1]
            data2 = results_dict[name2]
            accuracies1 = data1["accuracies"]
            accuracies2 = data2["accuracies"]

            # Truncate to same length if needed
            min_len = min(len(accuracies1), len(accuracies2))
            
            # Skip if no data to compare
            if min_len == 0:
                f.write(f"{name2} vs {name1}\n")
                f.write("-" * 100 + "\n")
                f.write("  No data available for comparison\n\n")
                continue
            
            acc1 = accuracies1[:min_len]
            acc2 = accuracies2[:min_len]

            differences = [a2 - a1 for a1, a2 in zip(acc1, acc2)]
            mean_diff = np.mean(differences)
            wins_2 = sum(1 for d in differences if d > 0)
            wins_1 = sum(1 for d in differences if d < 0)
            ties = sum(1 for d in differences if d == 0)

            f.write(f"{name2} vs {name1}\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Mean difference: {mean_diff:+.2f}% ")
            if mean_diff > 0:
                f.write(f"({name2} performs BETTER on average)\n")
            elif mean_diff < 0:
                f.write(f"({name1} performs BETTER on average)\n")
            else:
                f.write("(TIE)\n")
            f.write(
                f"  {name2} wins: {wins_2}/{min_len} cases ({100 * wins_2 / min_len:.1f}%)\n"
            )
            f.write(
                f"  {name1} wins: {wins_1}/{min_len} cases ({100 * wins_1 / min_len:.1f}%)\n"
            )
            f.write(f"  Ties: {ties}/{min_len} cases ({100 * ties / min_len:.1f}%)\n")

            # Find largest wins/losses
            max_win_2 = max(differences)
            max_win_1 = min(differences)
            max_win_2_idx = differences.index(max_win_2)
            max_win_1_idx = differences.index(max_win_1)

            f.write(
                f"\n  Largest advantage for {name2}: {max_win_2:+.2f}% (file index {max_win_2_idx})\n"
            )
            f.write(
                f"  Largest advantage for {name1}: {max_win_1:+.2f}% (file index {max_win_1_idx})\n"
            )
            f.write("\n")

        # Analysis by accuracy ranges
        f.write("\n" + "=" * 100 + "\n")
        f.write("PERFORMANCE ANALYSIS BY ACCURACY RANGES\n")
        f.write("=" * 100 + "\n\n")

        ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
        for low, high in ranges:
            f.write(f"Accuracy Range: {low}% - {high}%\n")
            f.write("-" * 100 + "\n")
            for name, data in results_dict.items():
                accuracies = data["accuracies"]
                if len(accuracies) > 0:
                    in_range = [
                        acc
                        for acc in accuracies
                        if low <= acc < high or (high == 100 and acc == 100)
                    ]
                    count = len(in_range)
                    percentage = 100 * count / len(accuracies)
                    f.write(f"  {name:<60} {count:>4} cases ({percentage:>5.1f}%)\n")
                else:
                    f.write(f"  {name:<60} {'N/A':>4}\n")
            f.write("\n")

        # Key insights
        f.write("=" * 100 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 100 + "\n\n")

        # Filter out empty results for insights
        valid_ranked = [(name, data) for name, data in ranked if len(data["accuracies"]) > 0]
        
        if len(valid_ranked) >= 2:
            best_model = valid_ranked[0][0]
            worst_model = valid_ranked[-1][0]
            best_mean = np.mean(valid_ranked[0][1]["accuracies"])
            worst_mean = np.mean(valid_ranked[-1][1]["accuracies"])

            f.write(f"1. BEST PERFORMER: {best_model}\n")
            f.write(f"   - Mean accuracy: {best_mean:.2f}%\n")
            f.write(
                f"   - Outperforms others by {best_mean - worst_mean:.2f}% on average\n\n"
            )

            f.write(f"2. LOWEST PERFORMER: {worst_model}\n")
            f.write(f"   - Mean accuracy: {worst_mean:.2f}%\n\n")
        elif len(valid_ranked) == 1:
            best_model = valid_ranked[0][0]
            best_mean = np.mean(valid_ranked[0][1]["accuracies"])
            f.write(f"1. ONLY VALID PERFORMER: {best_model}\n")
            f.write(f"   - Mean accuracy: {best_mean:.2f}%\n\n")
        else:
            f.write("No valid results to analyze.\n\n")

        # Check for unexpected patterns
        if len(valid_ranked) >= 1:
            best_model = valid_ranked[0][0]
            if "minimal" in best_model.lower():
                f.write("3. ⚠️  UNEXPECTED FINDING:\n")
                f.write(
                    "   The 'minimal' configuration outperforms configurations with more reasoning.\n"
                )
                f.write("   This suggests:\n")
                f.write(
                    "   - More reasoning tokens may introduce noise or hallucinations\n"
                )
                f.write("   - The task may benefit from concise, focused retrieval\n")
                f.write("   - Over-reasoning might dilute the key information\n")
                f.write(
                    "   - The embedding/retrieval system may work better with simpler representations\n\n"
                )

        # Consistency analysis
        f.write("4. CONSISTENCY ANALYSIS:\n")
        for name, data in results_dict.items():
            accuracies = data["accuracies"]
            if len(accuracies) > 0:
                std = np.std(accuracies)
                f.write(f"   {name}: Std Dev = {std:.2f}% ")
                if std < 15:
                    f.write("(Very consistent)\n")
                elif std < 20:
                    f.write("(Moderately consistent)\n")
                else:
                    f.write("(High variance)\n")
            else:
                f.write(f"   {name}: N/A (No data)\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("End of Summary\n")
        f.write("=" * 100 + "\n")

    print(f"\n✓ Summary written to {output_path}")


def compare_all_pairs(results_dict, output_dir="pairwise_comparisons"):
    """Generate pairwise comparisons for all directory pairs."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    folder_names = list(results_dict.keys())
    pairs = list(combinations(folder_names, 2))

    print(f"\nGenerating {len(pairs)} pairwise comparisons...")

    for name1, name2 in pairs:
        data1 = results_dict[name1]
        data2 = results_dict[name2]

        accuracies1 = data1["accuracies"]
        accuracies2 = data2["accuracies"]
        
        # Skip if either dataset is empty
        if len(accuracies1) == 0 or len(accuracies2) == 0:
            print(f"  Skipping {name1} vs {name2}: Empty dataset")
            continue
        
        file_paths1 = data1["file_paths"]
        file_paths2 = data2["file_paths"]

        # Handle different lengths by truncating to minimum
        if len(accuracies1) != len(accuracies2):
            min_len = min(len(accuracies1), len(accuracies2))
            accuracies1 = accuracies1[:min_len]
            accuracies2 = accuracies2[:min_len]
            file_paths1 = file_paths1[:min_len]
            file_paths2 = file_paths2[:min_len]
            print(f"  {name1} vs {name2}: Comparing first {min_len} files")

        # Create safe filename
        safe_name1 = name1.replace("/", "_").replace(" ", "_")
        safe_name2 = name2.replace("/", "_").replace(" ", "_")
        output_file = output_path / f"{safe_name1}_vs_{safe_name2}.png"

        plot_comparison(
            accuracies1,
            file_paths1,
            name1,
            accuracies2,
            file_paths2,
            name2,
            str(output_file),
        )

    print(f"✓ All pairwise comparisons saved to {output_dir}/")


def main():
    """
    Automatically discovers and compares all result directories in experiments/MINE/results/
    Generates comprehensive comparison and pairwise comparisons.
    """
    # Get the script directory to find results folder
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    print("=" * 80)
    print("KNOWLEDGE GRAPH EVALUATION - COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print(f"\nAuto-discovering directories in {results_dir}...")

    try:
        directories = discover_result_directories(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if not directories:
        print(f"No directories found in {results_dir}")
        return 1

    print(f"\nFound {len(directories)} directories:")
    for d in directories:
        print(f"  ✓ {d.name}")

    # Read accuracies from all directories
    results_dict = {}
    print("\nReading results from each directory...")
    print("-" * 80)

    for directory in directories:
        dir_name = directory.name
        try:
            print(f"\n{dir_name}:")
            accuracies, file_paths = read_accuracies_from_folder(directory)
            print(f"  → Found {len(accuracies)} JSON files")
            print(f"  → Mean accuracy: {np.mean(accuracies):.2f}%")
            results_dict[dir_name] = {
                "accuracies": accuracies,
                "file_paths": file_paths,
            }
        except Exception as e:
            print(f"  ✗ Warning: Could not read from {directory}: {e}")

    if len(results_dict) < 2:
        print(
            f"\n✗ Error: Need at least 2 valid directories to compare. Found {len(results_dict)}."
        )
        return 1

    print("\n" + "=" * 80)
    print(f"Successfully loaded {len(results_dict)} result sets")
    print("=" * 80)

    # Generate comprehensive comparison
    output_file = results_dir / "results.png"
    plot_all_directories_comparison(results_dict, str(output_file))

    # Write summary file
    summary_file = results_dir / "summary.txt"
    write_summary_file(results_dict, str(summary_file))

    # Generate pairwise comparisons
    pairwise_dir = results_dir / "comparisons"
    compare_all_pairs(results_dict, str(pairwise_dir))

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Output files:")
    print(f"  • Comprehensive comparison: {output_file}")
    print(f"  • Summary report: {summary_file}")
    print(f"  • Pairwise comparisons: {pairwise_dir}/")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
