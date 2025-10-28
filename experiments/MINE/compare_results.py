#!/usr/bin/env python3
"""
Reads all JSON files from each folder, extracts the accuracy field, and plots a comparison.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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
    json_files = sorted(folder.glob("*.json"))

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


def main():
    parser = argparse.ArgumentParser(
        description="Compare accuracy values from two folders with the same JSON file structure"
    )
    parser.add_argument("folder1", help="Path to first folder")
    parser.add_argument("folder2", help="Path to second folder")
    parser.add_argument(
        "--name1",
        default=None,
        help="Display name for first folder (default: folder name)",
    )
    parser.add_argument(
        "--name2",
        default=None,
        help="Display name for second folder (default: folder name)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="comparison.png",
        help="Output file for the plot (default: comparison.png)",
    )

    args = parser.parse_args()

    # Get display names
    folder1_name = args.name1 if args.name1 else Path(args.folder1).name
    folder2_name = args.name2 if args.name2 else Path(args.folder2).name

    print(f"Reading from {args.folder1}...")
    accuracies1, file_paths1 = read_accuracies_from_folder(args.folder1)
    print(f"Found {len(accuracies1)} JSON files in {args.folder1}")

    print(f"Reading from {args.folder2}...")
    accuracies2, file_paths2 = read_accuracies_from_folder(args.folder2)
    print(f"Found {len(accuracies2)} JSON files in {args.folder2}")

    # Check if they have the same number of files
    if len(accuracies1) != len(accuracies2):
        print(
            f"Warning: Different number of files ({len(accuracies1)} vs {len(accuracies2)})"
        )
        min_len = min(len(accuracies1), len(accuracies2))
        accuracies1 = accuracies1[:min_len]
        accuracies2 = accuracies2[:min_len]
        print(f"Comparing first {min_len} files")

    # Check if file names match
    mismatched_files = []
    for fp1, fp2 in zip(file_paths1, file_paths2):
        if fp1 != fp2:
            mismatched_files.append((fp1, fp2))

    if mismatched_files:
        print(f"\nWarning: Found {len(mismatched_files)} mismatched file names")
        if len(mismatched_files) <= 5:
            for fp1, fp2 in mismatched_files:
                print(f"  {fp1} != {fp2}")

    # Plot comparison
    plot_comparison(
        accuracies1,
        file_paths1,
        folder1_name,
        accuracies2,
        file_paths2,
        folder2_name,
        args.output,
    )


if __name__ == "__main__":
    main()
