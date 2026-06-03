import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(json_path="results.json"):
    """Loads results from the JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_sensitivity_plots(results, output_path="sensitivity_comparison.png"):
    """
    Plots the sensitivity curves (Label Fraction vs. Test mAP@0.5)
    for all methods head-to-head.
    """
    fractions = [1, 5, 10, 20]
    fractions_labels = ["1%", "5%", "10%", "20%"]
    
    # Extract data series
    supervised_map = []
    stac_fixed_map = []
    stac_adapt_map = []
    soft_fixed_map = []
    soft_adapt_map = []
    
    for frac in fractions:
        frac_str = f"{frac}%"
        # Supervised
        supervised_map.append(results["supervised"].get(frac_str, {}).get("mAP@0.5", 0.0))
        # STAC Fixed
        stac_fixed_map.append(results["stac_fixed"].get(frac_str, {}).get("mAP@0.5", 0.0))
        # STAC Adaptive
        stac_adapt_map.append(results["stac_adaptive"].get(frac_str, {}).get("mAP@0.5", 0.0))
        # Soft Teacher Fixed
        soft_fixed_map.append(results["soft_teacher_fixed"].get(frac_str, {}).get("mAP@0.5", 0.0))
        # Soft Teacher Adaptive
        soft_adapt_map.append(results["soft_teacher_adaptive"].get(frac_str, {}).get("mAP@0.5", 0.0))
        
    plt.figure(figsize=(10, 6))
    
    # Plot curves with distinct markers and colors
    plt.plot(fractions_labels, supervised_map, 'o-', color='#e056fd', linewidth=2.5, label='Faster R-CNN (Supervised Baseline)')
    plt.plot(fractions_labels, stac_fixed_map, 's--', color='#30336b', linewidth=2, label='STAC (Fixed Threshold)')
    plt.plot(fractions_labels, stac_adapt_map, 's-', color='#130cb7', linewidth=2.5, label='STAC (Adaptive Threshold)')
    plt.plot(fractions_labels, soft_fixed_map, '^--', color='#f0932b', linewidth=2, label='Soft Teacher (Fixed Threshold)')
    plt.plot(fractions_labels, soft_adapt_map, '^-', color='#eb4d4b', linewidth=2.5, label='Soft Teacher (Adaptive Threshold)')
    
    plt.title("Annotation Sensitivity: Label Fraction vs. WBC Detector Performance", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Label Fraction (%)", fontsize=12, labelpad=10)
    plt.ylabel("mAP@0.5", fontsize=12, labelpad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved sensitivity curves plot to {output_path}")

def generate_pseudo_label_quality_plots(results, output_path="pseudo_label_quality.png"):
    """
    Plots the pseudo-label precision vs. recall for Fixed vs. Adaptive thresholds,
    demonstrating degradation trends at low annotation rates.
    """
    fractions = [1, 5, 10, 20]
    fractions_labels = ["1%", "5%", "10%", "20%"]
    
    # Collect values for STAC (used as reference)
    fixed_precision = []
    fixed_recall = []
    adapt_precision = []
    adapt_recall = []
    
    for frac in fractions:
        frac_str = f"{frac}%"
        fixed_precision.append(results["stac_fixed"].get(frac_str, {}).get("pl_precision", 0.0))
        fixed_recall.append(results["stac_fixed"].get(frac_str, {}).get("pl_recall", 0.0))
        
        adapt_precision.append(results["stac_adaptive"].get(frac_str, {}).get("pl_precision", 0.0))
        adapt_recall.append(results["stac_adaptive"].get(frac_str, {}).get("pl_recall", 0.0))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Pseudo-label Precision
    ax1.plot(fractions_labels, fixed_precision, 'o--', color='#30336b', linewidth=2, label='Fixed Threshold (0.85)')
    ax1.plot(fractions_labels, adapt_precision, 'o-', color='#130cb7', linewidth=2.5, label='Adaptive Threshold')
    ax1.set_title("Pseudo-Label Precision Degradation", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Label Fraction (%)", fontsize=10)
    ax1.set_ylabel("Precision (TP / Predicted)", fontsize=10)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')
    
    # Subplot 2: Pseudo-label Recall
    ax2.plot(fractions_labels, fixed_recall, 's--', color='#f0932b', linewidth=2, label='Fixed Threshold (0.85)')
    ax2.plot(fractions_labels, adapt_recall, 's-', color='#eb4d4b', linewidth=2.5, label='Adaptive Threshold')
    ax2.set_title("Pseudo-Label Recall Starvation", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Label Fraction (%)", fontsize=10)
    ax2.set_ylabel("Recall (TP / Ground Truth)", fontsize=10)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')
    
    plt.suptitle("Analysis of Pseudo-Label Degradation under Scarcity (Fixed vs. Adaptive)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved pseudo-label quality plot to {output_path}")

def generate_markdown_table(results):
    """
    Generates a beautifully formatted markdown table summarizing all results.
    """
    fractions = [1, 5, 10, 20]
    
    headers = [
        "Method & Strategy", 
        "Fraction", 
        "Test mAP@0.5", 
        "Test mAP@0.5:0.95", 
        "Test Precision", 
        "Test Recall", 
        "Pseudo-Label Precision", 
        "Pseudo-Label Recall"
    ]
    
    row_format = "| {:<30} | {:<8} | {:<12} | {:<16} | {:<14} | {:<11} | {:<22} | {:<19} |"
    
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * len(h) for h in headers]))
    
    def format_val(v, pct=False):
        if v is None or v == 0.0:
            return "-"
        return f"{v:.1%}" if pct else f"{v:.4f}"
        
    configs = [
        ("supervised", "Faster R-CNN (Supervised Baseline)"),
        ("stac_fixed", "STAC (Fixed Thresh = 0.85)"),
        ("stac_adaptive", "STAC (Adaptive Thresh)"),
        ("soft_teacher_fixed", "Soft Teacher (Fixed Thresh = 0.85)"),
        ("soft_teacher_adaptive", "Soft Teacher (Adaptive Thresh)")
    ]
    
    for key, name in configs:
        for frac in fractions:
            frac_str = f"{frac}%"
            res = results.get(key, {}).get(frac_str, {})
            
            row = [
                name,
                frac_str,
                format_val(res.get("mAP@0.5")),
                format_val(res.get("mAP@0.5:0.95")),
                format_val(res.get("precision")),
                format_val(res.get("recall")),
                format_val(res.get("pl_precision")),
                format_val(res.get("pl_recall"))
            ]
            lines.append(row_format.format(*row))
            name = ""  # blank out name for cleaner layout in subsequent fraction rows
            
    table = "\n".join(lines)
    print("\n" + "=" * 40 + " COMPARISON TABLE " + "=" * 40)
    print(table)
    print("=" * 98)
    
    with open("results_table.md", "w") as f:
        f.write("# Experimental Evaluation Summary Table\n\n")
        f.write(table)
        f.write("\n")
    print("Saved summary table to results_table.md")

if __name__ == "__main__":
    try:
        res = load_results()
        generate_sensitivity_plots(res)
        generate_pseudo_label_quality_plots(res)
        generate_markdown_table(res)
    except FileNotFoundError:
        print("results.json not found. Run run_experiments.py first to generate results.")
