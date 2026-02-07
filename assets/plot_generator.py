import matplotlib.pyplot as plt
import numpy as np
import os

# Set global professional style
plt.style.use('seaborn-v0_8-muted')
COLORS = ['#4A90E2', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7']

def save_plot(filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

# --- 1. Dataset Distribution ---
def plot_distribution():
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    counts = [327, 514, 1099, 115, 1113, 6705, 142]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color=COLORS[0])
    plt.ylabel("Number of Images")
    plt.title("HAM10000 Class Distribution (Severe Imbalance)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 50, yval, ha='center', va='bottom')
    
    save_plot("01_dataset_distribution.png")

# --- 2. CNN Benchmarking (Macro-F1 & Accuracy) ---
def plot_cnn_benchmarking():
    models = ["CustomCNN", "EffNetB0", "EffNetB3", "DenseNet121"]
    test_acc = [0.6660, 0.6700, 0.6487, 0.7864]
    test_f1 = [0.3314, 0.1242, 0.1476, 0.6029]
    
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, test_acc, width, label='Accuracy', color='#A0C4FF')
    plt.bar(x + width/2, test_f1, width, label='Macro-F1', color='#FFADAD')
    
    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("CNN Architecture Comparison: Accuracy vs. Macro-F1")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    
    save_plot("03_cnn_benchmark_comparison.png")

# --- 3. Class-wise Analysis (DenseNet121) ---
def plot_class_wise():
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    f1_scores = [0.40, 0.62, 0.61, 0.43, 0.50, 0.90, 0.75]
    recall = [0.37, 0.73, 0.62, 0.29, 0.49, 0.90, 0.68]
    
    x = np.arange(len(classes))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, f1_scores, width, label='F1 Score', color='#59A14F')
    plt.bar(x + width/2, recall, width, label='Recall (Sensitivity)', color='#EDC948')
    
    plt.xticks(x, classes)
    plt.ylabel("Score")
    plt.title("DenseNet121 Class-wise Performance Breakdown")
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    save_plot("06_class_wise_sensitivity.png")

# --- 4. Final Hybrid Model Comparison (Ablation Results) ---
def plot_final_comparison():
    models = [
        "DenseNet121\n(Baseline)", 
        "CNN + Transf.\n+ MLP", 
        "CNN + Transf.\n+ XGBoost", 
        "DenseNet\n+ XGBoost"
    ]
    macro_f1 = [0.6029, 0.5427, 0.5619, 0.5131]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, macro_f1, color=['#4A90E2', '#95A5A6', '#95A5A6', '#95A5A6'])
    plt.ylabel("Macro-F1 (Test)")
    plt.title("Ablation Study: Baseline vs. Hybrid Architectures")
    plt.ylim(0.4, 0.65) # Zoomed in to show differences
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Highlight that baseline is superior
    plt.text(0, 0.61, "Best Performer", ha='center', color='blue', fontweight='bold')
    
    save_plot("07_final_hybrid_comparison.png")

# --- 5. Accuracy vs F1 Scatter (The "Collapse" Visualization) ---
def plot_scatter():
    models = ["CustomCNN", "EffNetB0", "EffNetB3", "DenseNet121"]
    test_acc = [0.6660, 0.6700, 0.6487, 0.7864]
    test_f1 = [0.3314, 0.1242, 0.1476, 0.6029]
    
    plt.figure(figsize=(8, 7))
    for i, model in enumerate(models):
        plt.scatter(test_acc[i], test_f1[i], s=200, label=model, alpha=0.8)
        plt.text(test_acc[i] + 0.005, test_f1[i] + 0.005, model, fontsize=10)

    plt.xlabel("Overall Accuracy")
    plt.ylabel("Balanced Macro-F1")
    plt.title("The 'Imbalance Gap': Accuracy vs. Macro-F1")
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.3, label='Majority Collapse Zone')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc='lower right')
    
    save_plot("04_acc_vs_f1_scatter.png")

if __name__ == "__main__":
    print("🚀 Generating academic assets...")
    plot_distribution()
    plot_cnn_benchmarking()
    plot_class_wise()
    plot_final_comparison()
    plot_scatter()
    print("✨ All assets generated successfully in the current folder.")