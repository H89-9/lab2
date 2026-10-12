from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "brainsize.txt", sep="\t", na_values="NA")

output = folder / "plots"
output.mkdir(exist_ok=True)
features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

for feature in features:
    plt.figure(figsize=(7, 5))

    for gender in ["Male", "Female"]:
        group = df[df["Gender"] == gender]

        plt.scatter(
            group[feature],
            group["MRI_Count"],
            label=gender
        )

    plt.xlabel(feature)
    plt.ylabel("MRI_Count")
    plt.title(f"{feature} and MRI_Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f"scatter_{feature}.png", dpi=150)
    plt.close()


columns = features + ["MRI_Count"]

for gender in ["Male", "Female"]:
    group = df[df["Gender"] == gender]
    correlations = group[columns].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlations,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f"
    )

    plt.title(f"Correlation heatmap - {gender}")
    plt.tight_layout()
    plt.savefig(output / f"heatmap_{gender}.png", dpi=150)
    plt.close()

print("Все графики сохранены в папке plots")