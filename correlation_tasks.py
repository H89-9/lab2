from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    import pandas as pd

    df = pd.read_csv(
        data.csv_path,
        sep="\t",
        na_values="NA"
    )

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    features = [
        "FSIQ",
        "VIQ",
        "PIQ",
        "Weight",
        "Height"
    ]

    women_mri_correlation = {}
    men_mri_correlation = {}

    for feature in features:
        women_mri_correlation[feature] = float(
        women[feature].corr(women["MRI_Count"])
        )
        men_mri_correlation[feature] = float(
        men[feature].corr(men["MRI_Count"])
        )
    strongest_mri_feature = ""
    strongest_value = -1

    for feature in features:
        women_value = abs(women_mri_correlation[feature])
        men_value = abs(men_mri_correlation[feature])

        if women_value > strongest_value:
            strongest_value = women_value
            strongest_mri_feature = feature
        if men_value > strongest_value:
            strongest_value = men_value
            strongest_mri_feature = feature

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )