# Global Energy Pattern Classification

An end-to-end machine learning project that predicts `continent` from global energy indicators using multiple classification models, evaluation metrics, cross-validation, and visual analytics.

## Key Highlights

- End-to-end reproducible ML workflow in one script
- Comparison of 7 classification models with multiple metrics
- Cross-validation and learning-curve based model reliability checks
- Hyperparameter tuning for ensemble models
- Export-ready CSV reports and publication-style plots

## Snapshot

| Item | Details |
|---|---|
| Problem Type | Multiclass Classification |
| Target Column | `continent` |
| Input File | `global_energy_analytics.csv` |
| Script | `commands.py` |
| Output Folder | `output/` |

## Pipeline Overview

The workflow in `commands.py` includes:

1. Data loading and cleaning
2. Numeric feature selection and label encoding
3. Train/test split with stratification
4. Imputation + scaling via preprocessing pipeline
5. Training 7 ML models
6. Performance comparison (Accuracy, Precision, Recall, F1, ROC-AUC)
7. 5-fold cross-validation
8. Best-model diagnostics and visualizations
9. Learning curve analysis
10. Hyperparameter tuning (Random Forest, Extra Trees)

## Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (RBF)
- Random Forest
- Gradient Boosting
- Extra Trees

## Quick Start

1. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

2. Run the script

```bash
python commands.py
```

## Project Structure

```text
Comprehensive Energy Analytics/
|-- commands.py
|-- global_energy_analytics.csv
|-- README.md
`-- output/
	|-- model_comparison_results_extended.csv
	|-- classwise_classification_report.csv
	|-- cross_validation_results.csv
	|-- learning_curve_results.csv
	|-- hyperparameter_tuning_results.csv
	|-- best_model_predictions.csv
	`-- *.png visual outputs
```

## Output Files

All generated files are saved to `output/`.

### CSV Reports

- `model_comparison_results_extended.csv`
- `classwise_classification_report.csv`
- `cross_validation_results.csv`
- `learning_curve_results.csv`
- `hyperparameter_tuning_results.csv`
- `best_model_predictions.csv`

### Visual Outputs

- `model_accuracy_comparison.png`
- `precision_recall_f1_comparison.png`
- `cross_validation_accuracy.png`
- `best_model_confusion_matrix.png`
- `best_model_multiclass_roc_curve.png`
- `feature_importance_top10.png` (if supported by best model)
- `pca_projection.png`
- `correlation_heatmap.png`
- `class_distribution.png`
- `feature_histograms.png`
- `learning_curve_best_model.png`

### Conditional Output

- `feature_importance.csv` (generated only when the selected best model exposes feature importances)

## Evaluation Metrics Tracked

The project tracks model quality using:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- ROC-AUC (multiclass one-vs-rest, where supported)

## Reproducibility

- Fixed random seed (`random_state=42`) in train/test split and model configurations where available
- Stratified split to preserve class distribution
- Output artifacts saved consistently in `output/` for easier reruns and comparison

## Notes

- Column names are standardized to lowercase with underscores.
- Duplicate rows are removed.
- Rows with missing `continent` values are dropped.
- Only numeric columns are used as model features.
- The script raises an error if `continent` is missing or if no numeric features are found.
