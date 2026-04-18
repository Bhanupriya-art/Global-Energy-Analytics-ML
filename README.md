# Global Energy Pattern Classification using Ensemble Machine Learning Models

This project builds and compares multiple machine learning classifiers to predict the `continent` label from numeric global energy indicators in `global_energy_analytics.csv`.

The script handles preprocessing, model training, evaluation, and visualization in one end-to-end pipeline.

## What The Script Does

The main workflow in `commands.py`:

1. Loads `global_energy_analytics.csv`
2. Cleans data:
- Removes duplicate rows
- Standardizes column names (lowercase, underscores)
- Drops rows with missing target values (`continent`)
3. Selects numeric feature columns only
4. Encodes target labels using `LabelEncoder`
5. Splits data into train/test with stratification (`test_size=0.2`, `random_state=42`)
6. Builds a preprocessing pipeline:
- Median imputation for missing numeric values
- Standard scaling
7. Trains and evaluates multiple classifiers:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (RBF kernel)
- Random Forest
- Gradient Boosting
- Extra Trees
8. Compares model accuracies and selects the best model
9. Generates visual analysis:
- Model accuracy bar chart
- Confusion matrix for the best model
- PCA 2D projection plot
- Correlation heatmap
- Class distribution chart
- Feature histograms
10. Exports predictions and metrics to CSV files

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib

## How To Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

2. From the project root, run:

```bash
python commands.py
```

## Generated Outputs

The script writes the following files in the project root:

- `model_comparison_results.csv`
- `model_accuracy_comparison.png`
- `best_model_confusion_matrix.png`
- `feature_importance.csv` (only if the best model supports feature importances)
- `feature_importance_top10.png` (only if the best model supports feature importances)
- `pca_projection.png`
- `correlation_heatmap.png`
- `class_distribution.png`
- `feature_histograms.png`
- `best_model_predictions.csv`

## Notes

- The target column is fixed as `continent`.
- Only numeric feature columns are used for model training.
- If no numeric columns are available, the script raises an error.
- If `continent` is missing from the dataset, the script raises an error.
