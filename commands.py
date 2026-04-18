import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    RandomizedSearchCV,
    learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.decomposition import PCA


# =========================================================
# 1. Setup Output Folder
# =========================================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_path = "global_energy_analytics.csv"


# =========================================================
# 2. Load Dataset
# =========================================================
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())


# =========================================================
# 3. Basic Cleaning
# =========================================================
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\nMissing Values:\n", df.isnull().sum())

target_col = "continent"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_features]

if len(numeric_features) == 0:
    raise ValueError("No numeric features found in dataset.")

print("\nNumeric Features Used:\n", numeric_features)


# =========================================================
# 4. Encode Target
# =========================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nTarget Classes:", list(label_encoder.classes_))
n_classes = len(label_encoder.classes_)


# =========================================================
# 5. Train-Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# =========================================================
# 6. Preprocessing
# =========================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features)
])


# =========================================================
# 7. Models
# =========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42)
}


# =========================================================
# 8. Train and Evaluate Models
# =========================================================
results = []
trained_pipelines = {}
classification_reports = []

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Multiclass ROC-AUC
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_score = pipeline.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
    else:
        roc_auc = np.nan

    results.append([model_name, acc, prec, rec, f1, roc_auc])
    trained_pipelines[model_name] = pipeline

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    for cls_name in label_encoder.classes_:
        if cls_name in report_dict:
            classification_reports.append({
                "Model": model_name,
                "Class": cls_name,
                "Precision": report_dict[cls_name]["precision"],
                "Recall": report_dict[cls_name]["recall"],
                "F1-Score": report_dict[cls_name]["f1-score"],
                "Support": report_dict[cls_name]["support"]
            })

    print(f"\n===== {model_name} =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC:   {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))


# =========================================================
# 9. Save Main Results Table
# =========================================================
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
).sort_values(by="Accuracy", ascending=False)

print("\nModel Comparison:\n")
print(results_df)

results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results_extended.csv"), index=False)

classwise_df = pd.DataFrame(classification_reports)
classwise_df.to_csv(os.path.join(OUTPUT_DIR, "classwise_classification_report.csv"), index=False)


# =========================================================
# 10. Accuracy Comparison Plot
# =========================================================
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("Comparison of ML Model Accuracies")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_accuracy_comparison.png"), dpi=300)
plt.show()


# =========================================================
# 11. Precision / Recall / F1 Comparison Plot
# =========================================================
x = np.arange(len(results_df))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, results_df["Precision"], width, label="Precision")
plt.bar(x, results_df["Recall"], width, label="Recall")
plt.bar(x + width, results_df["F1_Score"], width, label="F1 Score")
plt.xticks(x, results_df["Model"], rotation=45, ha="right")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_f1_comparison.png"), dpi=300)
plt.show()


# =========================================================
# 12. Cross-Validation
# =========================================================
cv_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n===== CROSS VALIDATION RESULTS =====")
for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring="accuracy")
    cv_mean = scores.mean()
    cv_std = scores.std()

    cv_results.append([model_name, cv_mean, cv_std])

    print(f"{model_name}: Mean Accuracy = {cv_mean:.4f}, Std = {cv_std:.4f}")

cv_results_df = pd.DataFrame(cv_results, columns=["Model", "CV_Mean_Accuracy", "CV_Std"])
cv_results_df = cv_results_df.sort_values(by="CV_Mean_Accuracy", ascending=False)
cv_results_df.to_csv(os.path.join(OUTPUT_DIR, "cross_validation_results.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.bar(cv_results_df["Model"], cv_results_df["CV_Mean_Accuracy"], yerr=cv_results_df["CV_Std"], capsize=5)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Cross-Validation Accuracy")
plt.title("5-Fold Cross-Validation Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation_accuracy.png"), dpi=300)
plt.show()


# =========================================================
# 13. Best Model Selection
# =========================================================
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

print(f"\nBest Model: {best_model_name}")

y_pred_best = best_pipeline.predict(X_test)


# =========================================================
# 14. Confusion Matrix for Best Model
# =========================================================
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png"), dpi=300)
plt.show()


# =========================================================
# 15. ROC Curves for Best Model (Multiclass One-vs-Rest)
# =========================================================
if hasattr(best_pipeline.named_steps["classifier"], "predict_proba"):
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    y_score_best = best_pipeline.predict_proba(X_test)

    fpr = {}
    tpr = {}
    roc_auc_dict = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_best[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{label_encoder.classes_[i]} (AUC = {roc_auc_dict[i]:.2f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multiclass ROC Curves - {best_model_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_model_multiclass_roc_curve.png"), dpi=300)
    plt.show()


# =========================================================
# 16. Feature Importance
# =========================================================
best_classifier = best_pipeline.named_steps["classifier"]

if hasattr(best_classifier, "feature_importances_"):
    importances = best_classifier.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": numeric_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Feature Importances:\n", feature_importance_df.head(10))
    feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 6))
    top_10 = feature_importance_df.head(10)
    plt.barh(top_10["Feature"][::-1], top_10["Importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top 10 Important Features - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_top10.png"), dpi=300)
    plt.show()


# =========================================================
# 17. PCA Visualization
# =========================================================
X_processed = preprocessor.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Continent": y.values
})

plt.figure(figsize=(10, 6))
for continent in pca_df["Continent"].unique():
    subset = pca_df[pca_df["Continent"] == continent]
    plt.scatter(subset["PC1"], subset["PC2"], label=continent, alpha=0.7)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Global Energy Data")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_projection.png"), dpi=300)
plt.show()


# =========================================================
# 18. Correlation Heatmap
# =========================================================
corr = df[numeric_features].corr()

plt.figure(figsize=(12, 8))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.show()


# =========================================================
# 19. Class Distribution
# =========================================================
class_counts = y.value_counts()

plt.figure(figsize=(8, 5))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel("Continent")
plt.ylabel("Count")
plt.title("Class Distribution of Continents")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=300)
plt.show()


# =========================================================
# 20. Feature Histograms
# =========================================================
top_features_for_hist = numeric_features[:6]

df[top_features_for_hist].hist(figsize=(14, 8), bins=20)
plt.suptitle("Distribution of Selected Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_histograms.png"), dpi=300)
plt.show()


# =========================================================
# 21. Learning Curve for Best Model
# =========================================================
train_sizes, train_scores, test_scores = learning_curve(
    best_pipeline,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
plt.plot(train_sizes, test_mean, marker="o", label="Validation Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title(f"Learning Curve - {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve_best_model.png"), dpi=300)
plt.show()

learning_curve_df = pd.DataFrame({
    "Train_Size": train_sizes,
    "Train_Accuracy_Mean": train_mean,
    "Train_Accuracy_Std": train_std,
    "Validation_Accuracy_Mean": test_mean,
    "Validation_Accuracy_Std": test_std
})
learning_curve_df.to_csv(os.path.join(OUTPUT_DIR, "learning_curve_results.csv"), index=False)


# =========================================================
# 22. Hyperparameter Tuning (Random Forest + Extra Trees)
# =========================================================
tuning_results = []

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_param_dist = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_dist,
    n_iter=10,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

tuning_results.append({
    "Model": "Random Forest",
    "Best_Params": str(rf_search.best_params_),
    "Best_CV_Score": rf_search.best_score_
})

et_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", ExtraTreesClassifier(random_state=42))
])

et_param_dist = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

et_search = RandomizedSearchCV(
    estimator=et_pipeline,
    param_distributions=et_param_dist,
    n_iter=10,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1
)

et_search.fit(X_train, y_train)

tuning_results.append({
    "Model": "Extra Trees",
    "Best_Params": str(et_search.best_params_),
    "Best_CV_Score": et_search.best_score_
})

tuning_results_df = pd.DataFrame(tuning_results)
tuning_results_df.to_csv(os.path.join(OUTPUT_DIR, "hyperparameter_tuning_results.csv"), index=False)

print("\n===== HYPERPARAMETER TUNING RESULTS =====")
print(tuning_results_df)


# =========================================================
# 23. Save Best Model Predictions
# =========================================================
predicted_labels = label_encoder.inverse_transform(y_pred_best)
actual_labels = label_encoder.inverse_transform(y_test)

predictions_df = X_test.copy()
predictions_df["Actual_Continent"] = actual_labels
predictions_df["Predicted_Continent"] = predicted_labels

predictions_df.to_csv(os.path.join(OUTPUT_DIR, "best_model_predictions.csv"), index=False)


# =========================================================
# 24. Final Output Summary
# =========================================================
print("\nFiles saved successfully inside 'output' folder:")
print("- model_comparison_results_extended.csv")
print("- classwise_classification_report.csv")
print("- model_accuracy_comparison.png")
print("- precision_recall_f1_comparison.png")
print("- cross_validation_results.csv")
print("- cross_validation_accuracy.png")
print("- best_model_confusion_matrix.png")
print("- best_model_multiclass_roc_curve.png")
print("- feature_importance.csv")
print("- feature_importance_top10.png")
print("- pca_projection.png")
print("- correlation_heatmap.png")
print("- class_distribution.png")
print("- feature_histograms.png")
print("- learning_curve_best_model.png")
print("- learning_curve_results.csv")
print("- hyperparameter_tuning_results.csv")
print("- best_model_predictions.csv")
