import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

# ── STEP 1: Load Data ──────────────────────────────────────
df = pd.read_csv('heart.csv')
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])
feature_names = list(X.columns)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]-1} features")
print(f"HeartDisease balance: {y.value_counts().to_dict()}")

# ── STEP 2: Train/Test Split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── STEP 3: Scale Features ─────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Scaling done")

# ── STEP 4: Train Model ────────────────────────────────────
model = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs',
                           max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model trained!")

# ── STEP 5: Evaluate ───────────────────────────────────────
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print(f"\n📊 RESULTS:")
print(f"  Accuracy  : {accuracy*100:.1f}%")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")

# ── STEP 6: Confusion Matrix ───────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease','Has Disease'],
            yticklabels=['No Disease','Has Disease'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Saved: confusion_matrix.png")

# ── STEP 7: Cross Validation ───────────────────────────────
pipe = Pipeline([('scaler', StandardScaler()),
                 ('model', LogisticRegression(max_iter=1000, random_state=42))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
print(f"\n5-Fold CV: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

# ── STEP 8: Feature Importance ─────────────────────────────
coefs = model.coef_[0]
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefs
}).reindex(pd.Series(np.abs(coefs), index=feature_names).sort_values(ascending=False).index)

plt.figure(figsize=(9,6))
colors = ['red' if c > 0 else 'green' for c in importance_df['Coefficient']]
plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("Saved: feature_importance.png")

# ── STEP 9: ROC Curve ──────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1],[0,1],'r--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
print("Saved: roc_curve.png")

# ── STEP 10: Save Model ────────────────────────────────────
model_bundle = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_names,
    'metrics': {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'cv_mean': round(cv_scores.mean(), 4),
        'cv_std': round(cv_scores.std(), 4)
    }
}
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

print("\n✅ Model saved as heart_disease_model.pkl")
print("✅ ALL DONE! Now run: streamlit run app.py")