
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)


df = pd.read_csv("C:\\Users\\Vishnu Prahalathan\\Desktop\\heart.csv")  


print("🧹 Missing values:\n", df.isnull().sum())


X = df.drop("target", axis=1)
y = df["target"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=df.columns[:-1], class_names=["No HD", "HD"])
plt.title("📊 Decision Tree Visualization")
plt.show()


train_acc = []
test_acc = []
depths = range(1, 21)

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc.append(dt.score(X_train, y_train))
    test_acc.append(dt.score(X_test, y_test))

plt.plot(depths, train_acc, label='Train Accuracy')
plt.plot(depths, test_acc, label='Test Accuracy')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("📉 Overfitting Analysis")
plt.grid(True)
plt.show()


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"\n🔍 Evaluation: {name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_true, y_proba):.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, y_proba):.2f})")

evaluate_model("Decision Tree", y_test, y_pred_dt, dt_model.predict_proba(X_test)[:, 1])

evaluate_model("Random Forest", y_test, y_pred_rf, rf_model.predict_proba(X_test)[:, 1])


plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("🎯 ROC Curves")
plt.legend()
plt.grid(True)
plt.show()

importances = rf_model.feature_importances_
features = df.columns[:-1]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=features)
plt.title("💡 Feature Importances (Random Forest)")
plt.show()


dt_cv = cross_val_score(dt_model, X_scaled, y, cv=5)
rf_cv = cross_val_score(rf_model, X_scaled, y, cv=5)

print("\n📈 Decision Tree CV Accuracy: {:.2f}%".format(np.mean(dt_cv) * 100))
print("📈 Random Forest CV Accuracy: {:.2f}%".format(np.mean(rf_cv) * 100))
