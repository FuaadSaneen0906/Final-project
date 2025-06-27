# Final-project
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
file_path = r"C:\Users\sanee\PycharmProjects\pythonstudy\HR_Analytics_Sanket.xlsx"
df = pd.read_excel(file_path, sheet_name="HR Data", header=2)

# Clean columns
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Attrition"])
df = df[df["Attrition"].isin(["Yes", "No"])]
df["Attrition_Y"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Split X and y
X = df.drop(columns=["Attrition", "Attrition_Y", "EmployeeNumber", "Unnamed: 0"], errors='ignore')
y = df["Attrition_Y"]

# One-hot encoding
X = pd.get_dummies(X)

X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# SHAP explanation
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

from fpdf import FPDF
import matplotlib.pyplot as plt
import shap
import os

# ========== Save SHAP plot ==========
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
shap_plot_path = "shap_summary_plot.png"
plt.savefig(shap_plot_path, bbox_inches='tight')
plt.close()

# ========== Save classification metrics ==========
from sklearn.metrics import classification_report, confusion_matrix

report_text = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Generate confusion matrix plot
import seaborn as sns
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
conf_matrix_path = "conf_matrix.png"
plt.savefig(conf_matrix_path, bbox_inches='tight')
plt.close()

# ========== Create PDF ==========
pdf = FPDF()
pdf.add_page()

# Title
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "HR Analytics - Attrition Prediction Report", ln=True)

# Classification Report
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "1. Classification Report", ln=True)
pdf.set_font("Arial", '', 10)
for line in report_text.split('\n'):
    pdf.cell(0, 5, line.strip(), ln=True)

# Confusion Matrix Image
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "2. Confusion Matrix", ln=True)
pdf.image(conf_matrix_path, w=100)

# SHAP Summary Plot
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "3. SHAP Feature Importance", ln=True)
pdf.image(shap_plot_path, w=150)

# Insights (example)
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "4. Key Insights", ln=True)
pdf.set_font("Arial", '', 10)
pdf.multi_cell(0, 5,
    "- The model achieved a balanced performance across both classes.\n"
    "- Overtime, JobRole, and MonthlyIncome are among the top features influencing attrition.\n"
    "- SHAP reveals that higher Overtime and lower Work-Life Balance strongly correlate with attrition."
)

pdf_path = "HR_Analytics_Report.pdf"
pdf.output(pdf_path)

print(f"\n✅ PDF report saved as: {pdf_path}")

import os
print("PDF saved at:", os.path.abspath("HR_Analytics_Report.pdf"))
