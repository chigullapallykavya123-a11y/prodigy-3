import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
# Update the path based on where you downloaded/extracted the files
# Usually, bank-full.csv is located inside the 'bank' folder
df = pd.read_csv('bank-full.csv', sep=';')

# 2. Preprocessing
# Convert categorical variables into numbers (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# 3. Define Features (X) and Target (y)
# The target is likely the 'y_yes' column after encoding
X = df.drop('y_yes', axis=1)
y = df['y_yes']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build and train the Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))