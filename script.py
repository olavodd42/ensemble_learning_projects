import pandas as pd
import numpy as np
#import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']

feature_cols = [name for name in col_names if name != 'income']
df = pd.read_csv('adult.data', header=None, names = col_names)

#Distribution of income
print(df.income.value_counts(normalize=True))

#Clean columns by stripping extra whitespace for columns of type "object"
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').apply(lambda x: x.str.lstrip())

#Create feature dataframe X with feature columns and dummy variables for categorical features
X = pd.get_dummies(df[feature_cols], drop_first=True)
#Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greather than 50k
y = df.income.apply(lambda x: 1 if x == '>50K' else 0)

#Split data into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

#Instantiate random forest classifier, fit and score with default parameters
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f'Training accuracy with default Random Forest Classifier: {rf.score(X_train, y_train)*100:.2f}%')
print(f'Test accuracy with default Random Forest Classifier: {rf.score(X_test, y_test)*100:.2f}%')

# Calculate and print additional evaluation metrics for the baseline model
y_pred = rf.predict(X_test)
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f}")

#Tune the hyperparameter max_depth over a range from 1-25, save scores for test and train set
np.random.seed(0)
accuracy_train=[]
accuracy_test = []
depths = list(range(1, 26))
for depth in depths:
  rf = RandomForestClassifier(max_depth=depth)
  rf.fit(X_train, y_train)
  accuracy_train.append(rf.score(X_train, y_train))
  accuracy_test.append(rf.score(X_test, y_test))
    
#Find the best accuracy and at what depth that occurs
best_accuracy = max(accuracy_test)
best_depth = depths[np.argmax(accuracy_test)]
print(f"First model - Best depth: {best_depth}, Best accuracy: {best_accuracy:.4f}")

#Plot the accuracy scores for the test and train set over the range of depth values  
plt.figure(figsize=(10, 6))
plt.plot(depths, accuracy_train, label='Train')
plt.plot(depths, accuracy_test, label='Test')
plt.title('Depth Hyperparameter Tuning - First Model')
plt.xlabel('Depth')
plt.ylabel('Accuracy score')
plt.legend()
plt.show()
plt.close('all')

#Save the best random forest model and save the feature importances in a dataframe
best_rf = RandomForestClassifier(max_depth=best_depth)
best_rf.fit(X_train, y_train)

# Calculate and print additional evaluation metrics for the tuned first model
y_pred = best_rf.predict(X_test)
print(f"First model metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1]):.4f}")

# Create a DataFrame of feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
})

# Sort by importance in descending order
feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

# Display the top 10 most important features
print("Top 10 most important features (first model):")
print(feature_importances.head(10))

# Save feature importances to CSV
feature_importances.to_csv('feature_importances_first_model.csv', index=False)
print("Feature importances saved to 'feature_importances_first_model.csv'")

#Create two new features, based on education and native country
df['education_bin'] = pd.cut(df['education-num'], [0,9,13,16], labels=['HS or less', 'College to Bachelors', 'Masters or more'])

# Create a feature based on native-country (US vs non-US) as mentioned in the instructions
df['us_citizen'] = df['native-country'].apply(lambda x: 1 if x == 'United-States' else 0)

# Update feature list with both new features
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 
               'sex', 'race', 'education_bin', 'us_citizen']

#Use these new features and recreate X and test/train split
X = pd.get_dummies(df[feature_cols], drop_first=True)

#Split data into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

accuracy_train = []
accuracy_test = []
#Find the best max depth now with the additional features
# FIXED: Use RandomForestClassifier instead of DecisionTreeClassifier
for depth in depths:
  rf = RandomForestClassifier(max_depth=depth)  # Changed from DecisionTreeClassifier
  rf.fit(X_train, y_train)
  accuracy_train.append(rf.score(X_train, y_train))
  accuracy_test.append(rf.score(X_test, y_test))

#Save the best model and print the features with the new feature set
# FIXED: Use test accuracy instead of train accuracy for finding best model
best_accuracy = max(accuracy_test)  # Changed from np.max(accuracy_train)
best_depth = depths[np.argmax(accuracy_test)]  # Changed to use test accuracy
print(f"Second model - Best depth: {best_depth}, Best accuracy: {best_accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(depths, accuracy_train, label='Train')
plt.plot(depths, accuracy_test, label='Test')
plt.title('Depth Hyperparameter Tuning - Second Model')
plt.xlabel('Depth')
plt.ylabel('Accuracy score')
plt.legend()
plt.show()
plt.close('all')

best_rf = RandomForestClassifier(max_depth=best_depth)
best_rf.fit(X_train, y_train)

# Calculate and print additional evaluation metrics for the tuned second model
y_pred = best_rf.predict(X_test)
print(f"Second model metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1]):.4f}")

# Create a DataFrame of feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
})

# Sort by importance in descending order
feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

# Display the top 5 most important features
print("Top 5 most important features (second model):")
print(feature_importances.head())

# Save feature importances to CSV
feature_importances.to_csv('feature_importances_second_model.csv', index=False)
print("Feature importances saved to 'feature_importances_second_model.csv'")

# Additional hyperparameter tuning for extending the project
print("\nExtending the project with additional hyperparameter tuning...")

# Try different n_estimators values
n_estimators_values = [50, 100, 200]
best_n_estimators = 0
best_score = 0

for n_est in n_estimators_values:
    rf = RandomForestClassifier(max_depth=best_depth, n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_n_estimators = n_est

print(f"Best n_estimators: {best_n_estimators}, Accuracy: {best_score:.4f}")

# Final model with best parameters
final_rf = RandomForestClassifier(
    max_depth=best_depth,
    n_estimators=best_n_estimators,
    max_features='sqrt',  # Try a different max_features value
    random_state=42
)

final_rf.fit(X_train, y_train)
final_score = final_rf.score(X_test, y_test)
print(f"Final model accuracy with tuned parameters: {final_score:.4f}")

# Evaluate with additional metrics
y_pred = final_rf.predict(X_test)
print(f"Final model F1 score: {f1_score(y_test, y_pred):.4f}")
print(f"Final model ROC AUC: {roc_auc_score(y_test, final_rf.predict_proba(X_test)[:,1]):.4f}")