import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   #designing charts
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


#stage1 data loading
df = pd.read_csv('Taxi_Trip_Data_preprocessed.csv')
df = df.sample(n=10000, random_state=42).reset_index(drop=True) #it reads 10k instead whole to save time.

print("--- Dataset Structure & Data Types ---")
print(df.info())
print("\n--- Statistical Summary of Key Features ---")
print(df.describe())
print("\n--- Checking for Missing Values (Nulls) ---")
print(df.isnull().sum())

#histogram analiz
df.hist(bins=20, figsize=(10, 8))
plt.suptitle("Feature Distributions (Histograms)")
plt.savefig('histograms.png')
plt.close()

#generate matrix heatmap.
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()


df_cleaned = df.dropna()  #stage2 cleaning data
# converts card cash to 0,1 onehot encode
df_cleaned = pd.get_dummies(df_cleaned, drop_first=True)

#stage 3 standarization data split

# seperates the data x and y
X = df_cleaned.drop('fare_amount', axis=1)
y = df_cleaned['fare_amount']

#split data %80 train %20 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#puts the data value 0 between 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#stage 4 mL randomforest regression

#crossvalidation
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6]
}

# 3foldcross validation, test,setacross select accruate.
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    verbose=2
)
grid_search.fit(X_train_scaled, y_train)
#identify the best one
best_model = grid_search.best_estimator_
print("\n--- Programmatically Optimized Parameters ---")
print(grid_search.best_params_)

#stage5 results
#unseen predictions
y_pred = best_model.predict(X_test_scaled)

# alculates
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Final Model Metrics ---")
print(f"R2 Score (Variance Fit Accuracy): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} units")

# generate scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')  #final validation charting fares
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Fares')
plt.ylabel('Predicted Fares')
plt.title('Actual vs. Predicted Taxi Trip Fares')
plt.tight_layout()
plt.savefig('results_chart.png')
plt.close()

print("\nExecution complete. Statistical images and results chart saved successfully.")