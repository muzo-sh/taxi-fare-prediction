# taxi-fare-prediction

1. DESCRIPTION OF THE CHOSEN DATASET
* Dataset Title: Taxi Trip Fare Data
* Source Link: https://www.kaggle.com/datasets/hrish4/taxi-trip-fare-data-2023
* Objective: This project resolves a regression task. The primary objective is
  to build a machine learning pipeline capable of accurately predicting the
  total fare amount of an urban taxi trip based on journey parameters.
* Features Profile: The records evaluate metrics including total trip distance,
  passenger count, trip duration in minutes, payment methods, and historical
  base fare tracking patterns.

2. STATISTICAL ANALYSIS OF THE DATASET
* Exploratory Finding Summaries:
  - Missing Values: Programmatic verification returned 0 null records across
    the active columns, establishing a complete profile needing no structural drop rows.
  - Distributions (Histograms): Data plots isolate clear right-skewed profiles for
    trip distance and total fares. A massive portion of transit profiles map to short
    urban movements, while high-fare occurrences populate an extended tail.
  - Correlation Heatmap: The statistical analysis confirms that trip duration and
    trip distance display the strongest linear correlation values relative to the final
    fare payouts.
* Analysis Conclusion: Simple linear lines might miscalculate compound urban elements
  like localized congestion adjustments. Therefore, a robust non-linear multi-tree
  ensemble strategy provides the highest logic fit.

3. MACHINE LEARNING ALGORITHM CHOICE
* Algorithm: Random Forest Regressor
* Justification: Random Forest works wonderfully for transit pricing because it builds
  an ensemble of multiple independent decision trees. It smoothly blends numerical features
  with encoded categorical values, reduces the risk of structural overfitting, and naturally
  maps non-linear splits without requiring massive external data modifications.

4. PARAMETER DETERMINATION & OPTIMIZATION
* Strategy: Rather than choosing arbitrary settings, parameters were optimized
  programmatically via Scikit-Learn's Grid Search with 3-Fold Cross-Validation.
* Hyperparameter Matrix Checked:
  - 'n_estimators' (Number of decision trees): [50, 100]
  - 'max_depth' (Maximum allowed tree depth): [4, 6]
* Selection Optimization Result: The cross-validation engine isolated a depth cap
  of 6 running 50 distinct tree models to deliver the highest predictive R2 accuracy.
* Tutorial Disclosures: The script structures were written cleanly utilizing core
  documentation modules without directly copying online code templates.

5. RESULTS & PERFORMANCE ANALYSIS
* Quantitative Evaluation Output:
  - R2 Score: ~0.9818 (The model explains approximately 98% of the fare variances
    on the unseen testing partition).
  - Mean Absolute Error (MAE): 0.50 units (On average, the model's predictions
    deviate by a mere 0.50 units of currency per ride).
* Output Chart Evaluation: The generated actual-vs-predicted scatter validation plot
  ('results_chart.png') shows data points tightly gathering along the primary diagonal
  reference axis, proving excellent prediction stability across urban trips.

6. FINAL CONCLUSIONS
* Approach Viability: The project setup is highly successful. Achieving an R2 score
  of 98.18% using basic trip tracking inputs proves the model is exceptionally reliable.
* Result Justification: Performance remains strong because the grid search accurately
  capped depth boundaries, allowing the algorithm to properly isolate variations without
  overfitting to random anomalies in the training set.
* Future Improvement Plans: Accuracy could be enhanced further by engineering new feature
  interactions, such as cross-multiplying trip distance by time of day to create an
  approximate "traffic density factor."

"""
