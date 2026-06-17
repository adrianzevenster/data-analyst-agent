# Outlier and Anomaly Detection

When a user asks to find outliers, unusual values, or anomalies in a dataset,
prefer the IsolationForest-based anomaly scan over simple z-score thresholds
for multivariate data, since z-scores only catch outliers one column at a
time. Run the scan on numeric columns only, and exclude boolean/id-like
columns since they produce meaningless outlier flags. A reasonable default
contamination rate is 1-5% of rows; raise it only if the user explicitly
says outliers are common in this dataset.
