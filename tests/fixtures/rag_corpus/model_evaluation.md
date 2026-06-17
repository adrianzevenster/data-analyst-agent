# Evaluating Model Predictions

When a user asks to evaluate model performance, predictions, or accuracy,
first detect whether the task is classification or regression from the
actual/prediction column dtypes and cardinality rather than asking the
user. For classification, report accuracy, precision, recall, and ROC-AUC
when a probability column is present; skip ROC-AUC if predictions cover
only one class. For regression or forecasting, prefer WMAPE over plain
MAPE so a few near-zero actual values don't blow up the error metric.
