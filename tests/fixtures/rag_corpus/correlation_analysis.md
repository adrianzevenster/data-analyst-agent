# Interpreting Correlations and Associations

When a user asks what correlates with what, or which columns are related,
report Pearson correlation for numeric-numeric pairs and the correlation
ratio (eta) for categorical-numeric pairs rather than trying to coerce
categories into numbers. Only call a relationship "strong" above roughly
0.7 absolute correlation or 0.5 correlation ratio, and always check the
pairwise row overlap before trusting a correlation computed on a small,
unrepresentative subset of the data.
