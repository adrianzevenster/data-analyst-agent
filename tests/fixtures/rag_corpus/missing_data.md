# Handling Missing Values

When a user asks about missing values, nulls, NaNs, or incomplete data,
start with the missingness matrix to rank columns by missing percentage
before recommending any imputation. Don't suggest dropping a column unless
its missing rate exceeds roughly 50%, since moderate missingness is often
still informative. Mention whether the missingness looks structural (e.g.
concentrated in one category) rather than random, since that changes
whether imputation or a missing-value indicator column is more appropriate.
