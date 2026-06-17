# Trend and Time Series Analysis

When a user asks about trends, seasonality, growth rate, or how a metric
changed over time, resample to a sensible frequency based on the date
span: daily for short spans, weekly for medium spans, monthly for spans
over roughly a year. Report the overall direction using a fitted trend
line rather than the raw first and last values, since a partial first or
last bucket can make a simple endpoint comparison misleading. Always call
out the peak and trough periods alongside the overall direction.
