# Clustering and Segmentation

When a user asks to cluster, segment, or group similar rows, use KMeans on
standardized numeric columns and default to k=5 unless the user names a
specific number of segments. Warn that KMeans assumes roughly spherical,
similarly-sized clusters, and that categorical columns need encoding or
exclusion before clustering. Summarize each cluster by its size and the
mean of each numeric feature so the user can see what distinguishes the
segments, not just which row landed in which cluster.
