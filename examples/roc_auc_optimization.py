"""
ROC AUC Optimization
"""

from eosdxanalysis.models.clustering import image_normality_batch

print("Normality distance threshold, ROC AUC optimization")

step = 0.1
threshold_min = 0
threshold_max = 10
threshold_range = np.arange(threshold_min, threshold_max, step)

print("distance_threshold,roc_auc")
for idx in range(distance_threshold_range.size):
    # Set distance threshold
    distance_threshold = distance_threshold_range[idx]
    # Calculate predictions
    y_true, y_score = image_normality_batch(
        patients_db=patients_db,
        source_data_path=source_data_path,
        distance_threshold=distance_threshold,
        image_area=image_area,
        image_mask=image_mask)
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_true, y_score)
    # Print csv results
    print("{:0.2f}".format(distance_threshold), end=",")
    print("{:0.2f}".format(roc_auc))
