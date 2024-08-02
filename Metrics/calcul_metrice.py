def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1_score

# Total apples in original masks
T_original = 2940

# Total apples in segmented images
T_unet = 2609
T_svm = 3258
T_svm_crf = 3115

# Calculate TP, FP, FN for U-Net
TP_unet = T_unet
FP_unet = max(T_unet - T_original, 0)
FN_unet = max(T_original - T_unet, 0)

# Calculate TP, FP, FN for SVM
TP_svm = T_svm
FP_svm = max(T_svm - T_original, 0)
FN_svm = max(T_original - T_svm, 0)

# Calculate TP, FP, FN for SVM + CRF
TP_svm_crf = T_svm_crf
FP_svm_crf = max(T_svm_crf - T_original, 0)
FN_svm_crf = max(T_original - T_svm_crf, 0)

# Calculate metrics for U-Net
precision_unet, recall_unet, f1_unet = calculate_metrics(TP_unet, FP_unet, FN_unet)
print(f"U-Net - Precision: {precision_unet}, Recall: {recall_unet}, F1 Score: {f1_unet}")
print(f"TP: {TP_unet}, FP: {FP_unet}, FN: {FN_unet}")
TP_svm = 2940
TP_svm_crf = 2940
# Calculate metrics for SVM
precision_svm, recall_svm, f1_svm = calculate_metrics(TP_svm, FP_svm, FN_svm)
print(f"SVM - Precision: {precision_svm}, Recall: {recall_svm}, F1 Score: {f1_svm}")
print(f"TP: {TP_svm}, FP: {FP_svm}, FN: {FN_svm}")

# Calculate metrics for SVM + CRF
precision_svm_crf, recall_svm_crf, f1_svm_crf = calculate_metrics(TP_svm_crf, FP_svm_crf, FN_svm_crf)
print(f"SVM + CRF - Precision: {precision_svm_crf}, Recall: {recall_svm_crf}, F1 Score: {f1_svm_crf}")
print(f"TP: {TP_svm_crf}, FP: {FP_svm_crf}, FN: {FN_svm_crf}")