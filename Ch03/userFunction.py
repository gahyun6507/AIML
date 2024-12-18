from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)  # 'accuaracy'를 'accuracy'로 수정
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # pred_proba가 None일 가능성을 대비한 조건 추가
    if pred_proba is not None:
        roc_auc = roc_auc_score(y_test, pred_proba)
    else:
        roc_auc = None

    print("오차행렬")
    print(confusion)
    print(f"정확도 : {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, F1 : {f1:.4f}")
    if roc_auc is not None:
        print(f"AUC : {roc_auc:.4f}")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve
import numpy as np

def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, threshold = precision_recall_curve(y_test, pred_proba_c1)
    plt.figure(figsize=(5, 4))
    threshold_boundary = threshold.shape[0]
    plt.plot(threshold, precisions[0:threshold_boundary], linestyle='--', label="precision")
    plt.plot(threshold, recalls[0:threshold_boundary], label="recall")
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlabel("Threshold value");plt.ylabel("Precision and Recall value")
    plt.legend();plt.grid()
    plt.show()


from sklearn.preprocessing import Binarizer

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print("\n임계값 : ", custom_threshold)
        get_clf_eval(y_test, custom_predict, pred_proba_c1)