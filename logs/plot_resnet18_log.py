import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(text):
    epoch, loss, acc, precision, recall, f1, time_ = [], [], [], [], [], [], []
    pattern = re.compile(
        r"Epoch \[(\d+)/\d+\]\s*\|\s*Loss ([0-9.]+)\s*\|\s*Accuracy ([0-9.]+)%\s*\|\s*Precision ([0-9.]+)\s*\|\s*Recall ([0-9.]+)\s*\|\s*F1-score ([0-9.]+)\s*\|\s*Inference Time\s+([0-9.]+)",
        re.S
    )
    for m in pattern.finditer(text):
        epoch.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        acc.append(float(m.group(3)))
        precision.append(float(m.group(4)))
        recall.append(float(m.group(5)))
        f1.append(float(m.group(6)))
        time_.append(float(m.group(7)))
    return np.array(epoch), np.array(loss), np.array(acc), np.array(precision), np.array(recall), np.array(f1), np.array(time_)

def plot_custom(model_name, log_text, save_dir):
    epoch, loss, acc, precision, recall, f1, time_ = parse_log(log_text)
    if len(epoch) == 0:
        print("[Error] 로그 파싱 실패. 형식을 확인하세요.")
        return
    
    best_idx = np.argmax(f1)
    best_ep, best_f1 = epoch[best_idx], f1[best_idx]
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(epoch, loss)
    plt.scatter(best_ep, loss[best_idx], color="red")
    plt.title("Loss")
    
    plt.subplot(2, 3, 2)
    plt.plot(epoch, acc)
    plt.scatter(best_ep, acc[best_idx], color="red")
    plt.title("Accuracy (%)")
    
    plt.subplot(2, 3, 3)
    plt.plot(epoch, precision, label="Precision")
    plt.plot(epoch, recall, label="Recall")
    plt.scatter(best_ep, precision[best_idx], color="red")
    plt.scatter(best_ep, recall[best_idx], color="red")
    plt.legend()
    plt.title("Precision / Recall")
    
    plt.subplot(2, 3, 4)
    plt.plot(epoch, f1)
    plt.scatter(best_ep, best_f1, color="red")
    plt.text(best_ep, best_f1, f"BEST\nEp {best_ep}", fontsize=9)
    plt.title("F1-score")
    
    plt.subplot(2, 3, 5)
    plt.plot(epoch, time_)
    plt.scatter(best_ep, time_[best_idx], color="red")
    plt.title("Inference Time (min)")
    
    plt.suptitle(f"{model_name} Training Result", fontsize=14)
    save_path = os.path.join(save_dir, "result.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"[Saved] {save_path}")
    print(f"[BEST] Epoch {best_ep}, F1 {best_f1:.4f}")

# ==================================================
# 📁 ResNet18 로그 데이터 세션
# ==================================================
resnet_log = """
Epoch [01/50] | Loss 0.1739 | Accuracy 93.25% | Precision 0.6095 | Recall 0.6718 | F1-score 0.6207 | Inference Time 0.0786 min
Epoch [02/50] | Loss 0.1185 | Accuracy 96.66% | Precision 0.8205 | Recall 0.7806 | F1-score 0.7762 | Inference Time 0.0783 min
Epoch [03/50] | Loss 0.1000 | Accuracy 96.63% | Precision 0.8491 | Recall 0.7703 | F1-score 0.8025 | Inference Time 0.0782 min
Epoch [04/50] | Loss 0.0848 | Accuracy 97.02% | Precision 0.8528 | Recall 0.7702 | F1-score 0.7978 | Inference Time 0.0784 min
Epoch [05/50] | Loss 0.0734 | Accuracy 95.87% | Precision 0.7553 | Recall 0.8277 | F1-score 0.7785 | Inference Time 0.0784 min
Epoch [06/50] | Loss 0.0619 | Accuracy 97.05% | Precision 0.8804 | Recall 0.7334 | F1-score 0.7896 | Inference Time 0.0785 min
Epoch [07/50] | Loss 0.0507 | Accuracy 97.27% | Precision 0.8774 | Recall 0.7607 | F1-score 0.7863 | Inference Time 0.0784 min
Epoch [08/50] | Loss 0.0415 | Accuracy 97.33% | Precision 0.8866 | Recall 0.8188 | F1-score 0.8454 | Inference Time 0.0784 min
Epoch [09/50] | Loss 0.0333 | Accuracy 96.84% | Precision 0.8382 | Recall 0.8364 | F1-score 0.8335 | Inference Time 0.0783 min
Epoch [10/50] | Loss 0.0282 | Accuracy 97.24% | Precision 0.8760 | Recall 0.8410 | F1-score 0.8577 | Inference Time 0.0783 min
Epoch [11/50] | Loss 0.0247 | Accuracy 97.20% | Precision 0.8771 | Recall 0.8421 | F1-score 0.8578 | Inference Time 0.0783 min
Epoch [12/50] | Loss 0.0216 | Accuracy 97.13% | Precision 0.8890 | Recall 0.7909 | F1-score 0.8345 | Inference Time 0.0783 min
Epoch [13/50] | Loss 0.0200 | Accuracy 97.28% | Precision 0.8880 | Recall 0.8607 | F1-score 0.8720 | Inference Time 0.0784 min
Epoch [14/50] | Loss 0.0187 | Accuracy 97.05% | Precision 0.8642 | Recall 0.8519 | F1-score 0.8573 | Inference Time 0.0785 min
Epoch [15/50] | Loss 0.0165 | Accuracy 97.25% | Precision 0.8790 | Recall 0.8295 | F1-score 0.8505 | Inference Time 0.0783 min
Epoch [16/50] | Loss 0.0155 | Accuracy 97.05% | Precision 0.8567 | Recall 0.8470 | F1-score 0.8502 | Inference Time 0.0783 min
Epoch [17/50] | Loss 0.0145 | Accuracy 96.96% | Precision 0.8741 | Recall 0.8464 | F1-score 0.8556 | Inference Time 0.0783 min
Epoch [18/50] | Loss 0.0144 | Accuracy 97.11% | Precision 0.8976 | Recall 0.8099 | F1-score 0.8369 | Inference Time 0.0783 min
Epoch [19/50] | Loss 0.0144 | Accuracy 96.94% | Precision 0.8462 | Recall 0.8456 | F1-score 0.8434 | Inference Time 0.0785 min
Epoch [20/50] | Loss 0.0128 | Accuracy 96.67% | Precision 0.8636 | Recall 0.8294 | F1-score 0.8417 | Inference Time 0.0783 min
Epoch [21/50] | Loss 0.0139 | Accuracy 96.79% | Precision 0.8411 | Recall 0.8411 | F1-score 0.8369 | Inference Time 0.0783 min
Epoch [22/50] | Loss 0.0120 | Accuracy 96.82% | Precision 0.8601 | Recall 0.8210 | F1-score 0.8388 | Inference Time 0.0783 min
Epoch [23/50] | Loss 0.0124 | Accuracy 97.19% | Precision 0.8874 | Recall 0.8449 | F1-score 0.8634 | Inference Time 0.0784 min
Epoch [24/50] | Loss 0.0125 | Accuracy 96.87% | Precision 0.8599 | Recall 0.8339 | F1-score 0.8458 | Inference Time 0.0785 min
Epoch [25/50] | Loss 0.0114 | Accuracy 96.74% | Precision 0.8398 | Recall 0.8502 | F1-score 0.8434 | Inference Time 0.0784 min
Epoch [26/50] | Loss 0.0114 | Accuracy 97.22% | Precision 0.8630 | Recall 0.8445 | F1-score 0.8512 | Inference Time 0.0783 min
Epoch [27/50] | Loss 0.0115 | Accuracy 97.19% | Precision 0.9145 | Recall 0.8150 | F1-score 0.8543 | Inference Time 0.0783 min
Epoch [28/50] | Loss 0.0105 | Accuracy 97.17% | Precision 0.8655 | Recall 0.8351 | F1-score 0.8450 | Inference Time 0.0783 min
Epoch [29/50] | Loss 0.0112 | Accuracy 97.15% | Precision 0.8903 | Recall 0.8251 | F1-score 0.8535 | Inference Time 0.0783 min
Epoch [30/50] | Loss 0.0104 | Accuracy 97.02% | Precision 0.8685 | Recall 0.8338 | F1-score 0.8487 | Inference Time 0.0782 min
Epoch [31/50] | Loss 0.0103 | Accuracy 97.08% | Precision 0.8874 | Recall 0.8196 | F1-score 0.8469 | Inference Time 0.0783 min
Epoch [32/50] | Loss 0.0105 | Accuracy 96.82% | Precision 0.8528 | Recall 0.8422 | F1-score 0.8371 | Inference Time 0.0783 min
Epoch [33/50] | Loss 0.0103 | Accuracy 97.02% | Precision 0.8959 | Recall 0.8056 | F1-score 0.8441 | Inference Time 0.0784 min
Epoch [34/50] | Loss 0.0109 | Accuracy 96.99% | Precision 0.8649 | Recall 0.8432 | F1-score 0.8524 | Inference Time 0.0791 min
Epoch [35/50] | Loss 0.0098 | Accuracy 97.07% | Precision 0.8604 | Recall 0.8374 | F1-score 0.8470 | Inference Time 0.0784 min
Epoch [36/50] | Loss 0.0101 | Accuracy 96.87% | Precision 0.8678 | Recall 0.8157 | F1-score 0.8318 | Inference Time 0.0783 min
Epoch [37/50] | Loss 0.0101 | Accuracy 97.25% | Precision 0.8968 | Recall 0.8229 | F1-score 0.8535 | Inference Time 0.0802 min
Epoch [38/50] | Loss 0.0099 | Accuracy 96.99% | Precision 0.8810 | Recall 0.8173 | F1-score 0.8453 | Inference Time 0.0798 min
Epoch [39/50] | Loss 0.0102 | Accuracy 96.23% | Precision 0.8188 | Recall 0.8627 | F1-score 0.8326 | Inference Time 0.0798 min
Epoch [40/50] | Loss 0.0093 | Accuracy 97.02% | Precision 0.8586 | Recall 0.8428 | F1-score 0.8449 | Inference Time 0.0783 min
Epoch [41/50] | Loss 0.0101 | Accuracy 97.10% | Precision 0.9006 | Recall 0.8227 | F1-score 0.8575 | Inference Time 0.0798 min
Epoch [42/50] | Loss 0.0094 | Accuracy 96.89% | Precision 0.8318 | Recall 0.8483 | F1-score 0.8313 | Inference Time 0.0797 min
Epoch [43/50] | Loss 0.0099 | Accuracy 97.20% | Precision 0.9055 | Recall 0.8301 | F1-score 0.8569 | Inference Time 0.0797 min
Epoch [44/50] | Loss 0.0097 | Accuracy 97.02% | Precision 0.8727 | Recall 0.8346 | F1-score 0.8526 | Inference Time 0.0796 min
Epoch [45/50] | Loss 0.0098 | Accuracy 97.30% | Precision 0.8893 | Recall 0.8462 | F1-score 0.8622 | Inference Time 0.0797 min
Epoch [46/50] | Loss 0.0098 | Accuracy 97.04% | Precision 0.8612 | Recall 0.8332 | F1-score 0.8441 | Inference Time 0.0794 min
Epoch [47/50] | Loss 0.0087 | Accuracy 97.06% | Precision 0.8904 | Recall 0.8194 | F1-score 0.8514 | Inference Time 0.0782 min
Epoch [48/50] | Loss 0.0100 | Accuracy 96.92% | Precision 0.8560 | Recall 0.8454 | F1-score 0.8496 | Inference Time 0.0800 min
Epoch [49/50] | Loss 0.0096 | Accuracy 96.79% | Precision 0.8523 | Recall 0.8509 | F1-score 0.8508 | Inference Time 0.0798 min
Epoch [50/50] | Loss 0.0092 | Accuracy 97.04% | Precision 0.8843 | Recall 0.8309 | F1-score 0.8562 | Inference Time 0.0798 min
"""

plot_custom("ResNet18", resnet_log, "logs/resnet18")
