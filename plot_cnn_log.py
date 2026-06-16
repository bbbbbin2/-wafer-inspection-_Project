import os
import re
import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# 로그 파서 (출력 포맷 전용)
# ==================================================
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

    return (
        np.array(epoch),
        np.array(loss),
        np.array(acc),
        np.array(precision),
        np.array(recall),
        np.array(f1),
        np.array(time_)
    )


# ==================================================
# best epoch 찾기
# ==================================================
def get_best_epoch(epoch, f1):
    best_idx = np.argmax(f1)
    return epoch[best_idx], f1[best_idx], best_idx


# ==================================================
# plotting (사용자 정의 스타일)
# ==================================================
def plot_custom(model_name, log_text, save_dir):

    epoch, loss, acc, precision, recall, f1, time_ = parse_log(log_text)

    if len(epoch) == 0:
        print("[Error] 로그 파싱에 실패했습니다. 텍스트 형식을 확인하세요.")
        return

    best_ep, best_f1, best_idx = get_best_epoch(epoch, f1)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(14, 8))

    # =========================
    # 1. Loss
    # =========================
    plt.subplot(2, 3, 1)
    plt.plot(epoch, loss)
    plt.scatter(best_ep, loss[best_idx], color="red")
    plt.title("Loss")

    # =========================
    # 2. Accuracy
    # =========================
    plt.subplot(2, 3, 2)
    plt.plot(epoch, acc)
    plt.scatter(best_ep, acc[best_idx], color="red")
    plt.title("Accuracy (%)")

    # =========================
    # 3. Precision / Recall
    # =========================
    plt.subplot(2, 3, 3)
    plt.plot(epoch, precision, label="Precision")
    plt.plot(epoch, recall, label="Recall")
    plt.scatter(best_ep, precision[best_idx], color="red")
    plt.scatter(best_ep, recall[best_idx], color="red")
    plt.legend()
    plt.title("Precision / Recall")

    # =========================
    # 4. F1-score (BEST 기준)
    # =========================
    plt.subplot(2, 3, 4)
    plt.plot(epoch, f1)
    plt.scatter(best_ep, best_f1, color="red")
    plt.text(best_ep, best_f1, f"BEST\nEp {best_ep}", fontsize=9)
    plt.title("F1-score")

    # =========================
    # 5. Inference Time
    # =========================
    plt.subplot(2, 3, 5)
    plt.plot(epoch, time_)
    plt.scatter(best_ep, time_[best_idx], color="red")
    plt.title("Inference Time (min)")

    plt.suptitle(f"{model_name} Training Result (Custom Style)", fontsize=14)

    save_path = os.path.join(save_dir, "result.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] {save_path}")
    print(f"[BEST] Epoch {best_ep}, F1 {best_f1:.4f}")


# ==================================================
# 📁 CNN 실행 세션 (추출 데이터 적용 완료)
# ==================================================
cnn_log = """
Epoch [01/50] | Loss 0.1932 | Accuracy 96.48% | Precision 0.7231 | Recall 0.7534 | F1-score 0.7163 | Inference Time 0.0420 min
Epoch [02/50] | Loss 0.1063 | Accuracy 96.30% | Precision 0.7962 | Recall 0.8035 | F1-score 0.7950 | Inference Time 0.0424 min
Epoch [03/50] | Loss 0.0840 | Accuracy 96.95% | Precision 0.8731 | Recall 0.7997 | F1-score 0.8157 | Inference Time 0.0436 min
Epoch [04/50] | Loss 0.0685 | Accuracy 97.04% | Precision 0.8725 | Recall 0.8173 | F1-score 0.8399 | Inference Time 0.0423 min
Epoch [05/50] | Loss 0.0532 | Accuracy 96.96% | Precision 0.8577 | Recall 0.8236 | F1-score 0.8340 | Inference Time 0.0419 min
Epoch [06/50] | Loss 0.0405 | Accuracy 96.39% | Precision 0.8124 | Recall 0.8603 | F1-score 0.8318 | Inference Time 0.0418 min
Epoch [07/50] | Loss 0.0322 | Accuracy 97.19% | Precision 0.8776 | Recall 0.8259 | F1-score 0.8498 | Inference Time 0.0409 min
Epoch [08/50] | Loss 0.0254 | Accuracy 96.89% | Precision 0.8585 | Recall 0.8256 | F1-score 0.8368 | Inference Time 0.0424 min
Epoch [09/50] | Loss 0.0220 | Accuracy 97.16% | Precision 0.9078 | Recall 0.8017 | F1-score 0.8478 | Inference Time 0.0419 min
Epoch [10/50] | Loss 0.0195 | Accuracy 97.02% | Precision 0.8323 | Recall 0.8295 | F1-score 0.8207 | Inference Time 0.0415 min
Epoch [11/50] | Loss 0.0174 | Accuracy 96.97% | Precision 0.8540 | Recall 0.8345 | F1-score 0.8386 | Inference Time 0.0423 min
Epoch [12/50] | Loss 0.0161 | Accuracy 96.89% | Precision 0.8726 | Recall 0.8146 | F1-score 0.8359 | Inference Time 0.0417 min
Epoch [13/50] | Loss 0.0142 | Accuracy 96.98% | Precision 0.8712 | Recall 0.8287 | F1-score 0.8456 | Inference Time 0.0419 min
Epoch [14/50] | Loss 0.0142 | Accuracy 97.09% | Precision 0.8734 | Recall 0.8228 | F1-score 0.8464 | Inference Time 0.0416 min
Epoch [15/50] | Loss 0.0133 | Accuracy 96.97% | Precision 0.8740 | Recall 0.8142 | F1-score 0.8383 | Inference Time 0.0418 min
Epoch [16/50] | Loss 0.0133 | Accuracy 97.15% | Precision 0.8712 | Recall 0.8363 | F1-score 0.8489 | Inference Time 0.0419 min
Epoch [17/50] | Loss 0.0125 | Accuracy 96.41% | Precision 0.8108 | Recall 0.8361 | F1-score 0.8213 | Inference Time 0.0424 min
Epoch [18/50] | Loss 0.0128 | Accuracy 97.07% | Precision 0.8735 | Recall 0.8267 | F1-score 0.8470 | Inference Time 0.0420 min
Epoch [19/50] | Loss 0.0116 | Accuracy 97.08% | Precision 0.8904 | Recall 0.8128 | F1-score 0.8463 | Inference Time 0.0416 min
Epoch [20/50] | Loss 0.0115 | Accuracy 96.79% | Precision 0.8558 | Recall 0.8199 | F1-score 0.8337 | Inference Time 0.0428 min
Epoch [21/50] | Loss 0.0104 | Accuracy 96.99% | Precision 0.8632 | Recall 0.7960 | F1-score 0.8199 | Inference Time 0.0413 min
Epoch [22/50] | Loss 0.0108 | Accuracy 96.92% | Precision 0.8898 | Recall 0.7973 | F1-score 0.8359 | Inference Time 0.0420 min
Epoch [23/50] | Loss 0.0107 | Accuracy 96.89% | Precision 0.8819 | Recall 0.8119 | F1-score 0.8381 | Inference Time 0.0430 min
Epoch [24/50] | Loss 0.0108 | Accuracy 96.74% | Precision 0.8449 | Recall 0.8253 | F1-score 0.8339 | Inference Time 0.0417 min
Epoch [25/50] | Loss 0.0105 | Accuracy 96.98% | Precision 0.8523 | Recall 0.8150 | F1-score 0.8314 | Inference Time 0.0421 min
Epoch [26/50] | Loss 0.0094 | Accuracy 97.04% | Precision 0.8665 | Recall 0.8329 | F1-score 0.8462 | Inference Time 0.0418 min
Epoch [27/50] | Loss 0.0095 | Accuracy 96.88% | Precision 0.8585 | Recall 0.8269 | F1-score 0.8371 | Inference Time 0.0420 min
Epoch [28/50] | Loss 0.0098 | Accuracy 96.93% | Precision 0.8604 | Recall 0.8266 | F1-score 0.8383 | Inference Time 0.0413 min
Epoch [29/50] | Loss 0.0087 | Accuracy 96.86% | Precision 0.8415 | Recall 0.8372 | F1-score 0.8310 | Inference Time 0.0423 min
Epoch [30/50] | Loss 0.0103 | Accuracy 96.56% | Precision 0.8179 | Recall 0.8009 | F1-score 0.8055 | Inference Time 0.0417 min
Epoch [31/50] | Loss 0.0088 | Accuracy 97.00% | Precision 0.8725 | Recall 0.8206 | F1-score 0.8424 | Inference Time 0.0413 min
Epoch [32/50] | Loss 0.0083 | Accuracy 97.01% | Precision 0.8877 | Recall 0.8072 | F1-score 0.8382 | Inference Time 0.0419 min
Epoch [33/50] | Loss 0.0090 | Accuracy 96.93% | Precision 0.8913 | Recall 0.8223 | F1-score 0.8412 | Inference Time 0.0423 min
Epoch [34/50] | Loss 0.0085 | Accuracy 96.77% | Precision 0.8541 | Recall 0.8151 | F1-score 0.8232 | Inference Time 0.0411 min
Epoch [35/50] | Loss 0.0087 | Accuracy 96.78% | Precision 0.8484 | Recall 0.7960 | F1-score 0.8182 | Inference Time 0.0412 min
Epoch [36/50] | Loss 0.0086 | Accuracy 97.12% | Precision 0.8992 | Recall 0.8081 | F1-score 0.8470 | Inference Time 0.0419 min
Epoch [37/50] | Loss 0.0097 | Accuracy 96.84% | Precision 0.8824 | Recall 0.8037 | F1-score 0.8391 | Inference Time 0.0421 min
Epoch [38/50] | Loss 0.0071 | Accuracy 96.53% | Precision 0.8544 | Recall 0.8323 | F1-score 0.8340 | Inference Time 0.0422 min
Epoch [39/50] | Loss 0.0082 | Accuracy 96.99% | Precision 0.8918 | Recall 0.8038 | F1-score 0.8387 | Inference Time 0.0417 min
Epoch [40/50] | Loss 0.0084 | Accuracy 96.38% | Precision 0.8094 | Recall 0.8278 | F1-score 0.8136 | Inference Time 0.0417 min
Epoch [41/50] | Loss 0.0071 | Accuracy 96.63% | Precision 0.8279 | Recall 0.8307 | F1-score 0.8264 | Inference Time 0.0425 min
Epoch [42/50] | Loss 0.0084 | Accuracy 97.09% | Precision 0.9105 | Recall 0.7984 | F1-score 0.8423 | Inference Time 0.0415 min
Epoch [43/50] | Loss 0.0085 | Accuracy 96.95% | Precision 0.9086 | Recall 0.7958 | F1-score 0.8357 | Inference Time 0.0409 min
Epoch [44/50] | Loss 0.0075 | Accuracy 96.99% | Precision 0.8944 | Recall 0.7982 | F1-score 0.8378 | Inference Time 0.0416 min
Epoch [45/50] | Loss 0.0080 | Accuracy 96.91% | Precision 0.8640 | Recall 0.8088 | F1-score 0.8263 | Inference Time 0.0423 min
Epoch [46/50] | Loss 0.0076 | Accuracy 96.94% | Precision 0.8775 | Recall 0.8086 | F1-score 0.8312 | Inference Time 0.0430 min
Epoch [47/50] | Loss 0.0072 | Accuracy 96.90% | Precision 0.8677 | Recall 0.8241 | F1-score 0.8357 | Inference Time 0.0408 min
Epoch [48/50] | Loss 0.0079 | Accuracy 96.79% | Precision 0.8741 | Recall 0.8060 | F1-score 0.8334 | Inference Time 0.0415 min
Epoch [49/50] | Loss 0.0070 | Accuracy 96.46% | Precision 0.8231 | Recall 0.8362 | F1-score 0.8256 | Inference Time 0.0410 min
Epoch [50/50] | Loss 0.0074 | Accuracy 96.94% | Precision 0.8771 | Recall 0.8148 | F1-score 0.8379 | Inference Time 0.0408 min
"""

plot_custom("CNN_Baseline", cnn_log, "logs/cnn")

