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
# 📁 EfficientNet-B0 로그 데이터 세션
# ==================================================
eff_log = """
Epoch [01/50] | Loss 0.2164 | Accuracy 95.67% | Precision 0.5923 | Recall 0.5548 | F1-score 0.5718 | Inference Time  0.1470 min
Epoch [02/50] | Loss 0.1365 | Accuracy 96.40% | Precision 0.7436 | Recall 0.6994 | F1-score 0.7113 | Inference Time  0.1456 min
Epoch [03/50] | Loss 0.1145 | Accuracy 96.70% | Precision 0.8051 | Recall 0.7023 | F1-score 0.6958 | Inference Time  0.1373 min
Epoch [04/50] | Loss 0.0982 | Accuracy 97.11% | Precision 0.8836 | Recall 0.7897 | F1-score 0.8296 | Inference Time  0.1408 min
Epoch [05/50] | Loss 0.0865 | Accuracy 96.87% | Precision 0.8336 | Recall 0.8276 | F1-score 0.8253 | Inference Time  0.1357 min
Epoch [06/50] | Loss 0.0783 | Accuracy 96.84% | Precision 0.8124 | Recall 0.8458 | F1-score 0.8171 | Inference Time  0.1367 min
Epoch [07/50] | Loss 0.0720 | Accuracy 97.50% | Precision 0.9142 | Recall 0.8319 | F1-score 0. sheet 0.8678 | Inference Time  0.1543 min
Epoch [08/50] | Loss 0.0670 | Accuracy 97.09% | Precision 0.8548 | Recall 0.8713 | F1-score 0.8603 | Inference Time  0.1428 min
Epoch [09/50] | Loss 0.0625 | Accuracy 97.40% | Precision 0.8428 | Recall 0.8668 | F1-score 0.8523 | Inference Time  0.1411 min
Epoch [10/50] | Loss 0.0568 | Accuracy 96.81% | Precision 0.7965 | Recall 0.8675 | F1-score 0.8235 | Inference Time  0.1423 min
Epoch [11/50] | Loss 0.0547 | Accuracy 97.40% | Precision 0.9072 | Recall 0.8123 | F1-score 0.8518 | Inference Time  0.1339 min
Epoch [12/50] | Loss 0.0500 | Accuracy 97.28% | Precision 0.8716 | Recall 0.8342 | F1-score 0.8486 | Inference Time  0.1356 min
Epoch [13/50] | Loss 0.0472 | Accuracy 97.28% | Precision 0.8728 | Recall 0.7798 | F1-score 0.7972 | Inference Time  0.1388 min
Epoch [14/50] | Loss 0.0457 | Accuracy 97.23% | Precision 0.8738 | Recall 0.8430 | F1-score 0.8555 | Inference Time  0.1338 min
Epoch [15/50] | Loss 0.0420 | Accuracy 97.40% | Precision 0.8619 | Recall 0.8441 | F1-score 0.8488 | Inference Time  0.1378 min
Epoch [16/50] | Loss 0.0405 | Accuracy 97.40% | Precision 0.8622 | Recall 0.8369 | F1-score 0.8409 | Inference Time  0.1352 min
Epoch [17/50] | Loss 0.0388 | Accuracy 97.29% | Precision 0.9005 | Recall 0.8353 | F1-score 0.8572 | Inference Time  0.1372 min
Epoch [18/50] | Loss 0.0368 | Accuracy 97.40% | Precision 0.8607 | Recall 0.8772 | F1-score 0.8676 | Inference Time  0.1390 min
Epoch [19/50] | Loss 0.0366 | Accuracy 96.82% | Precision 0.8478 | Recall 0.8511 | F1-score 0.8426 | Inference Time  0.1374 min
Epoch [20/50] | Loss 0.0352 | Accuracy 97.19% | Precision 0.8371 | Recall 0.8602 | F1-score 0.8453 | Inference Time  0.1346 min
Epoch [21/50] | Loss 0.0323 | Accuracy 97.19% | Precision 0.8615 | Recall 0.8359 | F1-score 0.8418 | Inference Time  0.1444 min
Epoch [22/50] | Loss 0.0334 | Accuracy 97.28% | Precision 0.8630 | Recall 0.8384 | F1-score 0.8450 | Inference Time  0.1367 min
Epoch [23/50] | Loss 0.0321 | Accuracy 97.54% | Precision 0.8872 | Recall 0.8446 | F1-score 0.8603 | Inference Time  0.1399 min
Epoch [24/50] | Loss 0.0307 | Accuracy 97.11% | Precision 0.8894 | Recall 0.8191 | F1-score 0.8504 | Inference Time  0.1430 min
Epoch [25/50] | Loss 0.0307 | Accuracy 97.60% | Precision 0.8960 | Recall 0.8473 | F1-score 0.8691 | Inference Time  0.1316 min
Epoch [26/50] | Loss 0.0288 | Accuracy 97.46% | Precision 0.8734 | Recall 0.8449 | F1-score 0.8575 | Inference Time  0.1340 min
Epoch [27/50] | Loss 0.0292 | Accuracy 97.34% | Precision 0.8443 | Recall 0.8493 | F1-score 0.8432 | Inference Time  0.1363 min
Epoch [28/50] | Loss 0.0300 | Accuracy 97.44% | Precision 0.8562 | Recall 0.8610 | F1-score 0.8560 | Inference Time  0.1323 min
Epoch [29/50] | Loss 0.0280 | Accuracy 97.23% | Precision 0.8698 | Recall 0.8481 | F1-score 0.8554 | Inference Time  0.1379 min
Epoch [30/50] | Loss 0.0282 | Accuracy 97.14% | Precision 0.8677 | Recall 0.8358 | F1-score 0.8465 | Inference Time  0.1414 min
Epoch [31/50] | Loss 0.0262 | Accuracy 96.98% | Precision 0.8363 | Recall 0.8577 | F1-score 0.8414 | Inference Time  0.1555 min
Epoch [32/50] | Loss 0.0278 | Accuracy 97.37% | Precision 0.8806 | Recall 0.8275 | F1-score 0.8520 | Inference Time  0.1509 min
Epoch [33/50] | Loss 0.0269 | Accuracy 97.37% | Precision 0.8628 | Recall 0.8316 | F1-score 0.8445 | Inference Time  0.1440 min
Epoch [34/50] | Loss 0.0269 | Accuracy 97.18% | Precision 0.8949 | Recall 0.8194 | F1-score 0.8510 | Inference Time  0.1472 min
Epoch [35/50] | Loss 0.0260 | Accuracy 97.31% | Precision 0.8306 | Recall 0.8658 | F1-score 0.8421 | Inference Time  0.1382 min
Epoch [36/50] | Loss 0.0269 | Accuracy 97.24% | Precision 0.8714 | Recall 0.8482 | F1-score 0.8570 | Inference Time  0.1407 min
Epoch [37/50] | Loss 0.0261 | Accuracy 97.19% | Precision 0.8668 | Recall 0.8512 | F1-score 0.8548 | Inference Time  0.1369 min
Epoch [38/50] | Loss 0.0256 | Accuracy 96.76% | Precision 0.8198 | Recall 0.8777 | F1-score 0.8466 | Inference Time  0.1340 min
Epoch [39/50] | Loss 0.0253 | Accuracy 97.32% | Precision 0.8843 | Recall 0.8294 | F1-score 0.8521 | Inference Time  0.1433 min
Epoch [40/50] | Loss 0.0250 | Accuracy 97.21% | Precision 0.8538 | Recall 0.8570 | F1-score 0.8528 | Inference Time  0.1377 min
Epoch [41/50] | Loss 0.0249 | Accuracy 97.47% | Precision 0.8916 | Recall 0.8524 | F1-score 0.8702 | Inference Time  0.1361 min
Epoch [42/50] | Loss 0.0251 | Accuracy 97.36% | Precision 0.8776 | Recall 0.8438 | F1-score 0.8577 | Inference Time  0.1317 min
Epoch [43/50] | Loss 0.0247 | Accuracy 97.43% | Precision 0.8776 | Recall 0.8567 | F1-score 0.8641 | Inference Time  0.1356 min
Epoch [44/50] | Loss 0.0252 | Accuracy 97.38% | Precision 0.8726 | Recall 0.8575 | F1-score 0.8634 | Inference Time  0.1372 min
Epoch [45/50] | Loss 0.0248 | Accuracy 97.19% | Precision 0.8678 | Recall 0.8386 | F1-score 0.8498 | Inference Time  0.1332 min
Epoch [46/50] | Loss 0.0241 | Accuracy 97.33% | Precision 0.8730 | Recall 0.8384 | F1-score 0.8501 | Inference Time  0.1320 min
Epoch [47/50] | Loss 0.0244 | Accuracy 97.42% | Precision 0.8913 | Recall 0.8445 | F1-score 0.8652 | Inference Time  0.1332 min
Epoch [48/50] | Loss 0.0237 | Accuracy 96.70% | Precision 0.8228 | Recall 0.8793 | F1-score 0.8464 | Inference Time  0.1292 min
Epoch [49/50] | Loss 0.0243 | Accuracy 97.31% | Precision 0.8524 | Recall 0.8242 | F1-score 0.8350 | Inference Time  0.1353 min
Epoch [50/50] | Loss 0.0236 | Accuracy 97.05% | Precision 0.8542 | Recall 0.8782 | F1-score 0.8646 | Inference Time  0.1393 min
"""

plot_custom("EfficientNet-B0", eff_log, "logs/efficientnet")

