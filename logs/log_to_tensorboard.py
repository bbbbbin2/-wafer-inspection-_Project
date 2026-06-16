import os
import re
from torch.utils.tensorboard import SummaryWriter

# 1. 모델별 텍스트 로그 정의
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

eff_log = """
Epoch [01/50] | Loss 0.2164 | Accuracy 95.67% | Precision 0.5923 | Recall 0.5548 | F1-score 0.5718 | Inference Time  0.1470 min
Epoch [02/50] | Loss 0.1365 | Accuracy 96.40% | Precision 0.7436 | Recall 0.6994 | F1-score 0.7113 | Inference Time  0.1456 min
Epoch [03/50] | Loss 0.1145 | Accuracy 96.70% | Precision 0.8051 | Recall 0.7023 | F1-score 0.6958 | Inference Time  0.1373 min
Epoch [04/50] | Loss 0.0982 | Accuracy 97.11% | Precision 0.8836 | Recall 0.7897 | F1-score 0.8296 | Inference Time  0.1408 min
Epoch [05/50] | Loss 0.0865 | Accuracy 96.87% | Precision 0.8336 | Recall 0.8276 | F1-score 0.8253 | Inference Time  0.1357 min
Epoch [06/50] | Loss 0.0783 | Accuracy 96.84% | Precision 0.8124 | Recall 0.8458 | F1-score 0.8171 | Inference Time  0.1367 min
Epoch [07/50] | Loss 0.0720 | Accuracy 97.50% | Precision 0.9142 | Recall 0.8319 | F1-score 0.8678 | Inference Time  0.1543 min
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

def parse_and_write_tb(model_name, log_text):
    # 각 실험 폴더 세분화
    writer = SummaryWriter(log_dir=f"runs/{model_name}")
    
    pattern = re.compile(
        r"Epoch \[(\d+)/\d+\]\s*\|\s*Loss ([0-9.]+)\s*\|\s*Accuracy ([0-9.]+)%\s*\|\s*Precision ([0-9.]+)\s*\|\s*Recall ([0-9.]+)\s*\|\s*F1-score ([0-9.]+)",
        re.S
    )
    
    for m in pattern.finditer(log_text):
        epoch = int(m.group(1))
        loss = float(m.group(2))
        acc = float(m.group(3))
        precision = float(m.group(4))
        recall = float(m.group(5))
        f1 = float(m.group(6))
        
        # 텐서보드에 값 주입
        writer.add_scalar("Loss/Validation", loss, epoch)
        writer.add_scalar("Accuracy/Validation", acc, epoch)
        writer.add_scalar("F1-score/Validation", f1, epoch)
        writer.add_scalar("Metrics/Precision", precision, epoch)
        writer.add_scalar("Metrics/Recall", recall, epoch)
        
    writer.close()
    print(f"[성공] {model_name} 데이터를 TensorBoard용 파일로 변환했습니다.")

# 변환 실행
parse_and_write_tb("ResNet18_Experiment", resnet_log)
parse_and_write_tb("EfficientNet_Experiment", eff_log)

