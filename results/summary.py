import pandas as pd

df = pd.DataFrame({
    "Model": ["CNN", "ResNet18", "EfficientNet"],
    "Accuracy": [0,0,0],
    "F1-score": [0,0,0],
    "Inference Time(ms)": [0,0,0],
    "Params(M)": [0,0,0]
})

print(df)
df.to_csv("results/metrics.csv", index=False)
