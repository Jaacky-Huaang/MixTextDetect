import numpy as np
import matplotlib.pyplot as plt
import json

# 读取数据
with open("/scratch/jh7956/mixset_dataset/1124_data.json", "r") as f:
    data = json.load(f)

# 提取训练过程中的损失
losses = []
for key, value in data.items():
    if "train" in key:
        losses += value["loss"]
losses = np.array(losses)

# 定义损失的百分位数
percentiles = np.linspace(0, 20, num=300)  # 创建200个百分位点，从0%到100%

# 计算每个百分位点对应的损失阈值
thresholds = np.percentile(losses, percentiles)

# 计算每个阈值下的平均损失
average_losses = []
for threshold in thresholds:
    filtered_losses = losses[losses <= threshold]  # 获取小于或等于当前阈值的损失
    average_loss = np.mean(filtered_losses) if len(filtered_losses) > 0 else 0  # 计算平均损失
    average_losses.append(average_loss)

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(percentiles, average_losses, marker='o', linestyle='-', color='blue')
plt.title('Average Loss vs. Percentile Threshold')
plt.xlabel('Percentile Threshold (%)')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()



# save the plot
plt.savefig("/scratch/jh7956/loss_threshold.png")
print("Plot saved to loss_threshold.png")
