import matplotlib.pyplot as plt
import numpy as np

# 原始数据
# n_values = [0.01, 0.05, 0.1, 0.2, 0.5]  # 原始横坐标
n_values = [0.05, 0.1, 0.2, 0.3, 0.4]
mapped_x = np.arange(len(n_values))  # 重新映射到等距索引 [0, 1, 2, 3, 4]

# f1 = [0.878, 0.883, 0.895, 0.879, 0.871]
# pairs_f1 = [0.845, 0.840, 0.860, 0.850, 0.839]
#
f1 = [0.889,0.877, 0.889, 0.895, 0.892]
pairs_f1 = [0.844, 0.824, 0.854, 0.860, 0.840]
# 颜色和标记
colors = ['steelblue', 'orange']
markers = ['v', 'o']
labels = [r'$F_1$', r'pairs-$F_1$']

# 绘制曲线
plt.figure(figsize=(5, 5))
plt.plot(mapped_x, f1, marker=markers[0], color=colors[0], label=labels[0], linewidth=2,linestyle='-')
plt.plot(mapped_x, pairs_f1, marker=markers[1], color=colors[1], label=labels[1],linewidth=2, linestyle='-')

# 轴标签
# plt.xlabel(r'$\lambda_1$',fontdict={'size': 18})
plt.xlabel(r'$\lambda_2$',fontdict={'size': 18})

# 重新设置横坐标，使其均匀
plt.xticks(mapped_x, labels=n_values,fontsize=18, rotation=0)  # 让映射后的横坐标对应原始数值
plt.yticks([ 0.75,0.8, 0.85, 0.9, 0.95],size=18)

# 图例
plt.legend(fontsize=15,loc='upper right')
plt.savefig('output1.png', format='png', bbox_inches='tight')
# 显示图像
plt.show()
