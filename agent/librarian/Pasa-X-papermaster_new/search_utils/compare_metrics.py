def compare():
    # 原始数据
    metrics = ["Crawler Recall", "Precision", "Recall", "Recall@100", "Recall@50", "Recall@20"]
    original = [0.5645, 0.4486, 0.4742, 0.5528, 0.5247, 0.4224]
    improved = [0.6795, 0.0919, 0.4352, 0.5184, 0.3761, 0.2316]

    # 计算并打印结果
    print(f"{'指标':<15} {'原方法':<10} {'改进方法':<10} {'绝对提升':<10} {'相对提升':<10}")
    print("-" * 60)

    for i in range(len(metrics)):
        orig = original[i]
        imp = improved[i]
        abs_increase = imp - orig
        rel_increase = (abs_increase / orig) * 100 if orig != 0 else 0
        print(f"{metrics[i]:<15} {orig:<10.4f} {imp:<10.4f} {abs_increase:<+10.4f} {rel_increase:<+9.2f}%")

def avg_():
    # 原始数据（每组 6 个指标，一共 4 组）
    data = [
        [0.5360, 0.4309, 0.4577, 0.5337, 0.5082, 0.4186],
        [0.5321, 0.4784, 0.4433, 0.5298, 0.5224, 0.4148],
    ]

    metric_names = ["Crawler Recall", "Precision", "Recall", "Recall@100", "Recall@50", "Recall@20"]

    # 计算每组的平均值
    for i, group in enumerate(data):
        avg = sum(group) / len(group)
        print(f"第{i+1}组平均值: {avg:.4f}")

    # 也可以按指标计算各个指标在四组中的平均值：
    print("\n按指标计算的平均值：")
    for j in range(6):
        values = [data[i][j] for i in range(2)]
        avg = sum(values) / len(values)
        print(f"{metric_names[j]:<15}: {avg:.4f}")

if __name__ == '__main__':
    compare()