import csv

def find_min_max(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        y_values = []
        time_values = []

        for row in reader:
            y_values.append(float(row['Y']))
            time_values.append(row['Time'])

        max_value = max(y_values)
        min_value = min(y_values)
        max_indices = [i for i, value in enumerate(y_values) if value == max_value]
        min_indices = [i for i, value in enumerate(y_values) if value == min_value]

        max_times = [time_values[i] for i in max_indices]
        min_times = [time_values[i] for i in min_indices]

        return max_value, max_times, min_value, min_times

# 示例用法
csv_file_path = 'output2.csv'  # 替换为实际的CSV文件路径
max_value, max_times, min_value, min_times = find_min_max(csv_file_path)

print(f"最大值: {max_value}")
print("对应的Time:")
for time in max_times:
    print(time)

print(f"最小值: {min_value}")
print("对应的Time:")
for time in min_times:
    print(time)