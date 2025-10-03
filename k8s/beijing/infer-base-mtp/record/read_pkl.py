import pickle

# --- 1. 加载数据 (与之前相同) ---
file_path = "/tmp/0.pkl"  # 你的 .pkl 文件路径
try:
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    print("数据加载成功。") # 仅在控制台打印一个成功提示
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
    exit() # 如果文件不存在，退出程序
except Exception as e:
    print(f"加载文件时发生错误: {e}")
    exit()


# --- 2. 将数据写入日志文件 ---
log_file_path = "data_log.log" # 日志文件的路径和名称

try:
    # 使用 "a" 模式（append，追加），这样每次运行都会在文件末尾添加新内容，而不是覆盖
    # 使用 "w" 模式会覆盖原有内容
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        # 写入一个标题，方便区分不同批次的日志
        log_file.write("="*50 + "\n")
        log_file.write(f"--- 新日志记录: {file_path} ---\n")
        log_file.write("="*50 + "\n")
        
        # 将加载的数据转换为字符串并写入
        # str() 函数可以处理大部分基本数据类型（列表、字典、字符串等）
        log_content = str(loaded_data)
        
        log_file.write(log_content)
        log_file.write("\n\n") # 写入空行，方便下次阅读

    print(f"数据已成功写入日志文件: {log_file_path}")

except Exception as e:
    print(f"写入日志文件时发生错误: {e}")