import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

stop_event = threading.Event()

def signal_handler(signum, frame):
    print(f"\n收到信号 {signum}，正在优雅退出...")
    stop_event.set()
    print("程序已请求停止，等待当前任务完成...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def exec_parallel(input_path, output_path, max_workers,
                  input_func, process_func, output_func):
    """并行主线程"""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []
    # 读取待处理数据
    lines = input_func(input_path, output_path)
    #####
    for line in lines:
        if stop_event.is_set():
            break
        # 提交任务
        futures.append(executor.submit(process_func, line, stop_event))
        #####
    for idx, future in enumerate(as_completed(futures), 1):
        if stop_event.is_set():
            for f in futures:
                f.cancel()
            break
        try:
            result = future.result(timeout=300)                    
            # 写入结果
            output_func(result, output_path)
            #####
            print('finish', idx)
        except Exception as e:
            print(f"处理第{idx}个future时出错: {str(e)}")
            continue
    executor.shutdown(wait=True)
    print('All done!')
    # send('a')


if __name__ == "__main__":
    """
    input_func(input_path, output_path)
    process_func(line, stop_event)
    output_func(result, output_path)
    """
    pass
