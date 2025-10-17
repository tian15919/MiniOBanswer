import h5py
from multiprocessing import Process, JoinableQueue
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import argparse

# 默认数据库参数
DATABASE = "postgres"
USER = "postgres"
PASSWORD = ""
HOST = "127.0.0.1"
PORT = "5432"

def parse_args():
    parser = argparse.ArgumentParser(description="parser")
    
    parser.add_argument("--database", type=str, default="postgres")
    parser.add_argument("--user", type=str, default="postgres")
    parser.add_argument("--password", type=str)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="5432")
    
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--tablename", type=str, required=True)
    
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--dim_bytes", type=int, default=4)
    parser.add_argument("--vec_offset", type=int, default=0)
    parser.add_argument("--vec_num_limit", type=int, default=100000000)
    parser.add_argument("--num_workers", type=int, default=8)
    
    return parser.parse_args()

def worker_process(task_queue, batch_size, tablename):
    
    conn = psycopg2.connect(
        database=DATABASE, 
        user=USER, 
        password=PASSWORD, 
        host=HOST, 
        port=PORT)
    
    conn.autocommit = True  # 每个进程独立提交
    cursor = conn.cursor()
    
    while True:
        start_id, end_id, chunk = task_queue.get()
        if start_id is None:  # 结束信号
            task_queue.task_done()
            break
            
        try:
            # 为每个向量生成ID并准备插入数据
            data = []
            for vec_id, vec in zip(range(start_id, end_id+1), chunk):
                data.append((vec_id, vec.tolist()))  # (id, vector)元组

            execute_values(
                cursor,
                "INSERT INTO "+tablename+" (id, embedding) VALUES %s",
                data,
                page_size=batch_size
            )
        except Exception as e:
            print(f"Error in worker: {e}")
        finally:
            task_queue.task_done()  # 标记当前任务完成
    
    cursor.execute("select count(*) from " + tablename)
    result = cursor.fetchall()
    print("[INFO] Final count:", result[0][0])
    cursor.close()
    conn.close()

def read_h5py_file_and_insert(hdf5_filename, tablename, num_workers, 
                              batch_size, vec_offset, vec_num_limit):
    # 创建任务队列, 最大容量为工作进程数的2倍
    task_queue = JoinableQueue(maxsize=num_workers*2)
    
    # 打开HDF5文件进行读取
    f = h5py.File(hdf5_filename, 'r')	
    print(f"[INFO] HDF5文件中的数据集键: {list(f.keys())}")
    
    # 获取向量维度, 优先从文件属性中读取, 否则从训练数据第一项推断
    dim = int(f.attrs["dimension"]) if "dimension" in f.attrs else len(f["train"][0])
    print(f"[INFO] 向量维度: {dim}")
    
    # 读取训练数据集作为基础向量
    base_vectors = np.array(f["train"])
    print(f"[INFO] 成功加载基础向量集, 大小为 ({base_vectors.shape[0]} * {dim})")
    
    # 创建工作进程
    workers = []
    for _ in range(num_workers):
        p = Process(target=worker_process, args=(task_queue, batch_size, tablename))
        p.start()
        workers.append(p)
         
    # 初始化已插入向量计数和当前ID
    inserted_vec_num = vec_offset
    vec_num = min(vec_num_limit, base_vectors.shape[0])
    current_id = vec_offset+1
    
    # 分批处理向量数据并添加到任务队列
    while inserted_vec_num < vec_num:
        actual_batch_size = min(batch_size, vec_num - inserted_vec_num)
        
        # 为当前批次分配ID范围
        start_id = current_id
        end_id = start_id + actual_batch_size - 1
        chunk = base_vectors[start_id-1:end_id, :]
        task_queue.put((start_id, end_id, chunk))  # 发送ID范围和数据
        
        inserted_vec_num += actual_batch_size
        current_id += actual_batch_size
        
        # 打印本次插入的记录数量
        print(f"[INFO] 插入 {len(chunk)} 条记录到表 {tablename}, 从 {start_id} 到 {end_id}")
    
    for _ in range(num_workers):
        task_queue.put((None, None, None))
    
    task_queue.join()
    
    for p in workers:
        p.join()
    

if __name__ == "__main__":
    args = parse_args()
    
    DATABASE = args.database
    USER = args.user
    PASSWORD = args.password
    HOST = args.host
    PORT = args.port

    read_h5py_file_and_insert(args.filename, args.tablename, args.num_workers, 
                              args.batch_size, args.vec_offset, args.vec_num_limit)
