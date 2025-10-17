import argparse
import numpy as np
import struct
import multiprocessing
import psycopg2
from functools import partial
from psycopg2.extras import execute_values
import time
import h5py


def parse_arguments():

    parser = argparse.ArgumentParser(description='query')
    
    parser.add_argument('--explain', type=bool, default=False, help='explain_sql')
    parser.add_argument('--hdf5_file', type=str, required=True, help='hdf5_file')
    parser.add_argument("--table_name", type=str, required=True, help='table_name')
    
    parser.add_argument('--host', type=str, default="127.0.0.1", help='PostgreSQL host')
    parser.add_argument('--port', type=str, default="5432", help='PostgreSQL port')
    parser.add_argument('--database', type=str, default="postgres", help='database name')
    parser.add_argument('--user', type=str, default="postgres", help='username')
    parser.add_argument('--password', help='password')
    
    parser.add_argument('--k', type=int, default=10, help='top k')
    parser.add_argument('--metric', choices=['l2', 'cosine', 'inner'], default='l2', help='distance metrics')
    parser.add_argument('--max_vid', type=int, default=100000000, help='max vid')
    parser.add_argument('--workers', type=int, default=8, help='worker num')
    
    # index search parameters
    parser.add_argument('--pgvector_hnsw_ef_search', type=int, default=0, help='hnsw_ef_search')
    parser.add_argument('--pgvector_ivf_probes', type=int, default=0, help='pgvector_ivf_probes')

    parser.add_argument('--query_num', type=int, default=0, help='number of queries to run')
    parser.add_argument('--offset', type=int, default=0, help='offset of queries to run')
    
    return parser.parse_args()

def get_db_connection(args):
    
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password
    )

def get_distance_operator(metric):
    
    return {
        'l2': '<->',
        'cosine': '<=>',
        'inner': '<#>'
    }[metric]

def write_logs(results, log_name):
    for res in results:
        query_idx = res['query_idx']
        query_plan = res['plan']

        with open(log_name, 'a', encoding='utf-8') as file:
            file.write(f'query_id: {query_idx}\n')
            file.write(f'log_info:\n {query_plan}\n')
            file.write('--------------------------\n')
            
def set_local_search_parameters(cur, args):
    # cur.execute("SET effective_io_concurrency = 200;")
    # cur.execute("SET jit = off;")
    
    if args.pgvector_hnsw_ef_search != 0:
        cur.execute(f"SET hnsw.ef_search = {args.pgvector_hnsw_ef_search}")
    
    if args.pgvector_ivf_probes != 0:
        cur.execute(f"SET ivfflat.probes = {args.pgvector_ivf_probes}")

def cal_avg_recall(results, ground_truth, k, max_vid):
    avg_recall = 0

    query_num = len(results)

    for res in results:
        query_idx = res['query_idx']
        query_results = res['results']

        query_vid = []
        for row in query_results:
            query_vid.append(row[0])

        ground_truth_of_query_idx = ground_truth[query_idx]

        match_num = 0
        compare_num = 0
        for gt_vid in ground_truth_of_query_idx:
            if(gt_vid <= max_vid):
                compare_num += 1
                if(compare_num > k):
                    break

                if((gt_vid+1) in query_vid):
                    match_num += 1
        recall = match_num / min(k, compare_num) 
        avg_recall += recall

    avg_recall = avg_recall / query_num
    return avg_recall

def cal_latency_distribution(results):

    latencies = []
    for res in results:
        latency = res['latency']
        latencies.append(latency)

    percentiles = [50, 90, 95, 99, 99.9]
    print('\nlatency distribution:')
    for p in percentiles:
        print(f"{p}%", end="   ")
    print()
    for p in percentiles:
        print(f"{np.percentile(latencies, p):.3f}ms" , end="   ")
    print()
    avg_latency = np.mean(latencies)
    print(f'avg latency is: {avg_latency:.3f}ms')

def worker_func(args, query_data):

    query, query_idx = query_data
    conn = get_db_connection(args)
    conn.autocommit = True
    
    operator = get_distance_operator(args.metric)
    vec_str = '[' + ','.join(map(str, query)) + ']'
    
    try:
        with conn.cursor() as cur:
            set_local_search_parameters(cur, args)

            execution_plan = ""

            if(args.explain):
                cur.execute(f"""
                    EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
                    SELECT id FROM {args.table_name} ORDER BY embedding {operator} %s
                    LIMIT %s
                """, (vec_str, args.k))
                execution_plan = "\n".join(row[0] for row in cur.fetchall())
                tmp_search_result=[]
                latency=0
            else:
                start = time.time()
                cur.execute(f"""
                    SELECT id FROM {args.table_name} ORDER BY embedding {operator} %s
                    LIMIT %s
                """, (vec_str, args.k))
                end = time.time()
                tmp_search_result = cur.fetchall()
                latency = (end - start) * 1000

            result = {
                'query_idx': query_idx,
                'results': tmp_search_result,
                'latency': latency,
                'metric': args.metric,
                'plan': execution_plan
            }

            
    except Exception as e:
        result = {
            'query_idx': query_idx,
            'error': str(e),
            'results': [],
            'latency': 0,
            'plan': execution_plan
        }
    finally:
        conn.close()
    
    return result

def query_database(args, queries):
    # 查询解码
    query_data = [(query, idx) for idx, query in enumerate(queries)]
    
    # 数据查询
    num_workers = args.workers
    with multiprocessing.Pool(num_workers) as pool:
        worker = partial(worker_func, args)
        results = pool.map(worker, query_data)
    
    return results

def get_output_file_name(args):
    if args.pgvector_hnsw_ef_search != 0:
        index = "hnsw"
    
    if args.pgvector_ivf_probes != 0:
        index = "ivf" 
        
    return f"explain_{args.table_name}_{index}.log"


if __name__ == "__main__":
    args = parse_arguments()
    
    # 打开HDF5文件读取测试向量集
    f = h5py.File(args.hdf5_file, 'r')
    print(f"[INFO] HDF5文件中的数据集键: {list(f.keys())}")

    queries = np.array(f["test"])
    ground_truth = np.array(f["neighbors"])
    
    # 根据 offset 和 query_num 参数控制查询范围
    if args.query_num > 0:
        query_num = min(args.query_num, len(queries))
        offset = min(args.offset, len(queries) - 1)
        queries = queries[offset:offset+query_num]
        ground_truth = ground_truth[offset:offset+query_num]
    
    print(f"[INFO] Read {len(queries)} query vectors(dim={queries.shape[1]})")
    print(f"[INFO] Read {len(ground_truth)} ground truth ids(top k={ground_truth.shape[1]})")
    
    if args.pgvector_hnsw_ef_search != 0:
        print(f"[INFO] SET hnsw.ef_search = {args.pgvector_hnsw_ef_search}")
    
    if args.pgvector_ivf_probes != 0:
        print(f"[INFO] SET ivfflat.probes = {args.pgvector_ivf_probes}")
    
    # 向量查询
    start = time.time()
    results = query_database(args, queries)
    end = time.time()
    
    # 计算执行时间与 QPS
    execute_time = end - start
    QPS = len(queries) / execute_time
    print(f"\nquery finished, distance metric: {args.metric}, execute time is: {execute_time:.4f}s, QPS is: {QPS:.3f}")

    if(args.explain):
        file_name = get_output_file_name(args)
        write_logs(results, file_name)
    else:
        cal_latency_distribution(results)
        avg_recall = cal_avg_recall(results, ground_truth, args.k, args.max_vid)
        print(f"avg recall@{args.k} of {len(queries)} queries is: {avg_recall:.6f}")
