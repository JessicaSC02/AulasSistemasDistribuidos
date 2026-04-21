import os
import time
import random
from pathlib import Path

import pandas as pd
from dask.distributed import Client, LocalCluster, wait
import dask.bag as db

# -----------------------------
# 0) Parâmetros da prática
# -----------------------------
LOG_PATH = Path("logs_http.txt")
NUM_LINES = 1_000_000
CHUNK_SIZE = 50_000
SLEEP_PER_RECORD = 0.00002
N_WORKERS_START = 4
N_WORKERS_MID = 2

# -----------------------------
# 1) Funções de Geração e Processamento
#    (Definições podem ficar fora do bloco principal)
# -----------------------------
def generate_logs(path: Path, n_lines: int, chunk: int = 100_000):
    status_pool = [200, 200, 200, 200, 301, 302, 400, 401, 403, 404, 500, 502, 503]
    methods = ["GET", "POST", "PUT", "DELETE"]
    urls = ["/", "/home", "/api/v1/items", "/login", "/checkout", "/search?q=abc"]

    path.unlink(missing_ok=True)
    rng = random.Random(42)
    with path.open("w", encoding="utf-8") as f:
        remaining = n_lines
        while remaining > 0:
            batch = min(chunk, remaining)
            lines = []
            for _ in range(batch):
                ts = f"2025-01-01T12:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}Z"
                ip = f"192.168.{rng.randint(0,255)}.{rng.randint(0,255)}"
                method = rng.choice(methods)
                url = rng.choice(urls)
                status = rng.choice(status_pool)
                size = rng.randint(100, 5000)
                lines.append(f"{ts} {ip} {method} {url} {status} {size}\n")
            f.writelines(lines)
            remaining -= batch

def parse_line(line: str):
    time.sleep(SLEEP_PER_RECORD)
    try:
        ts, ip, method, url, status_str, size_str = line.strip().split(" ", 5)
        return {
            "ts": ts,
            "ip": ip,
            "method": method,
            "url": url,
            "status": int(status_str),
            "size": int(size_str),
        }
    except Exception:
        return None

def is_valid(rec):
    return rec is not None and isinstance(rec.get("status", None), int)

def to_pair_status_one(rec):
    return (rec["status"], 1)

def to_pair_status_size(rec):
    return (rec["status"], rec["size"])

# -------------------------------------------------------------
# 2) Bloco de execução principal protegido para multiprocessamento
# -------------------------------------------------------------
if __name__ == '__main__':
    # Gerar dataset sintético de LOGs (caso não exista)
    if not LOG_PATH.exists():
        print(f"[i] Gerando arquivo de logs sintético: {LOG_PATH} (~{NUM_LINES} linhas)")
        generate_logs(LOG_PATH, NUM_LINES, CHUNK_SIZE)
        print(f"[ok] Arquivo gerado: {LOG_PATH.resolve()} | tamanho ~{LOG_PATH.stat().st_size/1e6:.1f} MB")

    # Subir um “cluster” local
    print(f"[i] Subindo LocalCluster com {N_WORKERS_START} workers...")
    cluster = LocalCluster(n_workers=N_WORKERS_START, threads_per_worker=1, dashboard_address=":8787")
    client = Client(cluster)
    print("[ok] Client conectado:", client)
    print("[i] Dashboard Dask em:", client.dashboard_link)

    # Definir pipeline (Map/Reduce)
    print("[i] Criando Dask Bag a partir do arquivo...")
    bag = db.read_text(str(LOG_PATH), blocksize="64MB").map(parse_line).filter(is_valid)

    counts = bag.map(to_pair_status_one).foldby(
        key=lambda x: x[0],
        binop=lambda acc, x: acc + x[1],
        initial=0
    )

    sums_counts = bag.map(to_pair_status_size).foldby(
        key=lambda x: x[0],
        binop=lambda acc, x: (acc[0] + x[1], acc[1] + 1),
        initial=(0, 0)
    ).map(lambda kv: (kv[0], kv[1][0] / kv[1][1] if kv[1][1] else 0.0))

    # Executar e simular falha
    print("[i] Disparando computação (counts e média de size por status)...")
    f_counts = client.compute(counts)
    f_avg = client.compute(sums_counts)

    time.sleep(1.5)
    print(f"[i] Escalando cluster de {N_WORKERS_START} -> {N_WORKERS_MID} workers...")
    cluster.scale(N_WORKERS_MID)

    wait([f_counts, f_avg])

    # Resultados
    status_counts = dict(sorted(f_counts.result(), key=lambda kv: kv[0]))
    status_avgsize = dict(sorted(f_avg.result(), key=lambda kv: kv[0]))

    df_counts = pd.DataFrame(list(status_counts.items()), columns=["status", "count"])
    df_avg = pd.DataFrame(list(status_avgsize.items()), columns=["status", "avg_size_bytes"])
    df = df_counts.merge(df_avg, on="status")

    print("\n=== RESULTADOS ===")
    print(df.sort_values("status").to_string(index=False))

    print("\n[i] Observações:")
    print("- Cluster local com múltiplos workers")
    print("- Processamento paralelo (Big Data)")
    print("- Escalabilidade e tolerância a falhas")
    print("- CAP: consistência eventual vs disponibilidade")
    print("- Casos de uso: logs HTTP")

    # É uma boa prática fechar o cliente ao final
    client.close()
    cluster.close()