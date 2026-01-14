import psutil

def kill_forticlient_processes():
    killed = []
    for proc in psutil.process_iter(['pid', 'name']):
        print(f"Processo encontrado: {proc.info['name']}"   )
        try:
            if proc.info['name'] and 'Forti' in proc.info['name']:
                proc.kill()
                killed.append((proc.info['pid'], proc.info['name']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return killed


if __name__ == "__main__":
    processes = kill_forticlient_processes()
    if processes:
        print("Processos encerrados:")
        for pid, name in processes:
            print(f"PID {pid} - {name}")
    else:
        print("Nenhum processo FortiClient encontrado.")
