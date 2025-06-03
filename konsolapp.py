import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import os
import glob
import psutil
import threading
import statistics

ioparallel_lock = threading.Lock()
ioparallel_total = [0.0]
ioserial_total = [0.0]

def process_chunk(args):
    """
    Görüntü parçasını işleyen fonksiyon
    """
    chunk, operation = args
    if operation == 'blur':
        return cv2.GaussianBlur(chunk, (5, 5), 0)
    elif operation == 'edge':
        return cv2.Canny(chunk, 100, 200)
    elif operation == 'grayscale':
        return cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
    return chunk

def split_image(image, num_chunks):
    """
    Görüntüyü parçalara böler
    """
    height = image.shape[0]
    chunk_height = height // num_chunks
    chunks = []
    
    for i in range(num_chunks):
        start = i * chunk_height
        end = start + chunk_height if i < num_chunks - 1 else height
        chunks.append(image[start:end])
    
    return chunks

def merge_chunks(chunks):
    """
    İşlenmiş parçaları birleştirir
    """
    return np.vstack(chunks)

def parallel_process_image(image_path, operation, n_cores):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    # Sadece tek çekirdekte işle (iç içe Pool yok!)
    return process_chunk((image, operation))

def serial_process_image(image_path, operation):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    return process_chunk((image, operation))

def process_image_parallel(args):
    image_path, operation, output_folder, n_cores = args
    io_time = 0.0
    try:
        t0 = time.time()
        image = cv2.imread(image_path)
        t1 = time.time()
        io_time += t1 - t0
        if image is None:
            return
        t2 = time.time()
        result = process_chunk((image, operation))
        t3 = time.time()
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{filename}_{operation}.jpg")
        t4 = time.time()
        cv2.imwrite(output_path, result)
        t5 = time.time()
        io_time += (t5 - t4)
    except Exception:
        pass
    with ioparallel_lock:
        ioparallel_total[0] += io_time

def process_image_serial(args):
    image_path, operation, output_folder = args
    io_time = 0.0
    try:
        t0 = time.time()
        image = cv2.imread(image_path)
        t1 = time.time()
        io_time += t1 - t0
        if image is None:
            return
        t2 = time.time()
        result = process_chunk((image, operation))
        t3 = time.time()
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{filename}_{operation}.jpg")
        t4 = time.time()
        cv2.imwrite(output_path, result)
        t5 = time.time()
        io_time += (t5 - t4)
    except Exception:
        pass
    ioserial_total[0] += io_time

def get_core_count():
    max_cores = cpu_count()
    while True:
        try:
            n = int(input(f"Kaç çekirdek kullanmak istiyorsunuz? (1-{max_cores}): "))
            if 1 <= n <= max_cores:
                return n
            else:
                print(f"Lütfen 1 ile {max_cores} arasında bir sayı girin.")
        except ValueError:
            print("Geçerli bir sayı girin.")

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(folder_path, exist_ok=True)

def worker(job):
    image_path, operation, output_folder = job
    try:
        image = cv2.imread(image_path)
        if image is None:
            return
        result = process_chunk((image, operation))
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{filename}_{operation}.jpg")
        cv2.imwrite(output_path, result)
    except Exception:
        pass

def live_cpu_energy_monitor(stop_flag, start_time):
    while not stop_flag.is_set():
        cpu = psutil.cpu_percent(interval=0.1)
        elapsed = time.time() - start_time
        print(f"[CANLI] {elapsed:.1f} sn - Anlık CPU: {cpu:.2f}%  Enerji Skoru: {elapsed*cpu:.2f}")
        time.sleep(2.9)

def time_limited_processing_both(n_cores, duration, image_paths, operations, output_folder_parallel, output_folder_serial):
    # Paralel
    print(f"\n[Paralel] {duration} saniyelik test başlıyor ({n_cores} çekirdek)...")
    total_jobs = [(image_path, op, output_folder_parallel) for image_path in image_paths for op in operations]
    count = 0
    stop_flag = threading.Event()
    start_time = time.time()
    monitor_thread = threading.Thread(target=live_cpu_energy_monitor, args=(stop_flag, start_time))
    monitor_thread.start()
    with Pool(n_cores) as pool:
        for _ in pool.imap_unordered(worker, total_jobs):
            count += 1
            if time.time() - start_time > duration:
                break
    stop_flag.set()
    monitor_thread.join()
    print(f"[Paralel] {duration} saniyede işlenen toplam görsel+işlem sayısı: {count}")

    # Seri
    print(f"\n[Seri] {duration} saniyelik test başlıyor...")
    count = 0
    stop_flag = threading.Event()
    start_time = time.time()
    monitor_thread = threading.Thread(target=live_cpu_energy_monitor, args=(stop_flag, start_time))
    monitor_thread.start()
    for image_path, op, output_folder in [(i, o, output_folder_serial) for i in image_paths for o in operations]:
        worker((image_path, op, output_folder))
        count += 1
        if time.time() - start_time > duration:
            break
    stop_flag.set()
    monitor_thread.join()
    print(f"[Seri] {duration} saniyede işlenen toplam görsel+işlem sayısı: {count}")

def measure_cpu_ram_during(stop_flag, cpu_samples, ram_samples):
    process = psutil.Process(os.getpid())
    while not stop_flag.is_set():
        cpu_samples.append(psutil.cpu_percent(percpu=True))
        ram_samples.append(process.memory_info().rss / (1024 * 1024))  # MB cinsinden
        time.sleep(0.2)

def get_cpu_average(samples):
    if not samples:
        return 0.0
    per_sample_avg = [statistics.mean(s) for s in samples]
    return statistics.mean(per_sample_avg)

def get_ram_max(samples):
    if not samples:
        return 0.0
    return max(samples)

def main():
    input_folder = "image/ai"
    output_folder_parallel = "result/ai"
    output_folder_serial = "resultseri/ai"

    # Sonuç klasörlerini temizle
    clear_folder(output_folder_parallel)
    clear_folder(output_folder_serial)

    image_paths = glob.glob(os.path.join(input_folder, "*.*"))
    operations = ['blur', 'edge', 'grayscale']

    print("Mod seçin:\n1) Kapsamlı karşılaştırma (tablo)\n2) Zaman sınırı ile test (ör: 15 saniye)")
    mode_choice = input("Seçiminiz (1/2): ").strip()

    if mode_choice == '2':
        while True:
            try:
                duration = int(input("Kaç saniye test edilsin? (örn: 15): "))
                break
            except ValueError:
                print("Geçerli bir sayı girin.")
        max_cores = cpu_count()
        while True:
            try:
                n_cores = int(input(f"Kaç çekirdek kullanılsın? (1-{max_cores}): "))
                if 1 <= n_cores <= max_cores:
                    break
                else:
                    print(f"1 ile {max_cores} arasında bir sayı girin.")
            except ValueError:
                print("Geçerli bir sayı girin.")
        time_limited_processing_both(n_cores, duration, image_paths, operations, output_folder_parallel, output_folder_serial)
        return

    # Varsayılan: kapsamlı karşılaştırma
    max_cores = cpu_count()
    test_cores = [1]
    if max_cores >= 2:
        test_cores.append(2)
    if max_cores >= 4:
        test_cores.append(4)
    if max_cores >= 8:
        test_cores.append(8)
    if max_cores > 8:
        test_cores.append(max_cores)

    print("\nÇekirdek sayısına göre paralel işleme karşılaştırması:")
    print(f"{'Çekirdek':<10}{'Süre (sn)':<15}{'I/O Süresi (sn)':<18}{'Ortalama CPU (%)':<20}{'Maks RAM (MB)':<15}{'Enerji Skoru':<15}")
    print("-"*95)

    for n_cores in test_cores:
        jobs_parallel = []
        ioparallel_total[0] = 0.0
        for image_path in image_paths:
            for operation in operations:
                jobs_parallel.append((image_path, operation, output_folder_parallel, n_cores))
        cpu_samples = []
        ram_samples = []
        stop_flag = threading.Event()
        t = threading.Thread(target=measure_cpu_ram_during, args=(stop_flag, cpu_samples, ram_samples))
        t.start()
        start_parallel = time.time()
        with Pool(n_cores) as pool:
            pool.map(process_image_parallel, jobs_parallel)
        end_parallel = time.time()
        stop_flag.set()
        t.join()
        elapsed = end_parallel - start_parallel
        avg_cpu = get_cpu_average(cpu_samples)
        max_ram = get_ram_max(ram_samples)
        energy_score = elapsed * avg_cpu
        io_time = ioparallel_total[0]
        print(f"{n_cores:<10}{elapsed:<15.2f}{io_time:<18.2f}{avg_cpu:<20.2f}{max_ram:<15.2f}{energy_score:<15.2f}")

    print("\nSeri işleme karşılaştırması:")
    jobs_serial = []
    ioserial_total[0] = 0.0
    for image_path in image_paths:
        for operation in operations:
            jobs_serial.append((image_path, operation, output_folder_serial))
    cpu_samples = []
    ram_samples = []
    stop_flag = threading.Event()
    t = threading.Thread(target=measure_cpu_ram_during, args=(stop_flag, cpu_samples, ram_samples))
    t.start()
    start_serial = time.time()
    for job in jobs_serial:
        process_image_serial(job)
    end_serial = time.time()
    stop_flag.set()
    t.join()
    elapsed = end_serial - start_serial
    avg_cpu = get_cpu_average(cpu_samples)
    max_ram = get_ram_max(ram_samples)
    energy_score = elapsed * avg_cpu
    io_time = ioserial_total[0]
    print(f"{'Seri':<10}{elapsed:<15.2f}{io_time:<18.2f}{avg_cpu:<20.2f}{max_ram:<15.2f}{energy_score:<15.2f}")

if __name__ == "__main__":
    main() 