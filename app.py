from flask import Flask, render_template, request, jsonify
import os
import time
import threading
import psutil
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import glob
import statistics

app = Flask(__name__)

# --- Progress Takibi ---
progress = {'total': 1, 'done': 0, 'running': False}
progress_lock = threading.Lock()

def process_chunk(args):
    chunk, operation = args
    if operation == 'blur':
        return cv2.GaussianBlur(chunk, (5, 5), 0)
    elif operation == 'edge':
        return cv2.Canny(chunk, 100, 200)
    elif operation == 'grayscale':
        return cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
    return chunk

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
    with progress_lock:
        progress['done'] += 1

def measure_cpu_ram_during(stop_flag, cpu_samples, ram_samples):
    process = psutil.Process(os.getpid())
    while not stop_flag.is_set():
        cpu_samples.append(psutil.cpu_percent(percpu=True))
        ram_samples.append(process.memory_info().rss / (1024 * 1024))  # MB
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

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        for name, entries in temps.items():
            for entry in entries:
                # Try to find a CPU/core temp
                if 'cpu' in name.lower() or 'core' in entry.label.lower() or 'package' in entry.label.lower():
                    return round(entry.current, 1)
            # fallback: return first found
            return round(entries[0].current, 1)
    except Exception:
        return None

def get_disk_io():
    io = psutil.disk_io_counters()
    return io.read_bytes, io.write_bytes

def get_total_cpu_time():
    p = psutil.Process(os.getpid())
    t = p.cpu_times()
    return t.user + t.system

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except (FileNotFoundError, PermissionError):
                    pass
    else:
        os.makedirs(folder_path, exist_ok=True)

def time_limited_processing(mode, n_cores, duration, image_paths, operations, output_folder):
    total_jobs = [(image_path, op, output_folder) for image_path in image_paths for op in operations]
    with progress_lock:
        progress['total'] = len(total_jobs)
        progress['done'] = 0
        progress['running'] = True
    count = 0
    cpu_samples = []
    ram_samples = []
    stop_flag = threading.Event()
    t = threading.Thread(target=measure_cpu_ram_during, args=(stop_flag, cpu_samples, ram_samples))
    t.start()
    start_time = time.time()
    read_start, write_start = get_disk_io()
    cpu_time_start = get_total_cpu_time()
    next_step = 3
    step_results = []
    if mode == 'paralel':
        with Pool(n_cores) as pool:
            for _ in pool.imap_unordered(worker, total_jobs):
                count += 1
                elapsed = time.time() - start_time
                if elapsed >= next_step and elapsed < duration:
                    avg_cpu = get_cpu_average(cpu_samples)
                    max_ram = get_ram_max(ram_samples)
                    energy_score = elapsed * avg_cpu
                    cpu_temp = get_cpu_temp()
                    read_now, write_now = get_disk_io()
                    cpu_time_now = get_total_cpu_time()
                    disk_read_mb = (read_now - read_start) / (1024*1024)
                    disk_write_mb = (write_now - write_start) / (1024*1024)
                    disk_read_speed = disk_read_mb / elapsed if elapsed > 0 else 0
                    disk_write_speed = disk_write_mb / elapsed if elapsed > 0 else 0
                    cpu_time_total = cpu_time_now - cpu_time_start
                    efficiency_energy = energy_score / count if count > 0 else 0
                    efficiency_time = elapsed / count if count > 0 else 0
                    step_results.append({
                        'time': int(next_step),
                        'count': count,
                        'cpu': round(avg_cpu, 2),
                        'ram': round(max_ram, 2),
                        'elapsed': round(elapsed, 2),
                        'energy': round(energy_score, 2),
                        'cpu_temp': cpu_temp,
                        'disk_read': round(disk_read_speed, 2),
                        'disk_write': round(disk_write_speed, 2),
                        'cpu_time': round(cpu_time_total, 2),
                        'eff_energy': round(efficiency_energy, 4),
                        'eff_time': round(efficiency_time, 4)
                    })
                    next_step += 3
                if elapsed > duration:
                    break
    else:
        for job in total_jobs:
            worker(job)
            count += 1
            elapsed = time.time() - start_time
            if elapsed >= next_step and elapsed < duration:
                avg_cpu = get_cpu_average(cpu_samples)
                max_ram = get_ram_max(ram_samples)
                energy_score = elapsed * avg_cpu
                cpu_temp = get_cpu_temp()
                read_now, write_now = get_disk_io()
                cpu_time_now = get_total_cpu_time()
                disk_read_mb = (read_now - read_start) / (1024*1024)
                disk_write_mb = (write_now - write_start) / (1024*1024)
                disk_read_speed = disk_read_mb / elapsed if elapsed > 0 else 0
                disk_write_speed = disk_write_mb / elapsed if elapsed > 0 else 0
                cpu_time_total = cpu_time_now - cpu_time_start
                efficiency_energy = energy_score / count if count > 0 else 0
                efficiency_time = elapsed / count if count > 0 else 0
                step_results.append({
                    'time': int(next_step),
                    'count': count,
                    'cpu': round(avg_cpu, 2),
                    'ram': round(max_ram, 2),
                    'elapsed': round(elapsed, 2),
                    'energy': round(energy_score, 2),
                    'cpu_temp': cpu_temp,
                    'disk_read': round(disk_read_speed, 2),
                    'disk_write': round(disk_write_speed, 2),
                    'cpu_time': round(cpu_time_total, 2),
                    'eff_energy': round(efficiency_energy, 4),
                    'eff_time': round(efficiency_time, 4)
                })
                next_step += 3
            if elapsed > duration:
                break
    stop_flag.set()
    t.join()
    elapsed = time.time() - start_time
    read_end, write_end = get_disk_io()
    cpu_time_end = get_total_cpu_time()
    avg_cpu = get_cpu_average(cpu_samples)
    max_ram = get_ram_max(ram_samples)
    energy_score = elapsed * avg_cpu
    cpu_temp = get_cpu_temp()
    disk_read_mb = (read_end - read_start) / (1024*1024)
    disk_write_mb = (write_end - write_start) / (1024*1024)
    disk_read_speed = disk_read_mb / elapsed if elapsed > 0 else 0
    disk_write_speed = disk_write_mb / elapsed if elapsed > 0 else 0
    cpu_time_total = cpu_time_end - cpu_time_start
    efficiency_energy = energy_score / count if count > 0 else 0
    efficiency_time = elapsed / count if count > 0 else 0
    with progress_lock:
        progress['running'] = False
    return {
        'count': count,
        'cpu': round(avg_cpu, 2),
        'ram': round(max_ram, 2),
        'time': round(elapsed, 2),
        'energy': round(energy_score, 2),
        'cpu_temp': cpu_temp,
        'disk_read': round(disk_read_speed, 2),
        'disk_write': round(disk_write_speed, 2),
        'cpu_time': round(cpu_time_total, 2),
        'eff_energy': round(efficiency_energy, 4),
        'eff_time': round(efficiency_time, 4)
    }, step_results

@app.route('/progress')
def get_progress():
    with progress_lock:
        return jsonify({
            'total': progress['total'],
            'done': progress['done'],
            'running': progress['running'],
            'percent': int(100 * progress['done'] / progress['total']) if progress['total'] else 0
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_benchmark():
    data = request.json
    cores = int(data.get('cores', 4))
    duration = int(data.get('duration', 15))
    mode = data.get('mode', 'both')
    testtype = data.get('testtype', 'timed')

    max_cores = min(16, cpu_count())
    if cores > max_cores:
        cores = max_cores

    input_folder = "image/ai"
    output_folder_parallel = "result/ai"
    output_folder_serial = "resultseri/ai"
    clear_folder(output_folder_parallel)
    clear_folder(output_folder_serial)
    image_paths = glob.glob(os.path.join(input_folder, "*.*"))
    operations = ['blur', 'edge', 'grayscale']

    results = []
    steps = []
    if testtype == 'all':
        # Tümünü işle: duration yerine tüm işler bitene kadar
        total_jobs = len(image_paths) * len(operations)
        def all_processing(mode, n_cores, image_paths, operations, output_folder):
            with progress_lock:
                progress['total'] = total_jobs
                progress['done'] = 0
                progress['running'] = True
            count = 0
            cpu_samples = []
            ram_samples = []
            stop_flag = threading.Event()
            t = threading.Thread(target=measure_cpu_ram_during, args=(stop_flag, cpu_samples, ram_samples))
            t.start()
            start_time = time.time()
            read_start, write_start = get_disk_io()
            cpu_time_start = get_total_cpu_time()
            jobs = [(image_path, op, output_folder) for image_path in image_paths for op in operations]
            if mode == 'paralel':
                with Pool(n_cores) as pool:
                    for _ in pool.imap_unordered(worker, jobs):
                        count += 1
            else:
                for job in jobs:
                    worker(job)
                    count += 1
            stop_flag.set()
            t.join()
            elapsed = time.time() - start_time
            read_end, write_end = get_disk_io()
            cpu_time_end = get_total_cpu_time()
            avg_cpu = get_cpu_average(cpu_samples)
            max_ram = get_ram_max(ram_samples)
            energy_score = elapsed * avg_cpu
            cpu_temp = get_cpu_temp()
            disk_read_mb = (read_end - read_start) / (1024*1024)
            disk_write_mb = (write_end - write_start) / (1024*1024)
            disk_read_speed = disk_read_mb / elapsed if elapsed > 0 else 0
            disk_write_speed = disk_write_mb / elapsed if elapsed > 0 else 0
            cpu_time_total = cpu_time_end - cpu_time_start
            efficiency_energy = energy_score / count if count > 0 else 0
            efficiency_time = elapsed / count if count > 0 else 0
            with progress_lock:
                progress['running'] = False
            return {
                'count': count,
                'cpu': round(avg_cpu, 2),
                'ram': round(max_ram, 2),
                'time': round(elapsed, 2),
                'energy': round(energy_score, 2),
                'cpu_temp': cpu_temp,
                'disk_read': round(disk_read_speed, 2),
                'disk_write': round(disk_write_speed, 2),
                'cpu_time': round(cpu_time_total, 2),
                'eff_energy': round(efficiency_energy, 4),
                'eff_time': round(efficiency_time, 4)
            }
        if mode in ['both', 'paralel']:
            res_par = all_processing('paralel', cores, image_paths, operations, output_folder_parallel)
            res_par['label'] = 'Paralel'
            results.append(res_par)
        if mode in ['both', 'seri']:
            res_seri = all_processing('seri', 1, image_paths, operations, output_folder_serial)
            res_seri['label'] = 'Seri'
            results.append(res_seri)
    else:
        # Süreli testte her 3 saniyede bir ara sonuçları steps'e ekle
        def timed_processing(mode, n_cores, duration, image_paths, operations, output_folder):
            total_jobs = [(image_path, op, output_folder) for image_path in image_paths for op in operations]
            with progress_lock:
                progress['total'] = len(total_jobs)
                progress['done'] = 0
                progress['running'] = True
            count = 0
            cpu_samples = []
            ram_samples = []
            stop_flag = threading.Event()
            t = threading.Thread(target=measure_cpu_ram_during, args=(stop_flag, cpu_samples, ram_samples))
            t.start()
            start_time = time.time()
            read_start, write_start = get_disk_io()
            cpu_time_start = get_total_cpu_time()
            next_step = 3
            step_results = []
            if mode == 'paralel':
                with Pool(n_cores) as pool:
                    for _ in pool.imap_unordered(worker, total_jobs):
                        count += 1
                        elapsed = time.time() - start_time
                        if elapsed >= next_step and elapsed < duration:
                            avg_cpu = get_cpu_average(cpu_samples)
                            max_ram = get_ram_max(ram_samples)
                            energy_score = elapsed * avg_cpu
                            cpu_temp = get_cpu_temp()
                            read_now, write_now = get_disk_io()
                            cpu_time_now = get_total_cpu_time()
                            disk_read_mb = (read_now - read_start) / (1024*1024)
                            disk_write_mb = (write_now - write_start) / (1024*1024)
                            disk_read_speed = disk_read_mb / elapsed if elapsed > 0 else 0
                            disk_write_speed = disk_write_mb / elapsed if elapsed > 0 else 0
                            cpu_time_total = cpu_time_now - cpu_time_start
                            efficiency_energy = energy_score / count if count > 0 else 0
                            efficiency_time = elapsed / count if count > 0 else 0
                            step_results.append({
                                'time': int(next_step),
                                'count': count,
                                'cpu': round(avg_cpu, 2),
                                'ram': round(max_ram, 2),
                                'elapsed': round(elapsed, 2),
                                'energy': round(energy_score, 2),
                                'cpu_temp': cpu_temp,
                                'disk_read': round(disk_read_speed, 2),
                                'disk_write': round(disk_write_speed, 2),
                                'cpu_time': round(cpu_time_total, 2),
                                'eff_energy': round(efficiency_energy, 4),
                                'eff_time': round(efficiency_time, 4)
                            })
                            next_step += 3
                        if elapsed > duration:
                            break
            else:
                for job in total_jobs:
                    worker(job)
                    count += 1
                    elapsed = time.time() - start_time
                    if elapsed >= next_step and elapsed < duration:
                        avg_cpu = get_cpu_average(cpu_samples)
                        max_ram = get_ram_max(ram_samples)
                        energy_score = elapsed * avg_cpu
                        cpu_temp = get_cpu_temp()
                        read_now, write_now = get_disk_io()
                        cpu_time_now = get_total_cpu_time()
                        disk_read_mb = (read_now - read_start) / (1024*1024)
                        disk_write_mb = (write_now - write_start) / (1024*1024)
                        disk_read_speed = disk_read_mb / elapsed if elapsed > 0 else 0
                        disk_write_speed = disk_write_mb / elapsed if elapsed > 0 else 0
                        cpu_time_total = cpu_time_now - cpu_time_start
                        efficiency_energy = energy_score / count if count > 0 else 0
                        efficiency_time = elapsed / count if count > 0 else 0
                        step_results.append({
                            'time': int(next_step),
                            'count': count,
                            'cpu': round(avg_cpu, 2),
                            'ram': round(max_ram, 2),
                            'elapsed': round(elapsed, 2),
                            'energy': round(energy_score, 2),
                            'cpu_temp': cpu_temp,
                            'disk_read': round(disk_read_speed, 2),
                            'disk_write': round(disk_write_speed, 2),
                            'cpu_time': round(cpu_time_total, 2),
                            'eff_energy': round(efficiency_energy, 4),
                            'eff_time': round(efficiency_time, 4)
                        })
                        next_step += 3
                    if elapsed > duration:
                        break
            stop_flag.set()
            t.join()
            elapsed = time.time() - start_time
            read_end, write_end = get_disk_io()
            cpu_time_end = get_total_cpu_time()
            avg_cpu = get_cpu_average(cpu_samples)
            max_ram = get_ram_max(ram_samples)
            energy_score = elapsed * avg_cpu
            cpu_temp = get_cpu_temp()
            disk_read_mb = (read_end - read_start) / (1024*1024)
            disk_write_mb = (write_end - write_start) / (1024*1024)
            disk_read_speed = disk_read_mb / elapsed if elapsed > 0 else 0
            disk_write_speed = disk_write_mb / elapsed if elapsed > 0 else 0
            cpu_time_total = cpu_time_end - cpu_time_start
            efficiency_energy = energy_score / count if count > 0 else 0
            efficiency_time = elapsed / count if count > 0 else 0
            with progress_lock:
                progress['running'] = False
            return {
                'count': count,
                'cpu': round(avg_cpu, 2),
                'ram': round(max_ram, 2),
                'time': round(elapsed, 2),
                'energy': round(energy_score, 2),
                'cpu_temp': cpu_temp,
                'disk_read': round(disk_read_speed, 2),
                'disk_write': round(disk_write_speed, 2),
                'cpu_time': round(cpu_time_total, 2),
                'eff_energy': round(efficiency_energy, 4),
                'eff_time': round(efficiency_time, 4)
            }, step_results
        steps = []
        if mode in ['both', 'paralel']:
            res_par, steps_par = timed_processing('paralel', cores, duration, image_paths, operations, output_folder_parallel)
            res_par['label'] = 'Paralel'
            results.append(res_par)
        if mode in ['both', 'seri']:
            res_seri, steps_seri = timed_processing('seri', 1, duration, image_paths, operations, output_folder_serial)
            res_seri['label'] = 'Seri'
            results.append(res_seri)
        # steps: [{time, paralel: {...}, seri: {...}}]
        if mode == 'both':
            # Eşleştir paralel ve seri adımlarını
            for i in range(max(len(steps_par), len(steps_seri))):
                step = {'time': 3*(i+1)}
                if i < len(steps_par):
                    step['paralel'] = steps_par[i]
                if i < len(steps_seri):
                    step['seri'] = steps_seri[i]
                steps.append(step)
        elif mode == 'paralel':
            for s in steps_par:
                steps.append({'time': s['time'], 'paralel': s})
        elif mode == 'seri':
            for s in steps_seri:
                steps.append({'time': s['time'], 'seri': s})

    return jsonify({
        'mode': mode,
        'cores': cores,
        'duration': duration,
        'results': results,
        'steps': steps
    })

if __name__ == '__main__':
    app.run(debug=True)