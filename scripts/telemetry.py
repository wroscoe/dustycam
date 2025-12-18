import time
import psutil
import csv
import subprocess
from datetime import datetime

# CONFIG
LOG_FILE = "crash_log.csv"

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return 0.0

def get_throttled_state():
    # Uses the Pi's internal vcgencmd tool to check for power issues
    try:
        output = subprocess.check_output(["vcgencmd", "get_throttled"]).decode()
        # Output looks like "throttled=0x50005"
        return output.strip().split("=")[1]
    except:
        return "N/A"

def decode_throttled_meaning(hex_code):
    if hex_code == "N/A":
        return "Unknown"
    
    try:
        # Handle 0x prefix if present
        code = int(hex_code, 16)
    except:
        return "Parse Error"

    if code == 0:
        return "OK"

    messages = []
    
    # Active Status (Bits 0-3)
    if code & (1 << 0): messages.append("Under-voltage detected")
    if code & (1 << 1): messages.append("Arm frequency capped")
    if code & (1 << 2): messages.append("Currently throttled")
    if code & (1 << 3): messages.append("Soft temp limit active")
    
    # Past Events (Bits 16-19)
    if code & (1 << 16): messages.append("Under-voltage has occurred")
    if code & (1 << 17): messages.append("Arm frequency capping has occurred")
    if code & (1 << 18): messages.append("Throttling has occurred")
    if code & (1 << 19): messages.append("Soft temp limit has occurred")
    
    # Specific user request clarifications
    if (code & (1 << 16)): # Under-voltage occurred
        messages.append("(Power supply dipped below 4.63V)")

    return "; ".join(messages)

print(f"Logging telemetry to {LOG_FILE}...")

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "CPU_Temp_C", "RAM_Used_MB", "RAM_Percent", "CPU_Percent", "Throttled_Hex", "Throttled_Meaning"])

    while True:
        try:
            # 1. Gather Stats
            temp = get_cpu_temp()
            ram = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=None)
            throttled_hex = get_throttled_state()
            throttled_meaning = decode_throttled_meaning(throttled_hex)
            timestamp = datetime.now().strftime("%H:%M:%S")

            # 2. Log to CSV
            row = [timestamp, temp, ram.used >> 20, ram.percent, cpu_usage, throttled_hex, throttled_meaning]
            writer.writerow(row)
            
            # 3. CRITICAL: Flush to disk immediately
            # If we don't flush, the data stays in RAM and disappears when power cuts.
            f.flush()
            
            # Print specifically high values to console for quick spotting
            if temp > 80 or ram.percent > 90:
                print(f"WARNING: Temp={temp}C, RAM={ram.percent}%")
                
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("Stopping telemetry.")
            break