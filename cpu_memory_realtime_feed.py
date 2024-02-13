import psutil
import time

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    return psutil.virtual_memory().percent

def main():
    try:
        while True:
            cpu_usage = get_cpu_usage()
            memory_usage = get_memory_usage()

            print(f"CPU Usage: {cpu_usage}%")
            print(f"Memory Usage: {memory_usage}%\n")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
