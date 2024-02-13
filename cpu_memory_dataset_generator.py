import csv
import random

def generate_data(rows):
    data = []
    for i in range(1, rows + 1):
        cpu_usage = random.randint(1, 100)
        memory_usage = random.randint(1, 100)
        signal = 'yes' if cpu_usage > 60 or memory_usage > 60 else 'no'
        data.append([i, cpu_usage, memory_usage, signal])
    return data

def write_to_csv(data, filename='output.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['count', 'cpu_usage', 'memory_usage', 'signal'])
        writer.writerows(data)

def main():
    total_rows = 100000 + 60000
    data = generate_data(total_rows)
    write_to_csv(data)

if __name__ == "__main__":
    main()
