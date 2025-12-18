import os
import csv
import yaml
import matplotlib.pyplot as plt

class Logger():
    def __init__(self, dir):
        self.dir = dir
        self.csv_path = None
        self.graph_path = None
        self.config_path = None
        
        os.makedirs(self.dir, exist_ok=True)
    
    def create_csv(self, headers, filename="loss_history.csv"):
        self.csv_path = self.dir / filename

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def update_csv(self, values):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(values)
    
    def create_graph(self, x_axis_header, y_axis_headers, title, figsize, filename="graph.png"):
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        x = [float(row[x_axis_header]) for row in rows]

        plt.figure(figsize=figsize)

        for y_header in y_axis_headers:
            y = [float(row[y_header]) for row in rows]
            plt.plot(x, y, label=y_header)

        plt.xlabel(x_axis_header)
        plt.title(title)
        plt.legend()
        plt.grid(True)

        self.graph_path = self.dir / filename
        plt.savefig(self.graph_path)
        plt.close()

    def create_config(self, config, filename="config.yaml"):
        self.config_path = self.dir / filename

        with open(self.config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
