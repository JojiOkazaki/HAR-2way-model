import os
import csv
import yaml
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

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
        
        # --- 【修正】色のリストを定義 (例: 1本目は濃いグレー、2本目は薄いグレー) ---
        colors = ['#4D4D4D', '#999999'] 
        # 必要に応じて線種も変えると白黒印刷で区別しやすくなります
        linestyles = ['-', '--']
        
        for i, y_header in enumerate(y_axis_headers):
            y = [float(row[y_header]) for row in rows]
            
            # --- 【修正】色と線種を指定してプロット ---
            # 色のリストの長さ以上にデータがある場合はモジュロ演算で循環させます
            c = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]
            
            plt.plot(x, y, label=y_header, color=c, linestyle=ls)

        #plt.xlabel(x_axis_header)
        #plt.title(title)
        plt.legend()
        plt.grid(True)

        self.graph_path = self.dir / filename
        
        # --- 【修正】bbox_inches='tight' を追加して余白を削除 ---
        plt.savefig(self.graph_path, bbox_inches='tight')
        plt.close()

    def create_config(self, config, filename="config.yaml"):
        self.config_path = self.dir / filename

        with open(self.config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
