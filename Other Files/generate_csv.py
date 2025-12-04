import os
import glob
import re
import pandas as pd
from bs4 import BeautifulSoup

# Directory containing the HTML reports
REPORT_DIR = "/Users/adityamanjunatha/Desktop/SSDS_Repo/HTML Network Reports/t0 reports"

# Prepare a list to store extracted data
data_records = []

# Pattern to extract parameters from filename
filename_pattern = re.compile(r'report_(?P<workload>\w+)_w(?P<workers>\d+)_tps(?P<tps>\d+)_txs(?P<txs>\d+)\.html')

# Iterate over all HTML files in the report directory
for html_path in glob.glob(os.path.join(REPORT_DIR, "report_*.html")):
    filename = os.path.basename(html_path)
    match = filename_pattern.match(filename)
    if not match:
        continue
    
    # Extract workload parameters from the filename
    workload = match.group("workload")
    workers = int(match.group("workers"))
    tps = int(match.group("tps"))
    txs = int(match.group("txs"))
    
    # Parse the HTML to extract average latency and throughput
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    
    # Locate the summary table under the 'benchmarksummary' div
    summary_div = soup.find("div", id="benchmarksummary")
    table = summary_div.find("table")
    
    # The first data row is at index 1 (0 is header)
    row = table.find_all("tr")[1]
    tds = row.find_all("td")
    
    # Extract the Avg Latency (column 6) and Throughput (column 7)
    avg_latency = float(tds[6].get_text(strip=True))
    throughput = float(tds[7].get_text(strip=True))
    
    # Append record
    data_records.append({
        "workload": workload,
        "workers": workers,
        "tps": tps,
        "txs": txs,
        "throughput": throughput,
        "avg_latency": avg_latency
    })

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data_records)
csv_path = os.path.join(REPORT_DIR, "fabric__dataset.csv")
df.to_csv(csv_path, index=False)

# # Display the resulting DataFrame
# import ace_tools as tools; tools.display_dataframe_to_user(name="Fabric Performance Dataset", dataframe=df)

# print(f"Dataset saved to: {csv_path}")
