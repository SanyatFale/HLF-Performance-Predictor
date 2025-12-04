import os
import pandas as pd
from bs4 import BeautifulSoup

def extract_table_data(html_file):
    """Extracts data from the first table in an HTML file."""
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the first table
    table = soup.find('table')
    
    if table:
        # Extract headers
        headers = [th.text.strip() for th in table.find_all('th')]
        
        # Extract row data
        data = []
        for row in table.find_all('tr')[1:]:  # Skip header row
            row_data = [td.text.strip() for td in row.find_all('td')]
            data.append(row_data)
        
        return headers, data
    else:
        return None, None

def process_reports(base_dir):
    """Processes HTML reports in subdirectories and saves to CSV files."""
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            all_data = []
            
            for file in os.listdir(folder_path):
                if file.endswith('.html'):
                    file_path = os.path.join(folder_path, file)
                    
                    headers, data = extract_table_data(file_path)
                    
                    if headers and data:
                        for row in data:
                            all_data.append(dict(zip(headers, row)))
            
            if all_data:
                df = pd.DataFrame(all_data)
                csv_file = os.path.join(base_dir, f"{folder}_data.csv")
                df.to_csv(csv_file, index=False)
                print(f"CSV file created: {csv_file}")
            else:
                print(f"No data found in folder: {folder}")

# Specify the base directory containing the folders t0, t1, ...
base_directory = "/Users/adityamanjunatha/Desktop/SSDS_Project/HTML Network Reports"
process_reports(base_directory)
