"""
Description: This script reads data from a JSON file and writes specific fields to a CSV file.
The JSON file contains instructions and corresponding GDScript code, which are extracted and
written into a CSV file with a specific delimiter.

Author: Torben Oschkinat - cgt104590 - Bachelor degree

Usage:
    python jsonToCsvConverter.py

Requirements:
    - The JSON input file ('godot_dodo_4x_60k_data.json') should be present in the same directory as this script.
    - The script uses standard Python libraries: json and csv.
"""

# Imports
import json
import csv

# Path to the JSON input file
json_file = 'godot_dodo_4x_60k_data.json'

# Define the output CSV file
csv_file = 'output.csv'

# Open the JSON file and load the data
with open(json_file, mode='r', encoding='utf-8') as f:
    data = json.load(f)

# Open the CSV file for writing
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')

    # Write the header
    writer.writerow(['prompt', 'gdscript_code', 'source'])

    # Process each entry in the JSON data
    for entry in data:
        prompt = entry['instruction']
        gdscript_code = entry['output']

        # Write the row to the CSV file
        writer.writerow([prompt, gdscript_code, ''])

# Print success message
print(f"CSV file '{csv_file}' has been created successfully.")