import json

input_file = 'logs_premarketToday.txt'
output_file = 'formatted_logs.json'

formatted_data = []

# Read the file and parse only the valid JSON lines
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Ignore the terminal commands and success messages
        if line.strip().startswith('{') and line.strip().endswith('}'):
            try:
                formatted_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

# Write the data out to a new file with proper indentation
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=4)

print(f"Success! Formatted file saved as {output_file}")