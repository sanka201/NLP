# import re

# log_entry = """
# 2024-06-18 01:43:19,220 (listeneragent-3.3 878937 [99]) __main__ INFO: Peer: pubsub, Sender: listeneragent-3.3_1:, Bus: , Topic: heartbeat/listeneragent-3.3_1, Headers: {'TimeStamp': '2024-06-18T06:43:19.216685+00:00', 'min_compatible_version': '3.0', 'max_compatible_version': ''}, Message: 
# 'GOOD'
# 'haaaa'
# """

# # Define the regular expression pattern
# pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \((?P<agent>[^)]+)\) (?P<debug>[A-Za-z_\.]+) (?P<level>[A-Z]+): (?P<message>(?:.|\n)*?)(?=\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}|$)'

# # Match the pattern against the log entry
# matches = re.finditer(pattern, log_entry, re.DOTALL)

# for match in matches:
#     # Extract the captured groups
#     timestamp = match.group('timestamp')
#     agent = match.group('agent')
#     debug = match.group('debug')
#     level = match.group('level')
#     message = match.group('message').strip()
    
#     print(f"Timestamp: {timestamp}")
#     print(f"Agent: {agent}")
#     print(f"Debug: {debug}")
#     print(f"Level: {level}")
#     print(f"Message: {message}")




# log_entries = """
# 2024-06-18 01:03:00,001 (platform_driveragent-4.0 878936 [223]) platform_driver.driver DEBUG: fake-campus/fake-building/fake-device next scrape scheduled: 2024-06-18 06:03:05+00:00
# Another part of the message that is on a new line
# 2024-06-18 01:05:00,002 (platform_driveragent-4.0 878937 [224]) platform_driver.driver INFO: another message
# """

# # Define the regular expression pattern for a single line
# pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \((?P<agent>[^)]+)\) (?P<debug>[A-Za-z_\.]+) (?P<level>[A-Z]+): (?P<message>.*)'

# lines = log_entries.strip().split('\n')
# current_log = []
# logs = []

# for line in lines:
#     # Check if the line matches the start of a new log entry
#     if re.match(pattern, line):
#         if current_log:
#             logs.append("\n".join(current_log))
#         current_log = [line]
#     else:
#         current_log.append(line)

# # Don't forget to append the last log entry
# if current_log:
#     logs.append("\n".join(current_log))

# for log_entry in logs:
#     match = re.match(pattern, log_entry, re.DOTALL)
#     if match:
#         timestamp = match.group('timestamp')
#         agent = match.group('agent')
#         debug = match.group('debug')
#         level = match.group('level')
#         message = match.group('message').strip()

#         print(f"Timestamp: {timestamp}")
#         print(f"Agent: {agent}")
#         print(f"Debug: {debug}")
#         print(f"Level: {level}")
#         print(f"Message: {message}")


import re
import pandas as pd

def process_log_file(file_path, output_file):
    pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \((?P<agent>[^)]+)\) (?P<debug>[A-Za-z_\.]+) (?P<level>[A-Z]+): (?P<message>.*)'

    data = {
        "timestamp": [],
        "agent": [],
        "debug": [],
        "level": [],
        "message": []
    }

    with open(file_path, 'r') as file:
        current_log = []
        line_count = 0

        for line in file:
            # Check if the line matches the start of a new log entry
            if re.match(pattern, line):
                if current_log:
                    full_log_entry = "\n".join(current_log)
                    match = re.match(pattern, full_log_entry, re.DOTALL)
                    if match:
                        data["timestamp"].append(match.group('timestamp'))
                        data["agent"].append(match.group('agent'))
                        data["debug"].append(match.group('debug'))
                        data["level"].append(match.group('level'))
                        data["message"].append(match.group('message').strip())

                current_log = [line.strip()]
                line_count = 1
            else:
                if line_count < 4:
                    current_log.append(line.strip())
                    line_count += 1

        # Append the last log entry
        if current_log:
            full_log_entry = "\n".join(current_log)
            match = re.match(pattern, full_log_entry, re.DOTALL)
            if match:
                data["timestamp"].append(match.group('timestamp'))
                data["agent"].append(match.group('agent'))
                data["debug"].append(match.group('debug'))
                data["level"].append(match.group('level'))
                data["message"].append(match.group('message').strip())

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Save the DataFrame to a Parquet file
    df.to_parquet(output_file, index=False)

# Process the log file and save to Parquet format
process_log_file('volttron.log', 'parsed_logs.parquet')

# Read the Parquet file
df = pd.read_parquet('parsed_logs.parquet')

# Display the DataFrame
print(df)