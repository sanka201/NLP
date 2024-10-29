import re

log_file = 'volttron.log'

xx = "2024-02 guru99,education is fun"
xx2= "2024-06-18 01:03:00,001 (platform_driveragent-4.0 878936 [223]) platform_driver.driver DEBUG: fake-campus/fake-building/fake-device next scrape scheduled: 2024-06-18 06:03:05+00:00"
match = re.match("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (\D+.\W)" ,xx2)
print(match.group(2))



""" import re

log_entry = "2024-06-18 01:30:50,007 (listeneragent-3.3 878937 [99]) __main__ INFO: Peer: pubsub, Sender: platform.driver:, Bus: , Topic: devices/fake-campus/fake-building/fake-device/all, Headers:"

# Define the regular expression pattern
pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \((?P<agent>[^)]+)\) (?P<debug>[A-Za-z_\.]+) (?P<level>[A-Z]+): (?P<message>.*)'

# Match the pattern against the log entry
match = re.match(pattern, log_entry)

if match:
    # Extract the captured groups
    timestamp = match.group('timestamp')
    agent = match.group('agent')
    debug = match.group('debug')
    level = match.group('level')
    message = match.group('message')
    
    print(f"Timestamp: {timestamp}")
    print(f"Agent: {agent}")
    print(f"Debug: {debug}")
    print(f"Level: {level}")
    print(f"Message: {message}")
else:
    print("No match found.") """


current_log = []
logs = []
 #Print the captured conte
with open(log_file, 'r') as file:
    
   for line in file:
       
        pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \((?P<agent>[^)]+)\) (?P<debug>[A-Za-z_\.]+) (?P<level>[A-Z]+): (?P<message>.*)'
        pattern2 = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \((?P<agent>[^)]+)\) (?P<debug>[A-Za-z_\.]+) (?P<level>[A-Z]+): (?P<message>(?:.|\n)*?)(?=\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}|$)'

    # Check if the line matches the start of a new log entry
        if re.match(pattern, line):
            if current_log:
                logs.append("\n".join(current_log))
            current_log = [line]
        else:
            current_log.append(line)

# Don't forget to append the last log entry
   if current_log:
        logs.append("\n".join(current_log))

   for log_entry in logs:
        match = re.match(pattern, log_entry, re.DOTALL)
        if match:
            timestamp = match.group('timestamp')
            agent = match.group('agent')
            debug = match.group('debug')
            level = match.group('level')
            message = match.group('message').strip()

            print(f"Timestamp: {timestamp}")
            print(f"Agent: {agent}")
            print(f"Debug: {debug}")
            print(f"Level: {level}")
            print(f"Message: {message}")
            
            
