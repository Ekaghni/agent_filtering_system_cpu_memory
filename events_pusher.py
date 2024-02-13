from opensearchpy import OpenSearch
from datetime import datetime
import json

host = "172.16.10.131"
port = 8200
auth = ("admin", "admin")
ca_certs_path = "/home/telaverge/agent_filtering_system/trial_cert.pem"

client = OpenSearch(
    hosts=[{"host": host, "port": port}],
    http_compress=True,
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    ca_certs=ca_certs_path,
)

index_name = 'trial_agent_score_11'

# Check if the index already exists
if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)

# Define the index mapping
# mapping = {
#     "properties": {
#         "@timestamp": {"type": "date"},
#         "dataset": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
#         "rating": {"type": "double"}
#     }
# }

# # Create the index with the specified mapping
# client.indices.create(index=index_name, body={"mappings": mapping})

# Load your dataset_ratings from JSON
with open('agent_scores.json', 'r') as file:
    data = json.load(file)

# Convert '@timestamp' to datetime object
timestamp_dt = datetime.strptime(data['@timestamp'], "%Y-%m-%dT%H:%M:%SZ")

# Extract epoch timestamp from datetime object
timestamp_epoch = timestamp_dt.timestamp()

# Extract rating value from the 'rating' dictionary
rating_value = data.get('ekaghni', 0.0)  # Replace 'ekaghni' with the actual key

# Create document with converted timestamp and other fields as floats
document = {
    '@timestamp': timestamp_epoch,
    'dataset': data['@timestamp'],
    'rating': rating_value
}

# Push data to the index
response = client.index(
    index=index_name,
    body=document,
    id=2,  # You can use a different ID if needed
    refresh=True
)

print(f'\nAdding document 1:')
print(response)
