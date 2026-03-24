"""Transition DOCBOT-801–805 (SCRUM-400–404) to Done."""
import json, requests
from requests.auth import HTTPBasicAuth

JIRA_URL = "https://dbdocbot.atlassian.net"
AUTH = HTTPBasicAuth("***REDACTED_EMAIL***", "***REDACTED_JIRA_TOKEN***")
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

TICKETS = [
    ("SCRUM-400", "DOCBOT-801"),
    ("SCRUM-401", "DOCBOT-802"),
    ("SCRUM-402", "DOCBOT-803"),
    ("SCRUM-403", "DOCBOT-804"),
    ("SCRUM-404", "DOCBOT-805"),
]

for key, docbot in TICKETS:
    r = requests.post(f"{JIRA_URL}/rest/api/3/issue/{key}/transitions",
        auth=AUTH, headers=HEADERS,
        data=json.dumps({"transition": {"id": "51"}}))
    print(f"  {'✅' if r.status_code == 204 else '❌'} {key} ({docbot}) → Done [{r.status_code}]")
