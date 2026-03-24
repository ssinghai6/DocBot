"""Transition DOCBOT-801–805 (SCRUM-400–404) to Done."""
import json, requests
from requests.auth import HTTPBasicAuth

JIRA_URL = "https://dbdocbot.atlassian.net"
AUTH = HTTPBasicAuth("singhai.sanshrit@gmail.com", "ATATT3xFfGF0S9OF3iHAA0J1UW64U5DP-b4r2RG2CH8hi87OebCXdYZGTgvLYZUP-123K5XxdwPYBd0yq-dZJFwcOB06XI4jlyyrAOM5I79pMawXazXQADKRdSKNvzjdOhrWpdA8hfUf_s-rlPWE_zb04-pwQ3aJZ_PpIm6bCovbfBtamxEznPo=92D43F04")
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
