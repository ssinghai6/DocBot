"""
Create DOCBOT-801 through DOCBOT-805 in Jira (SCRUM project).
EPIC-08: Smart Agent Auto-Routing
"""

import json
import requests
from requests.auth import HTTPBasicAuth

JIRA_URL  = "https://dbdocbot.atlassian.net"
EMAIL     = "***REDACTED_EMAIL***"
API_TOKEN = "***REDACTED_JIRA_TOKEN***"

AUTH    = HTTPBasicAuth(EMAIL, API_TOKEN)
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

# SCRUM project key
PROJECT_KEY = "SCRUM"

TICKETS = [
    {
        "docbot_id": "DOCBOT-801",
        "summary": "Extend EXPERT_PERSONAS with Structured Output Contracts",
        "description": (
            "Add 5 new fields to all 8 personas in EXPERT_PERSONAS: response_format, "
            "required_sections, detection_keywords, tool_preference, output_conventions. "
            "Append OUTPUT FORMAT CONTRACT block to each persona_def string. "
            "Update /api/personas route to expose new fields.\n\n"
            "Phase: 3 | Priority: Must Have | Points: 3 | Epic: EPIC-08"
        ),
        "story_points": 3,
    },
    {
        "docbot_id": "DOCBOT-802",
        "summary": "Client-Side Question Routing Function + Auto/Manual Mode State",
        "description": (
            "Implement routeQuestion() weighted keyword scorer in frontend. "
            "Add isAutoMode state (default: true). Extend Message type with agentPersona. "
            "Insert routing call in handleSend before API request. "
            "Tool preference biases chatMode selection.\n\n"
            "Phase: 3 | Priority: Must Have | Points: 5 | Epic: EPIC-08 | Depends: DOCBOT-801"
        ),
        "story_points": 5,
    },
    {
        "docbot_id": "DOCBOT-803",
        "summary": "Sidebar Auto/Override UX Transformation",
        "description": (
            "Replace sidebar 2x4 persona grid as primary UI with AUTO/Manual toggle pills. "
            "Existing grid becomes collapsible Manual Override section. "
            "Reset to Auto link shown when persona is manually pinned.\n\n"
            "Phase: 3 | Priority: Must Have | Points: 3 | Epic: EPIC-08 | Depends: DOCBOT-802"
        ),
        "story_points": 3,
    },
    {
        "docbot_id": "DOCBOT-804",
        "summary": "Per-Message Agent Badge Display",
        "description": (
            "Replace static DocBot label in message header with dynamic colored pill badge "
            "showing which expert agent answered. Badge color from accent_color per agent. "
            "Hybrid queries show two badges side by side.\n\n"
            "Phase: 3 | Priority: Must Have | Points: 2 | Epic: EPIC-08 | Depends: DOCBOT-802"
        ),
        "story_points": 2,
    },
    {
        "docbot_id": "DOCBOT-805",
        "summary": "Per-Agent Response Rendering (Finance tables, Lawyer highlights, Doctor callouts)",
        "description": (
            "Extend renderMessageContent to apply per-format styling: "
            "Finance Expert Key Metrics table gets amber header tint; "
            "Lawyer risk keywords highlighted in red; "
            "Doctor Medical Disclaimer in green left-border callout box; "
            "Data Analyst SQL block uses collapsible component.\n\n"
            "Phase: 3 | Priority: Should Have | Points: 5 | Epic: EPIC-08 | Depends: DOCBOT-801, DOCBOT-804"
        ),
        "story_points": 5,
    },
]


def get_project_id() -> str:
    r = requests.get(
        f"{JIRA_URL}/rest/api/3/project/{PROJECT_KEY}",
        auth=AUTH, headers={"Accept": "application/json"}
    )
    r.raise_for_status()
    return r.json()["id"]


def create_issue(ticket: dict, project_id: str) -> str | None:
    payload = {
        "fields": {
            "project": {"id": project_id},
            "summary": f"[{ticket['docbot_id']}] {ticket['summary']}",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": ticket["description"]}]
                    }
                ]
            },
            "issuetype": {"name": "Story"},
        }
    }

    r = requests.post(
        f"{JIRA_URL}/rest/api/3/issue",
        auth=AUTH, headers=HEADERS,
        data=json.dumps(payload)
    )

    if r.status_code == 201:
        key = r.json()["key"]
        print(f"  ✅ Created {key} — {ticket['docbot_id']}: {ticket['summary']}")
        return key
    else:
        print(f"  ❌ Failed {ticket['docbot_id']}: {r.status_code} {r.text[:300]}")
        return None


def main():
    print("Creating DOCBOT-801 through DOCBOT-805 in Jira...\n")

    project_id = get_project_id()
    print(f"Project ID: {project_id}\n")

    created = []
    for ticket in TICKETS:
        key = create_issue(ticket, project_id)
        if key:
            created.append((key, ticket["docbot_id"]))

    print(f"\n{'='*50}")
    print(f"Created {len(created)}/{len(TICKETS)} tickets")
    for jira_key, docbot_key in created:
        print(f"  {jira_key} → {docbot_key}")
    print("\nDone. Add the SCRUM keys above to jira_update_status.py for future Done transitions.")


if __name__ == "__main__":
    main()
