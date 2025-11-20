import os
import re
import pandas as pd
import streamlit as st
import requests
from collections import Counter
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime


# ============================================================
# ⚙️ STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="AI Incident Management", layout="wide")

# ============================================================
# 🔧 ENV + CONFIG
# ============================================================
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

INCIDENT_FILE = "ServiceNow_Incidents_Sample.xlsx"
SOP_FILE = "Generated_Incident_SOP.txt"
LOG_FILE = Path("Created_Incidents_Log.xlsx")

# ============================================================
# 🧩 LOAD DATA
# ============================================================
@st.cache_data
def load_incidents(path):
    try:
        return pd.read_excel(path)
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_sop(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in re.split(r"-{6,}", content) if b.strip()]
        rows = []
        for b in blocks:
            id_ = re.search(r"Incident ID\s*:\s*(\S+)", b)
            sd = re.search(r"Short Description\s*:\s*(.+)", b)
            ag = re.search(r"Assignment Group\s*:\s*(.+)", b)
            res = re.search(r"Resolution Notes:\s*(.*?)\n\s*(?:Verification:|Preventive|$)", b, re.S)
            rows.append({
                "Incident ID": id_.group(1) if id_ else "",
                "Short Description": sd.group(1).strip() if sd else "",
                "Assignment Group": ag.group(1).strip() if ag else "",
                "Resolution": res.group(1).strip().replace("\n", " ") if res else "",
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error reading SOP: {e}")
        return pd.DataFrame()

incidents_df = load_incidents(INCIDENT_FILE)
sop_df = load_sop(SOP_FILE)

# ============================================================
# 🧮 BUILD VECTORSTORE
# ============================================================
docs = []
for _, row in pd.concat([incidents_df, sop_df]).iterrows():
    text = f"{row.get('Short Description','')} ||| {row.get('Resolution','')} ||| {row.get('Assignment Group','')}"
    docs.append(text)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embeddings)

# ============================================================
# 🧠 PERPLEXITY HELPER
# ============================================================
def perplexity_chat(messages, model="sonar-pro"):
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "messages": messages}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Perplexity API call failed: {e}")
        return "Error fetching model response."

# ============================================================
# 🤖 AGENT DEFINITIONS
# ============================================================
class IncidentCreationAgent:
    def run(self, incident_input):
        incident = {
            "ID": incident_input.get("ID"),
            "Short_Description": incident_input.get("Short_Description", ""),
            "Description": incident_input.get("Description", ""),
            "Priority": incident_input.get("Priority", ""),
            "Status": "Created",
            "Resolution_Suggestion": "",
            "Assignment_Group": "",
            "KB_Matches": [],
        }
        return incident


class ResolutionAgent:
    def run(self, incident):
        query = f"{incident['Short_Description']} {incident['Description']}"
        docs_found = vectorstore.similarity_search(query, k=5)
        matches, groups = [], []
        for d in docs_found:
            content = d.page_content
            parts = content.split("|||")
            sd = parts[0].strip()
            res = parts[1].strip() if len(parts) > 1 else ""
            ag = parts[2].strip() if len(parts) > 2 else ""
            matches.append(f"{sd}\nResolution: {res}\nGroup: {ag}")
            if ag:
                groups.append(ag)

        base = "\n\n".join(matches)
        prompt = [{
            "role": "user",
            "content": (
                f"Incident: {incident['Short_Description']}\n"
                f"Description: {incident['Description']}\n\n"
                f"Relevant Knowledge:\n{base}\n\n"
                "Write detailed, step-by-step recommended resolution steps (do NOT close the incident)."
            )
        }]
        suggestion = perplexity_chat(prompt)
        incident["Resolution_Suggestion"] = suggestion
        incident["Status"] = "Resolution Suggested"
        incident["KB_Matches"] = matches
        return incident


class TriageAgent:
    def run(self, incident):
        groups = []
        for content in incident.get("KB_Matches", []):
            parts = content.split("|||")
            if len(parts) >= 3 and parts[2].strip():
                groups.append(parts[2].strip())

        if groups:
            group = Counter(groups).most_common(1)[0][0]
        else:
            desc = (incident["Short_Description"] + " " + incident["Description"]).lower()
            if any(k in desc for k in ["network", "wifi", "router"]):
                group = "Network Team"
            elif any(k in desc for k in ["printer", "monitor", "keyboard"]):
                group = "IT Support"
            elif any(k in desc for k in ["security", "malware", "unauthorized"]):
                group = "Security Team"
            else:
                group = "Software Team"

        incident["Assignment_Group"] = group

        query_text = f"{incident['Short_Description']} {incident['Description']}"
        similar_docs = vectorstore.similarity_search(query_text, k=5)

        similar_priorities = []
        for doc in similar_docs:
            content = doc.page_content
            for _, row in incidents_df.iterrows():
                text_ref = f"{row.get('Short Description','')} ||| {row.get('Resolution','')} ||| {row.get('Assignment Group','')}"
                if content.strip()[:40] in text_ref:
                    similar_priorities.append(row.get("Priority", "Medium"))
                    break

        if similar_priorities:
            predicted_priority = Counter(similar_priorities).most_common(1)[0][0]
        else:
            predicted_priority = "Medium"

        incident["Priority"] = predicted_priority
        incident["Status"] = "Assigned"
        return incident


# ============================================================
# 🧠 NEW: Ticket Status Agent
# ============================================================
class TicketStatusAgent:
    def run(self, user_request: str):
        if not LOG_FILE.exists():
            return "No incident log file found. Please create incidents first."

        try:
            df = pd.read_excel(LOG_FILE)

            if df.empty:
                return "No incidents have been logged yet."

            df.columns = [c.strip() for c in df.columns]
            user_request_lower = user_request.lower()

            # 1️⃣ List all tickets (summary view)
            if "all" in user_request_lower or "current ticket" in user_request_lower:
                summary_df = df[["Incident ID", "Short Description", "Description"]]
                return summary_df.to_markdown(index=False)

            # 2️⃣ Show specific ticket details
            import re
            match = re.search(r"(\d+)", user_request_lower)
            if match:
                ticket_id = int(match.group(1))
                ticket_row = df[df["Incident ID"] == ticket_id]
                if ticket_row.empty:
                    return f"No ticket found with ID {ticket_id}."
                record = ticket_row.iloc[0]
                details = (
                    f"### 🧾 Ticket {record['Incident ID']}\n"
                    f"**Short Description:** {record.get('Short Description','')}\n"
                    f"**Description:** {record.get('Description','')}\n"
                    f"**Priority:** {record.get('Priority','')}\n"
                    f"**Status:** {record.get('Status','')}\n"
                    f"**Assignment Group:** {record.get('Assignment Group','')}\n"
                    f"**Date Created:** {record.get('Date Created','')}\n"
                    f"**Last Updated:** {record.get('Last Updated','')}\n"
                    f"\n**Resolution Suggestion:**\n\n{record.get('Resolution Suggestion','')}"
                )
                return details

            return "Please specify 'Show all tickets' or 'Details for ticket X'."
        except Exception as e:
            return f"Error reading incident log: {e}"

# ============================================================
# 🕸️ BUILD LANGGRAPH STATEGRAPH
# ============================================================
class IncidentState(TypedDict, total=False):
    ID: Optional[int]
    Short_Description: str
    Description: str
    Priority: str
    Status: str
    Resolution_Suggestion: str
    Assignment_Group: str
    KB_Matches: List[str]

graph = StateGraph(state_schema=IncidentState)
graph.add_node("incident_creator", IncidentCreationAgent().run)
graph.add_node("resolution_agent", ResolutionAgent().run)
graph.add_node("triage_agent", TriageAgent().run)

graph.add_edge("incident_creator", "resolution_agent")
graph.add_edge("resolution_agent", "triage_agent")
graph.set_entry_point("incident_creator")
graph.set_finish_point("triage_agent")

# ============================================================
# 🖥️ STREAMLIT UI
# ============================================================
st.title("🧠 Agentic AI Incident Management System (LangGraph 1.x)")

# Session init
if "incidents" not in st.session_state:
    st.session_state["incidents"] = []

# Create Incident UI
st.subheader("📋 Create New Incident")
with st.form("incident_form"):
    sd = st.text_input("Short Description", placeholder="e.g. VPN not connecting")
    desc = st.text_area("Detailed Description")
    st.caption("🔄 Priority will be automatically assigned during triage.")
    submit = st.form_submit_button("Create Incident")

if submit:
    # persistent ID generator
    def get_next_incident_id():
        if LOG_FILE.exists():
            try:
                df_existing = pd.read_excel(LOG_FILE)
                if "Incident ID" in df_existing.columns and not df_existing["Incident ID"].dropna().empty:
                    nums = pd.to_numeric(df_existing["Incident ID"], errors="coerce").dropna()
                    return int(nums.max()) + 1 if not nums.empty else 1
            except Exception:
                return 1
        return 1

    inc_id = get_next_incident_id()
    data = {"ID": inc_id, "Short_Description": sd, "Description": desc}

    app = graph.compile()
    result_state = app.invoke(data)
    inc = result_state
    st.session_state["incidents"].append(inc)

    # Save to Excel log
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_incident_df = pd.DataFrame([{
        "Incident ID": int(inc.get("ID")),
        "Short Description": inc.get("Short_Description", ""),
        "Description": inc.get("Description", ""),
        "Priority": inc.get("Priority", ""),
        "Status": inc.get("Status", ""),
        "Resolution Suggestion": inc.get("Resolution_Suggestion", ""),
        "Assignment Group": inc.get("Assignment_Group", ""),
        "Date Created": current_time,
        "Last Updated": current_time
    }])


    if LOG_FILE.exists():
        existing = pd.read_excel(LOG_FILE)
        combined = pd.concat([existing, new_incident_df], ignore_index=True)
        combined.drop_duplicates(subset=["Incident ID"], keep="last", inplace=True)
    else:
        combined = new_incident_df

    combined.to_excel(LOG_FILE, index=False)


    st.success(f"Incident {inc_id} created successfully.")
    st.markdown("### 🧩 Suggested Resolution")
    st.code(inc.get("Resolution_Suggestion", "")[:1500])
    st.markdown(f"**Assignment Group:** {inc.get('Assignment_Group', 'N/A')}")
    st.markdown(f"**Priority:** {inc.get('Priority', 'N/A')}")

# ============================================================
# 🤖 AGENTIC TICKET STATUS QUERY SECTION
# ============================================================
st.markdown("---")
st.header("🤖 Ask AI: Check Current Ticket Status")

user_query = st.text_input(
    "Ask about tickets (e.g., 'Show all tickets' or 'Details for ticket 2')"
)

if st.button("Ask Agent"):
    agent = TicketStatusAgent()
    result = agent.run(user_query)

    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif isinstance(result, str) and result.strip().startswith("|"):
        st.markdown(result)
    else:
        # Detect if it contains a long resolution section
        if "Resolution Suggestion:" in result:
            parts = result.split("**Resolution Suggestion:**")
            st.markdown(parts[0])
            if len(parts) > 1:
                with st.expander("🧩 View Full Resolution Suggestion"):
                    st.markdown(parts[1])
        else:
            st.markdown(result)
