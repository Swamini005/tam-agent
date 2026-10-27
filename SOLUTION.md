# Support Ticket Triage Agent — Solution Design

## 1. Design

The agent automates single-ticket triage using an iterative, ReAct-style tool-calling loop that decouples multi-step information gathering from structured decision synthesis.

```mermaid
flowchart TD
    Start([Receive Ticket]) --> Filter[Hide Post-Intake Fields & Set Anchor Date]
    Filter --> Loop[Agent Loop: Groq gpt-oss-120b]
    Loop --> ToolChoice{LLM Decision}
    ToolChoice -- Calls Tool --> Exec[Execute Tool in tools.py]
    Exec --> FeedResult[Feed Output to LLM Context] --> Loop
    ToolChoice -- Done Investigating --> Final[Final Turn: Structured JSON Schema]
    Final --> ValCheck{Deterministic Validation check()}
    ValCheck -- Issues Found --> Repair[Repair Round: Self-Correction] --> ValCheck
    ValCheck -- Valid / Repaired --> Assembly[Assemble Customer Facts & Step Log]
    Assembly --> Output([Save outputs/TKT-xxxxx.json])
```

- **Tools Exposed**:
  1. `find_account`: Resolves accounts via `account_id`, `company_name`, or `email_domain`. Includes company normalization and mismatch warnings to recover from invalid ticket account IDs.
  2. `get_account`: Fetches plan tier, ARR, licensed/active seats, health status, usage trend, TAM, and computes `days_to_renewal` relative to ticket creation.
  3. `get_ticket_history`: Retrieves earlier tickets for the customer within a 90-day window to identify recurring issues or open threads.
  4. `get_ticket`: Inspects full historical ticket details to verify potential duplicates.
  5. `search_kb`: Keyword search across section headings and bodies, surfacing active vs. deprecated notices and replacements.
  6. `read_kb`: Reads specific sections (or full articles) and resolves version redirects (`(vN)`).
- **Stopping & Validation**:
  - The model decides when it has sufficient context (stopping autonomously) or reaches a 12-round safety limit.
  - Final decisions are generated via Groq's strict `json_schema` mode conforming to `schema.py`.
  - A deterministic Python validator ([`check()`](file:///c:/Users/zycus/Desktop/tam%20agent/agent.py#L104-L134)) verifies that cited IDs exist, duplicates are prior tickets, no deprecated KB articles are used, policy rules hold (e.g., Feature Requests are P4; high-risk renewals notify TAM), and internal data is not leaked in the customer reply. If validation fails, the agent receives one targeted repair round.

---

## 2. Evaluation

Before production deployment, triage quality would be evaluated against a curated golden dataset across:
- **Decision Benchmark (F1 / Accuracy)**: Measure precision, recall, and exact-match accuracy against human-labeled ground truth for `priority` (P1–P4), `team` routing, `escalate`, `notify_account_manager`, and `is_duplicate`.
- **Account Resolution Precision**: Success rate in mapping tickets with missing, corrupted, or mismatched `account_id`s to the correct canonical account entity.
- **KB Retrieval & Deprecation Compliance**: Top-K retrieval precision and recall; enforce a 0% tolerance rate for recommending deprecated KB sections when updated versions exist.
- **Data Leakage & Policy Safety**: Automated regex/classifier audits confirming that zero internal signals (ARR, churn status, NPS, escalation notes) appear in `customer_reply`.
- **Duplicate Identification Precision**: False-positive and false-negative rates for duplicate flags against historical tickets.
- **Draft Quality (LLM-as-a-Judge & TAM Blind Review)**: Dual evaluation of customer reply empathy/helpfulness and internal note conciseness/actionability on a 1–5 rubric.
- **Trajectory Efficiency**: Track average tool turns per ticket, token budget compliance, and repair trigger frequency.

---

## 3. Scale: Handling 10x Data (30,000 Tickets, 4,000 Articles)

- **Current Behavior at 10x**:
  - *Retrieval Quality*: The current in-memory keyword counter degrades significantly over 4,000 articles due to vocabulary mismatch, lack of semantic awareness, and high false-positive hit rates.
  - *Latency*: Linear scanning across 30,000 JSON ticket records and 4,000 markdown sections increases tool execution latency from milliseconds to multiple seconds.
  - *Cost*: Accumulating raw tool outputs inside multi-round LLM prompts increases token usage and hits provider rate limits.
- **Architectural Upgrades**:
  1. **Hybrid Vector + BM25 Search**: Replace in-memory string scanning with a dedicated retrieval engine (e.g., Qdrant or OpenSearch) combining dense embeddings with BM25 lexical retrieval and cross-encoder re-ranking.
  2. **Relational Indexing**: Migrate `accounts.json` and `tickets.json` to PostgreSQL/SQLite with B-Tree indexes on `account_id`, `normalized_company`, and `created_at` for sub-10ms history lookups.
  3. **Hierarchical Summarization**: Return compact document summaries and snippets to the agent first, fetching full text only upon explicit `read_kb` calls.
  4. **Tiered Model Routing**: Use smaller, cost-effective models (e.g., 8B/70B) for tool calling and data gathering, reserving larger reasoning models for policy evaluation and response drafting, cutting API costs by ~60–75%.

---

## 4. Limitations

1. **Lexical Keyword Retrieval**: `search_kb` relies on basic token overlap; it misses semantic synonyms, domain jargon, or queries without exact word matches.
2. **Provider Lock-in & Rate Limiting**: The client is tailored to Groq with aggressive context trimming (`_fit`) to survive 8,000 TPM free-tier caps; it lacks native adapters for local Ollama, OpenAI, or Anthropic.
3. **Single-Turn Workflow**: Handles only initial intake; cannot manage back-and-forth customer clarification loops once missing information is provided.
4. **Windowed Duplicate Scope**: Duplicate lookups are restricted to a 90-day window for the same company, missing cross-account widespread incidents or older recurring bugs.
5. **No File Attachment Parsing**: Cannot inspect real-world ticket attachments (logs, screenshots, stack traces, HAR files).

---

## 5. How to Run

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Configure GROQ_API_KEY in .env

# Run agent on specific tickets or sample
python agent.py TKT-10042 TKT-10050
python agent.py --sample 10

# Validate generated outputs
python check_output.py outputs/
```
