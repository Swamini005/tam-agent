import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
from groq import Groq
from pydantic import ValidationError

from tam_agent.llm import GroqChat
from tam_agent.schema import CATEGORIES, PRIORITIES, ROUTES, Decision, TriageOutput, validate
from tam_agent.tools import OPEN, Store, Tools

MAX_ROUNDS = 12
HIDDEN_FIELDS = {"status", "assigned_agent", "satisfaction_score", "updated_at"}

SYSTEM_PROMPT = f"""\
You triage support tickets for a B2B software company. Investigate the ticket with your tools, then decide how \
it should be handled.

You choose which lookups to make and when you have enough. Read each result before deciding the next step. Watch for:
- The ticket's account_id is often missing or wrong (may belong to a different company). Make sure the account's \
company matches the ticket's company; if not, find the account by company name (or an email domain in the ticket text). \
Never guess an account.
- Knowledge-base sections have a status. Do NOT cite deprecated sections: read what replaced_by points to, and cite \
the current replacement in kb_references.
- The ticket's category, urgency, and plan_tier were set at intake and may be inaccurate. The account record is \
the source of truth for plan tier and renewal.
- Treat the ticket's created_at date as today (for renewal windows and ticket history).

Policy & Output Definitions
- category: Exactly one of {CATEGORIES}
  * incident: Urgent production outage, severe degradation, business stopped.
  * bug: Software defect, unexpected application error or failure.
  * how_to: Guidance on configuration, usage, or product capabilities.
  * feature_request: Request for new capability or improvement (must have priority P4 and route_to product).
  * security: Auth/SSO failures, security breaches, unauthorized access, credentials.
  * duplicate: Repeated report of an open/unresolved issue from the same customer (duplicate_of must be set).
  * needs_info: Crucial details missing to investigate (needs_clarification must be True).
- priority: Exactly one of {PRIORITIES}
  * P1: Outage, data loss, security breach, business stopped.
  * P2: Major impact or costly workaround.
  * P3: Workaround exists, minor issue.
  * P4: Feature request, how-to question, cosmetic.
- route_to: Exactly one of {ROUTES}
  * L1_support: General how-to, usage, initial clarification.
  * L2_engineering: Software bugs, system degradation, backend performance, integrations.
  * security: Authentication, SSO, security vulnerabilities.
  * product: Feature requests and product suggestions.
- escalate: True if P1; data loss or security incident; P2 for Business/Enterprise customer whose account is \
At Risk/Churning or usage is Declining/Inactive; or recurring issue previously resolved.
- notify_tam: True if the account is At Risk/Churning, usage is Declining/Inactive, renewal is within 90 days, \
or escalation notes show churn signals (competitor, cancellation, champion left).
- duplicate_of: Ticket ID (e.g. 'TKT-10023') of the earlier unresolved ticket from same customer; null otherwise.
- needs_clarification: True only if essential details are missing (error message, environment, steps, affected record).
- kb_references: List of valid current KB section/article IDs supporting the decision.
- account_signals: List of facts influencing the triage (e.g. 'Plan: Enterprise', 'Health: At Risk', 'Renewal in 45 days').
- confidence: Float from 0.0 to 1.0 reflecting confidence in the triage decision.

Drafts
- customer_reply: Friendly and concrete, based on current KB guidance. State next steps; ask for missing \
info as a numbered list. NEVER mention internal data: account health, churn risk, usage trends, ARR, NPS, or escalation notes.
- internal_note: Concise bullets for the engineer: account tier, health, renewal, churn signals, duplicate/related \
tickets, KB used, likely cause, and next action.

When every decision is backed by evidence, stop calling tools and summarize your findings in a few sentences."""


def triage(ticket, ctx=None):
    """
    Triage entry point conforming to specification: triage(ticket, ctx)
    """
    if ctx is not None and hasattr(ctx, "client") and hasattr(ctx, "store"):
        client = ctx.client
        store = ctx.store
    elif isinstance(ctx, dict) and "client" in ctx and "store" in ctx:
        client = ctx["client"]
        store = ctx["store"]
    else:
        client = Groq(max_retries=10)
        store = Store()

    tools = Tools(store, ticket)
    reasoning_trace = []
    visible = {k: v for k, v in ticket.items() if k not in HIDDEN_FIELDS}
    chat = GroqChat(client, SYSTEM_PROMPT, f"Triage this ticket:\n\n{json.dumps(visible, indent=2)}")

    for _ in range(MAX_ROUNDS):
        calls = chat.turn()
        if not calls:
            break
        results = []
        for call_id, name, args in calls:
            args = dict(args)
            reason = args.pop("reason", None)
            output = json.loads(json.dumps(tools.run(name, args), default=str))
            obs = json.dumps(output)
            reasoning_trace.append({
                "step": len(reasoning_trace) + 1,
                "action": name,
                "input": f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})" + (f" - {reason}" if reason else ""),
                "observation": obs[:300] + ("..." if len(obs) > 300 else ""),
            })
            results.append((call_id, name, output))
        chat.send_results(results)

    if not reasoning_trace:
        reasoning_trace.append({
            "step": 1,
            "action": "inspect_ticket",
            "input": f"ticket_id={ticket.get('ticket_id')}",
            "observation": f"Directly triaged based on provided ticket intake: {ticket.get('subject', '')}",
        })

    instruction = "Write the final triage record as JSON matching the Decision schema. Only use IDs that appeared in tool results."
    decision = None
    output_dict = None
    problems = []

    for _ in range(2):
        try:
            raw_json = chat.final(instruction)
            decision = Decision.model_validate_json(raw_json)
            output_dict = build_output_dict(ticket, decision, tools, reasoning_trace)
            problems = check(output_dict, tools)
        except ValidationError as exc:
            problems = [f"JSON schema mismatch: {exc}"]
        except Exception as exc:
            problems = [f"Parsing error: {exc}"]

        if not problems:
            break
        instruction = "Fix these problems and output the complete JSON record again:\n- " + "\n- ".join(problems)

    if output_dict is None:
        raise RuntimeError(problems[0] if problems else "Failed to generate valid triage record")

    return output_dict


def build_output_dict(ticket, decision, tools, reasoning_trace):
    account_signals = list(decision.account_signals)
    account_id = decision.account_id

    # Supplement account signals from resolved account if available
    if account_id and account_id in tools.store.accounts:
        acc = tools.get_account(account_id)
        days = acc.get("days_to_renewal")
        if days is not None and not any("renewal" in s.lower() for s in account_signals):
            account_signals.append(f"Renewal in {days} days")
        if acc.get("plan_tier") and not any("plan" in s.lower() or "tier" in s.lower() for s in account_signals):
            account_signals.append(f"Plan: {acc['plan_tier']}")
        if acc.get("health_status") and not any("health" in s.lower() for s in account_signals):
            account_signals.append(f"Health: {acc['health_status']}")
        if acc.get("usage_trend") and not any("usage" in s.lower() for s in account_signals):
            account_signals.append(f"Usage: {acc['usage_trend']}")

    out = {
        "ticket_id": ticket["ticket_id"],
        "account_id": account_id,
        "category": decision.category,
        "priority": decision.priority,
        "route_to": decision.route_to,
        "escalate": decision.escalate,
        "notify_tam": decision.notify_tam,
        "duplicate_of": decision.duplicate_of,
        "needs_clarification": decision.needs_clarification,
        "kb_references": decision.kb_references,
        "account_signals": account_signals,
        "customer_reply": decision.customer_reply,
        "internal_note": decision.internal_note,
        "confidence": round(float(decision.confidence), 2),
        "reasoning_trace": reasoning_trace,
    }
    return out


def check(output_dict, tools):
    problems = validate(output_dict)
    account_id = output_dict.get("account_id")
    if account_id and account_id not in tools.store.accounts:
        problems.append(f"account_id {account_id} does not exist in accounts")

    duplicate_of = output_dict.get("duplicate_of")
    if duplicate_of:
        if tools.get_ticket(duplicate_of).get("found") is False:
            problems.append(f"{duplicate_of} is not an earlier ticket in the data")
        if output_dict.get("category") != "duplicate":
            problems.append("duplicate_of is set, so category must be 'duplicate'")
    elif output_dict.get("category") == "duplicate":
        problems.append("category is 'duplicate' but duplicate_of is null")

    if output_dict.get("category") == "feature_request" and output_dict.get("priority") != "P4":
        problems.append("Policy: feature requests must have priority P4")

    if output_dict.get("category") == "needs_info" and not output_dict.get("needs_clarification"):
        problems.append("category is 'needs_info' but needs_clarification is False")

    for ref in output_dict.get("kb_references", []):
        section = tools.store.kb.get(ref)
        if section is None and not any(k.startswith(ref + "#") for k in tools.store.kb):
            problems.append(f"{ref} is not a valid knowledge-base section/article")
        elif section and section.get("status") != "current":
            problems.append(f"{ref} is deprecated (replaced by {section.get('replaced_by')}); use the replacement")

    if re.search(r"churn|\bARR\b|\bNPS\b|at[- ]risk|escalation note", output_dict.get("customer_reply", ""), re.I):
        problems.append("customer_reply mentions internal account data")

    account = tools.store.accounts.get(account_id)
    if account and not output_dict.get("notify_tam"):
        days = tools.get_account(account["account_id"]).get("days_to_renewal")
        signals = [s for s in (account.get("health_status") in ("At Risk", "Churning") and f"health is {account.get('health_status')}",
                               account.get("usage_trend") in ("Declining", "Inactive") and f"usage is {account.get('usage_trend')}",
                               days is not None and 0 <= days <= 90 and f"renewal is in {days} days") if s]
        if signals:
            problems.append(f"Policy requires notify_tam=True: {', '.join(signals)}")

    return problems


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Triage support tickets with an AI agent.")
    parser.add_argument("ticket_ids", nargs="*", help="ticket IDs, e.g. TKT-10042")
    parser.add_argument("--sample", type=int, help="triage N random tickets")
    parser.add_argument("--set", choices=["review"], help="run review set")
    parser.add_argument("--data", default="data")
    parser.add_argument("--kb", default="knowledge-base")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    if not args.ticket_ids and not args.sample and not args.set:
        parser.error("give ticket IDs, --sample N, or --set review")

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY in .env.")

    store = Store(args.data, args.kb)
    client = Groq(max_retries=10)
    ctx = {"client": client, "store": store}
    Path(args.out).mkdir(exist_ok=True)

    if args.set == "review":
        review_file = Path(args.data, "review_set.txt")
        if review_file.exists():
            ticket_ids = [line.strip() for line in review_file.read_text().splitlines() if line.strip()]
        else:
            ticket_ids = [t["ticket_id"] for t in store.tickets[:10]]
        tickets = [store.ticket(tid) for tid in ticket_ids]
    elif args.sample:
        tickets = random.Random(0).sample(store.tickets, args.sample)
        ticket_ids = [t["ticket_id"] for t in tickets]
    else:
        ticket_ids = args.ticket_ids
        tickets = [store.ticket(tid) for tid in ticket_ids]

    run_log_path = Path(args.out, "run_log.jsonl")

    for ticket_id, ticket in zip(ticket_ids, tickets):
        if ticket is None:
            print(f"{ticket_id}: not found in tickets.json")
            continue
        try:
            result = triage(ticket, ctx)
        except Exception as exc:
            print(f"{ticket_id}: failed - {exc}")
            continue

        path = Path(args.out, f"{ticket_id}.json")
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        with run_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ticket_id": ticket_id, "status": "success", "category": result["category"], "priority": result["priority"]}) + "\n")

        print(f"{ticket_id}: {result['priority']} {result['category']} -> {result['route_to']} | "
              f"escalate={result['escalate']} notify_tam={result['notify_tam']} "
              f"duplicate={result['duplicate_of']} | {len(result['reasoning_trace'])} steps -> {path}")


if __name__ == "__main__":
    main()
