"use client";

import { useRef, useState } from "react";
import { streamOrchestrationScenario } from "@/lib/api";
import type { OrchestrationEvent } from "@/lib/api";

const agentNames = ["coordinator", "risk", "logistics", "evacuation"] as const;
type AgentName = (typeof agentNames)[number];
type AgentStatus = "waiting" | "running" | "verified" | "rejected" | "abstained";

type Briefing = {
  status?: string;
  label?: "historical_precedent" | "hypothetical_scenario";
  executive_summary?: string;
  action_plan?: string[];
  risk?: Record<string, unknown>;
  evacuation_recommendation?: Record<string, unknown>;
  logistics_summary?: Record<string, unknown> | null;
  citations?: Array<Record<string, unknown>>;
  data_gaps?: string[];
  disclaimers?: string[];
  human_review?: { required?: boolean; reasons?: string[] };
  conflicts?: Array<Record<string, unknown>>;
  abstentions?: Array<Record<string, unknown>>;
};

function object(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function text(value: unknown, empty = "Not provided") {
  return typeof value === "string" && value.trim() ? value : empty;
}

export default function Home() {
  const [disasterType, setDisasterType] = useState("");
  const [region, setRegion] = useState("");
  const [severity, setSeverity] = useState<"Low" | "Moderate" | "High" | "Critical">("Moderate");
  const [timeHorizon, setTimeHorizon] = useState("24 hours");
  const [notes, setNotes] = useState("");
  const [statuses, setStatuses] = useState<Record<AgentName, AgentStatus>>({
    coordinator: "waiting", risk: "waiting", logistics: "waiting", evacuation: "waiting",
  });
  const [events, setEvents] = useState<OrchestrationEvent[]>([]);
  const [briefing, setBriefing] = useState<Briefing | null>(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  function receiveEvent(event: OrchestrationEvent) {
    setEvents((current) => [...current, event]);
    if (!event.agent || !agentNames.includes(event.agent as AgentName)) return;
    const agent = event.agent as AgentName;
    const next: Partial<Record<string, AgentStatus>> = {
      agent_started: "running",
      agent_verified: "verified",
      agent_rejected: "rejected",
      agent_abstained: "abstained",
    };
    if (next[event.event]) setStatuses((current) => ({ ...current, [agent]: next[event.event] }));
  }

  async function run() {
    if (!disasterType.trim() || !region.trim() || running) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError("");
    setBriefing(null);
    setEvents([]);
    setStatuses({ coordinator: "waiting", risk: "waiting", logistics: "waiting", evacuation: "waiting" });
    try {
      const result = await streamOrchestrationScenario({
        disaster_type: disasterType.trim(), region: region.trim(), severity,
        time_horizon: timeHorizon.trim(), notes: notes.slice(0, 500),
      }, receiveEvent, controller.signal);
      setBriefing(result as Briefing);
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : "The orchestration service is unavailable.");
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }

  const risk = object(briefing?.risk);
  const evacuation = object(briefing?.evacuation_recommendation);
  const logistics = object(briefing?.logistics_summary);

  return (
    <main className="mx-auto min-h-screen max-w-5xl space-y-6 px-4 py-8 text-[var(--foreground)]">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-bold uppercase text-[var(--green)]">DisasterMind AI</p>
          <h1 className="text-3xl font-bold">Disaster response briefing</h1>
          <p className="mt-2 max-w-2xl text-sm text-[var(--muted)]">Evidence-grounded decision support for trained emergency personnel.</p>
        </div>
        <span className="status-pill">Human review required</span>
      </header>

      <section className="panel" aria-labelledby="scenario-heading">
        <div className="panel-header"><h2 id="scenario-heading" className="font-bold">Scenario</h2></div>
        <div className="panel-body grid gap-4 sm:grid-cols-2">
          <label className="field">Disaster type
            <select value={disasterType} onChange={(e) => setDisasterType(e.target.value)}>
              <option value="">Select disaster</option><option>Flood</option><option>Cyclone</option>
              <option>Earthquake</option><option>Wildfire</option><option>Storm</option>
            </select>
          </label>
          <label className="field">Region
            <input value={region} maxLength={200} onChange={(e) => setRegion(e.target.value)} placeholder="City, district, or operating zone" />
          </label>
          <label className="field">Severity
            <select value={severity} onChange={(e) => setSeverity(e.target.value as typeof severity)}>
              <option>Low</option><option>Moderate</option><option>High</option><option>Critical</option>
            </select>
          </label>
          <label className="field">Planning horizon
            <input value={timeHorizon} maxLength={100} onChange={(e) => setTimeHorizon(e.target.value)} />
          </label>
          <label className="field sm:col-span-2">Operational notes <span className="font-normal text-[var(--muted)]">({notes.length}/500)</span>
            <textarea value={notes} maxLength={500} onChange={(e) => setNotes(e.target.value)} placeholder="Known road closures, facility constraints, or verified local information" />
          </label>
          <div className="flex gap-3 sm:col-span-2">
            <button className="primary-button" disabled={!disasterType || !region || running} onClick={run} type="button">
              {running ? "Running verified agents…" : "Generate briefing"}
            </button>
            {running && <button className="secondary-button" onClick={() => abortRef.current?.abort()} type="button">Cancel</button>}
          </div>
          {error && <p role="alert" className="warning sm:col-span-2">Service unavailable or run incomplete: {error}</p>}
        </div>
      </section>

      <section className="panel" aria-labelledby="agents-heading">
        <div className="panel-header"><h2 id="agents-heading" className="font-bold">Agent progress</h2></div>
        <ol className="panel-body grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {agentNames.map((agent) => <li key={agent} className="rounded-lg border border-[var(--line)] p-3">
            <span className="font-bold capitalize">{agent}</span>
            <span className="mt-1 block text-sm text-[var(--muted)]">{statuses[agent]}</span>
          </li>)}
        </ol>
        {events.length > 0 && <details className="border-t border-[var(--line)] p-4">
          <summary className="cursor-pointer font-semibold">Run activity ({events.length} events)</summary>
          <ul className="mt-3 space-y-1 text-sm" aria-live="polite">
            {events.map((event, index) => <li key={`${event.run_id}-${index}`}>{event.event.replaceAll("_", " ")}{event.agent ? ` — ${event.agent}` : ""}</li>)}
          </ul>
        </details>}
      </section>

      {briefing && <>
        <section className="panel" aria-labelledby="briefing-heading">
          <div className="panel-header flex-wrap">
            <h2 id="briefing-heading" className="font-bold">Verified briefing</h2>
            <span className="status-pill">{text(briefing.label, "Scenario classification unavailable").replaceAll("_", " ")}</span>
          </div>
          <div className="panel-body space-y-5">
            <p className="text-sm font-semibold">Status: {text(briefing.status)}</p>
            <p>{text(briefing.executive_summary)}</p>
            {briefing.human_review?.required !== false && <aside className="rounded-lg border border-amber-500 bg-amber-50 p-4" role="note">
              <strong>Human approval required before any operational action.</strong>
              {briefing.human_review?.reasons?.length ? <ul className="mt-2 list-disc pl-5">{briefing.human_review.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul> : null}
            </aside>}
            <div className="grid gap-4 sm:grid-cols-2">
              <div><h3 className="font-bold">Risk assessment</h3><p className="mt-1 text-sm">Level: {text(risk.level)}</p><p className="text-sm">{text(risk.summary)}</p><p className="text-sm">Premise valid: {typeof risk.premise_valid === "boolean" ? String(risk.premise_valid) : "Not assessed"}</p></div>
              <div><h3 className="font-bold">Evacuation recommendation</h3><p className="mt-1 text-sm">{text(evacuation.recommendation).replaceAll("_", " ")}</p><p className="text-sm">Human approval: {evacuation.requires_human_approval === true ? "Required" : "Not stated"}</p></div>
            </div>
            <div><h3 className="font-bold">Logistics</h3><p className="mt-1 text-sm">Resource adequacy: {text(logistics.resource_adequacy)}</p></div>
            <div><h3 className="font-bold">Action plan</h3>{briefing.action_plan?.length ? <ol className="mt-2 list-decimal space-y-1 pl-5">{briefing.action_plan.map((item, i) => <li key={i}>{item}</li>)}</ol> : <p className="text-sm">No verified actions provided.</p>}</div>
          </div>
        </section>
        <section className="panel">
          <div className="panel-header"><h2 className="font-bold">Evidence and limitations</h2></div>
          <div className="panel-body space-y-4">
            <div><h3 className="font-semibold">Citations</h3>{briefing.citations?.length ? <ul className="mt-2 list-disc space-y-1 pl-5">{briefing.citations.map((citation, i) => <li key={i}>{text(citation.event_name, text(citation.record_id, "Source"))} — {text(citation.country)}, {text(citation.start_year)}; source ID {text(citation.record_id)}</li>)}</ul> : <p className="text-sm">No citations returned.</p>}</div>
            <div><h3 className="font-semibold">Data gaps</h3>{briefing.data_gaps?.length ? <ul className="mt-2 list-disc pl-5">{briefing.data_gaps.map((gap, i) => <li key={i}>{gap}</li>)}</ul> : <p className="text-sm">None reported by the agents.</p>}</div>
            {briefing.conflicts?.length ? <div><h3 className="font-semibold">Conflicts</h3><ul className="list-disc pl-5">{briefing.conflicts.map((item, i) => <li key={i}>{text(object(item).description)}</li>)}</ul></div> : null}
            {briefing.abstentions?.length ? <div><h3 className="font-semibold">Agent abstentions</h3><ul className="list-disc pl-5">{briefing.abstentions.map((item, i) => <li key={i}>{text(object(item).agent)}: {text(object(item).reason)}</li>)}</ul></div> : null}
            {briefing.disclaimers?.map((disclaimer, i) => <p className="text-sm text-[var(--muted)]" key={i}>{disclaimer}</p>)}
            <p className="border-t border-[var(--line)] pt-3 text-sm font-semibold">This briefing is decision support, not an emergency order. Confirm conditions and obtain authorization from responsible officials.</p>
          </div>
        </section>
      </>}
    </main>
  );
}
