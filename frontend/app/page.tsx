"use client";

import { useEffect, useRef, useState } from "react";
import { streamOrchestrationScenario } from "@/lib/api";
import type { OrchestrationEvent } from "@/lib/api";

const agentNames = ["coordinator", "risk", "logistics", "evacuation"] as const;
type AgentName = (typeof agentNames)[number];
type AgentStatus = "waiting" | "running" | "verified" | "rejected" | "abstained";
type RunStatus = "idle" | "running" | "complete" | "partial" | "insufficient_data" | "failed" | "cancelled";

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
  quantities?: Array<Record<string, unknown>>;
  model_ids?: string[];
  dataset_version?: string;
};

function object(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function text(value: unknown, empty = "Not provided") {
  return typeof value === "string" && value.trim() ? value : empty;
}

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(object) : [];
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0) : [];
}

function numberText(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not provided";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
}

function readableTool(value: unknown): string {
  const labels: Record<string, string> = {
    geocode_region: "Open-Meteo geocoding",
    get_weather: "Open-Meteo forecast",
    search_precedents: "EM-DAT historical retrieval (Pinecone)",
    get_historical_stats: "Historical disaster statistics",
    compute_logistics_requirements: "Sphere planning standards and supplied inventory",
  };
  const key = typeof value === "string" ? value : "";
  return labels[key] ?? (key ? key.replaceAll("_", " ") : "Verified source");
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
  const briefingRef = useRef<HTMLElement | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus>("idle");
  const [activity, setActivity] = useState("Ready to run");

  useEffect(() => {
    if (!briefing) return;
    requestAnimationFrame(() => {
      briefingRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }, [briefing]);

  function receiveEvent(event: OrchestrationEvent) {
    setEvents((current) => [...current, event]);
    const eventData = object(event.data);
    const agent = event.agent && agentNames.includes(event.agent as AgentName) ? event.agent as AgentName : null;
    const tool = typeof eventData.tool === "string" ? readableTool(eventData.tool) : "a verified data source";
    if (event.event === "run_started") setActivity("Run started · gathering scenario evidence");
    if (event.event === "agent_started" && agent) setActivity(`${agent} agent · analysis started`);
    if (event.event === "agent_retrying" && agent) {
      const delay = typeof eventData.delay_s === "number" ? Math.ceil(eventData.delay_s) : 30;
      const attempt = typeof eventData.retry_number === "number" ? eventData.retry_number : 1;
      setActivity(`${agent} agent · NIM rate-limited; retrying the same request in ${delay}s (retry ${attempt}/1)`);
    }
    if (event.event === "tool_called") setActivity(`${agent ?? "Workflow"} · querying ${tool}`);
    if (event.event === "tool_result") {
      const resultStatus = text(eventData.status, "returned");
      setActivity(`${agent ?? "Workflow"} · ${tool} ${resultStatus.replaceAll("_", " ")}`);
    }
    if (event.event === "agent_verified" && agent) setActivity(`${agent} agent · evidence verified`);
    if (event.event === "agent_rejected" && agent) setActivity(`${agent} agent · output rejected; validating again`);
    if (event.event === "agent_abstained" && agent) setActivity(`${agent} agent · abstained; missing evidence will be shown`);
    if (event.event === "briefing_ready") setActivity("Briefing assembled · checking final status");
    if (event.event === "run_failed") setActivity("Run failed · see the error details");
    if (!agent) return;
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
    setRunStatus("running");
    setActivity("Connecting to the orchestration service");
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
      const briefingStatus = (result as Briefing).status;
      const finalRunStatus: RunStatus = briefingStatus === "complete" || briefingStatus === "partial" || briefingStatus === "insufficient_data"
        ? briefingStatus
        : "failed";
      setRunStatus(finalRunStatus);
      setActivity(`Run finished · ${text(briefingStatus, "status unavailable").replaceAll("_", " ")}`);
    } catch (cause) {
      if (controller.signal.aborted) {
        setRunStatus("cancelled");
        setActivity("Run cancelled");
      } else {
        setRunStatus("failed");
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
  const settledAgentCount = Object.values(statuses).filter((status) => status === "verified" || status === "rejected" || status === "abstained").length;
  const verifiedSources = Array.from(new Set(events
    .filter((event) => event.event === "tool_result" && object(event.data).status === "ok")
    .map((event) => readableTool(object(event.data).tool))));
  const capacityComparisons = records(logistics.capacity_comparisons);
  const supplyRecommendations = strings(logistics.supply_recommendations);
  const evacuationSteps = strings(evacuation.trigger_criteria);
  const priorityZones = strings(evacuation.priority_zones);
  const actionPlan = strings(briefing?.action_plan);
  const citations = records(briefing?.citations);
  const models = strings(briefing?.model_ids);
  const runStatusLabel: Record<RunStatus, string> = {
    idle: "Ready",
    running: "In progress",
    complete: "Complete",
    partial: "Partial · some agents abstained or were unavailable",
    insufficient_data: "Insufficient data · do not act on this briefing",
    failed: "Failed",
    cancelled: "Cancelled",
  };

  return (
    <main className="mx-auto min-h-screen max-w-5xl space-y-6 px-4 py-8 text-[var(--foreground)]">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-bold uppercase text-[var(--green)]">DisasterMind AI</p>
          <h1 className="text-3xl font-bold">Disaster response briefing</h1>
          <p className="mt-2 max-w-2xl text-sm text-[var(--muted)]">Evidence-grounded decision support for trained emergency personnel.</p>
        </div>
        <span className="status-pill">Official review required</span>
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
        <div className="panel-header flex-wrap"><h2 id="agents-heading" className="font-bold">Run status</h2><span className={`status-pill ${runStatus === "partial" || runStatus === "insufficient_data" || runStatus === "failed" || runStatus === "cancelled" ? "warning" : ""}`}>{runStatusLabel[runStatus]}</span></div>
        <div className="panel-body space-y-3">
          <div className="progress-track" role="progressbar" aria-label="Briefing run progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={runStatus !== "idle" && runStatus !== "running" ? 100 : undefined} aria-valuetext={activity}>
            <div className="progress-indicator" data-state={runStatus} />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <p aria-live="polite">{activity}</p>
            <p className="text-[var(--muted)]">Agents settled: {settledAgentCount} / {agentNames.length}</p>
          </div>
        </div>
        <div className="panel-header"><h2 className="font-bold">Agent progress</h2></div>
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
        <section className="panel scroll-mt-6" aria-labelledby="briefing-heading" ref={briefingRef}>
          <div className="panel-header flex-wrap">
            <h2 id="briefing-heading" className="font-bold" tabIndex={-1}>Verified briefing</h2>
            <span className="status-pill">{text(briefing.label, "Scenario classification unavailable").replaceAll("_", " ")}</span>
          </div>
          <div className="panel-body space-y-5">
            <p className="text-sm font-semibold">Status: {text(briefing.status)}</p>
            <p>{text(briefing.executive_summary)}</p>
            {briefing.human_review?.required !== false && <aside className="rounded-lg border border-amber-500 bg-amber-50 p-4" role="note">
              <strong>Official human authorization is required before evacuation or resource deployment.</strong>
              {briefing.human_review?.reasons?.length ? <ul className="mt-2 list-disc pl-5">{briefing.human_review.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul> : null}
            </aside>}
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-lg border border-[var(--line)] p-4"><h3 className="font-bold">Risk assessment</h3><p className="mt-1 text-sm font-semibold">Model assessment: {text(risk.level)}</p><p className="text-sm">{text(risk.summary, "No verified risk assessment was returned.")}</p><dl className="mt-3 space-y-1 text-sm"><div><dt className="inline font-semibold">Assessment agent: </dt><dd className="inline">{text(risk.verified_by_agent, "No verified risk agent result")}</dd></div><div><dt className="inline font-semibold">Physical premise: </dt><dd className="inline">{typeof risk.premise_valid === "boolean" ? (risk.premise_valid ? "valid" : `invalid · ${text(risk.premise_issue)}`) : "not assessed"}</dd></div></dl><p className="mt-3 text-xs text-[var(--muted)]">This is a model assessment, not a measured probability or a live hazard alert. Historical retrieval provides context and does not establish current local conditions.</p></div>
              <div className="rounded-lg border border-[var(--line)] p-4"><h3 className="font-bold">Evacuation recommendation</h3><p className="mt-1 text-sm font-semibold">{text(evacuation.recommendation).replaceAll("_", " ")}</p>{typeof evacuation.target_headcount === "number" && <p className="text-sm">Target headcount: {numberText(evacuation.target_headcount)} · source: {text(evacuation.target_headcount_provenance)}</p>}<p className="mt-2 text-sm">Official authorization: {evacuation.requires_human_approval === true ? "Required" : "Not returned"}</p><h4 className="mt-4 font-semibold">Verified triggers and conditions</h4>{evacuationSteps.length ? <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm">{evacuationSteps.map((step, i) => <li key={i}>{step}</li>)}</ol> : <p className="mt-1 text-sm">No verified evacuation triggers or conditions were returned.</p>}{priorityZones.length > 0 && <><h4 className="mt-4 font-semibold">Priority zones identified</h4><ul className="mt-1 list-disc pl-5 text-sm">{priorityZones.map((zone, i) => <li key={i}>{zone}</li>)}</ul></>}<p className="mt-3 text-xs text-[var(--muted)]">No verified shelter destinations or route data are connected to this briefing. Use current instructions from local emergency authorities; do not infer a destination from historical events.</p></div>
            </div>
            <div className="rounded-lg border border-[var(--line)] p-4"><h3 className="font-bold">Logistics and available resources</h3><p className="mt-1 text-sm">Resource adequacy: {text(logistics.resource_adequacy, "No verified logistics assessment")}</p>{capacityComparisons.length ? <div className="mt-3 overflow-x-auto"><table className="table text-sm"><thead><tr><th>Resource</th><th>Required</th><th>Available</th><th>Difference</th></tr></thead><tbody>{capacityComparisons.map((item, i) => <tr key={`${text(item.resource)}-${i}`}><td>{text(item.resource).replaceAll("_", " ")}</td><td>{numberText(item.required)} {text(item.required_unit, "")}</td><td>{item.available_status === "provided_by_user" && typeof item.available === "number" ? `${numberText(item.available)} (user supplied)` : "Not provided"}</td><td>{typeof item.deficit_or_surplus === "number" ? numberText(item.deficit_or_surplus) : "Not calculable"}</td></tr>)}</tbody></table></div> : <p className="mt-2 text-sm">No verified resource inventory was returned.</p>}{supplyRecommendations.length > 0 && <><h4 className="mt-4 font-semibold">Agent-proposed supply actions</h4><ul className="mt-1 list-disc space-y-1 pl-5 text-sm">{supplyRecommendations.map((item, i) => <li key={i}>{item}</li>)}</ul></>}<p className="mt-3 text-xs text-[var(--muted)]">Shelter names, current occupancy, open/closed status, and verified destinations are not available from the connected sources. Availability must be confirmed with local authorities before dispatch.</p></div>
            <div><h3 className="font-bold">Agent-proposed response steps</h3>{actionPlan.length ? <ol className="mt-2 list-decimal space-y-1 pl-5">{actionPlan.map((item, i) => <li key={i}>{item}</li>)}</ol> : <p className="text-sm">No verified response instructions were provided for this run.</p>}<p className="mt-2 text-xs text-[var(--muted)]">Confirm each step against current field conditions and official emergency instructions.</p></div>
            <div className="rounded-lg bg-[var(--panel-muted)] p-4"><h3 className="font-bold">Assessment sources</h3>{verifiedSources.length ? <ul className="mt-2 flex flex-wrap gap-2">{verifiedSources.map((source) => <li className="status-pill" key={source}>{source}</li>)}</ul> : <p className="mt-1 text-sm">No successful external data-tool results were recorded.</p>}<p className="mt-3 text-sm">Historical citations in this briefing: {citations.length}</p><p className="text-sm">Dataset version: {text(briefing.dataset_version)}</p><p className="text-sm">Model(s): {models.length ? models.join(", ") : "Not reported"}</p></div>
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
