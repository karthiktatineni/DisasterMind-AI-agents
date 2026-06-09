"use client";

import {
  Activity,
  Ambulance,
  BadgeCheck,
  ClipboardList,
  DatabaseZap,
  FileText,
  Gauge,
  Hospital,
  Layers,
  MapPinned,
  Play,
  RadioTower,
  Route,
  ShieldCheck,
  Siren,
  Users,
  Warehouse,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { askAgents, orchestrateScenario } from "@/lib/api";
import type { AgentResult, OrchestrationResponse } from "@/lib/api";

type Severity = "Low" | "Moderate" | "High" | "Critical";
type AgentState = "idle" | "running" | "complete";

type Scenario = {
  disasterType: string;
  region: string;
  timeHorizon: string;
  severity: Severity;
  notes: string;
};

type Agent = {
  name: string;
  icon: LucideIcon;
};

const initialScenario: Scenario = {
  disasterType: "",
  region: "",
  timeHorizon: "24 hours",
  severity: "Moderate",
  notes: "",
};

const agents: Agent[] = [
  { name: "Coordinator", icon: ClipboardList },
  { name: "Intelligence", icon: DatabaseZap },
  { name: "Risk", icon: Gauge },
  { name: "Prediction", icon: Activity },
  { name: "Logistics", icon: Warehouse },
  { name: "Evacuation", icon: Route },
  { name: "Simulation", icon: Layers },
  { name: "Validation", icon: ShieldCheck },
  { name: "Planner", icon: BadgeCheck },
  { name: "Decision", icon: Siren },
  { name: "Reporting", icon: FileText },
];

const navItems = [
  { label: "Scenario", id: "scenario", icon: ClipboardList, always: true },
  { label: "Dashboard", id: "dashboard", icon: Gauge },
  { label: "Agents", id: "agents", icon: Activity },
  { label: "Risk", id: "risk", icon: MapPinned },
  { label: "Resources", id: "resources", icon: Ambulance },
  { label: "Simulation", id: "simulation", icon: Layers },
  { label: "Reports", id: "reports", icon: FileText },
];

function numberFormat(value: number) {
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

function getAgentState(index: number, activeAgent: number, completedAgents: number) {
  if (index < completedAgents) {
    return "complete" satisfies AgentState;
  }
  if (index === activeAgent) {
    return "running" satisfies AgentState;
  }
  return "idle" satisfies AgentState;
}

function getAgent(
  orchestration: OrchestrationResponse | null,
  name: string,
): AgentResult | undefined {
  return orchestration?.agent_results.find(
    (result) => result.agent_name.toLowerCase() === name.toLowerCase(),
  );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value)
    ? value.filter((item) => item && typeof item === "object") as Array<Record<string, unknown>>
    : [];
}

function asNumber(value: unknown, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function buildRiskTrend(riskScore: number, damageScore: number) {
  const baseRisk = Math.max(8, riskScore * 100 - 28);
  const baseImpact = Math.max(5, damageScore * 100 - 24);
  return ["0h", "6h", "12h", "18h", "24h", "36h", "48h"].map((hour, index) => ({
    hour,
    risk: Math.min(99, Math.round(baseRisk + index * 5.6)),
    impact: Math.min(99, Math.round(baseImpact + index * 4.9)),
  }));
}

function fallbackReport(scenario: Scenario) {
  return [
    "No agent report yet.",
    "",
    "Enter the disaster scenario, capacity, weather, and operational notes, then run agents.",
    `Current draft: ${scenario.disasterType || "No disaster selected"} in ${
      scenario.region || "no region selected"
    }.`,
  ].join("\n");
}

function buildIntake(scenario: Scenario) {
  return {
    disaster_type: scenario.disasterType,
    region: scenario.region,
    time_horizon: scenario.timeHorizon,
    severity: scenario.severity,
    notes: scenario.notes,
  };
}

export default function Home() {
  const [activeNav, setActiveNav] = useState("Scenario");
  const [scenario, setScenario] = useState<Scenario>(initialScenario);
  const [isRunning, setIsRunning] = useState(false);
  const [activeAgent, setActiveAgent] = useState(-1);
  const [completedAgents, setCompletedAgents] = useState(0);
  const [orchestration, setOrchestration] = useState<OrchestrationResponse | null>(null);
  const [report, setReport] = useState(fallbackReport(initialScenario));
  const [serviceState, setServiceState] = useState<"ready" | "offline" | "live">("ready");
  const [chartsReady, setChartsReady] = useState(false);
  const [question, setQuestion] = useState("");
  const [agentAnswer, setAgentAnswer] = useState("");
  const [isAnswering, setIsAnswering] = useState(false);

  useEffect(() => {
    setChartsReady(true);
  }, []);

  const canRun = Boolean(
    scenario.disasterType &&
      scenario.region,
  );
  const intake = useMemo(() => buildIntake(scenario), [scenario]);

  const riskAgent = getAgent(orchestration, "risk");
  const predictionAgent = getAgent(orchestration, "prediction");
  const logisticsAgent = getAgent(orchestration, "logistics");
  const evacuationAgent = getAgent(orchestration, "evacuation");
  const simulationAgent = getAgent(orchestration, "simulation");

  const riskOutput = asRecord(riskAgent?.output);
  const predictionOutput = asRecord(predictionAgent?.output);
  const logisticsOutput = asRecord(logisticsAgent?.output);
  const evacuationOutput = asRecord(evacuationAgent?.output);
  const simulationOutput = asRecord(simulationAgent?.output);

  const riskScore = asNumber(riskOutput.risk_score);
  const damageScore = asNumber(predictionOutput.damage_score);
  const affectedPopulation = asNumber(riskOutput.affected_population);
  const evacuationNeed = asNumber(predictionOutput.evacuation_need);
  const casualtyEstimate = asNumber(predictionOutput.casualty_estimate);
  const shelterGap = asNumber(evacuationOutput.shelter_gap);
  const riskTrend = buildRiskTrend(riskScore, damageScore);

  const resourcePlan = asRecord(logisticsOutput.resource_plan);
  const availableResources = asRecord(logisticsOutput.available_resources);
  const resourceChart = Object.entries(resourcePlan).map(([name, required]) => ({
    name: name.replaceAll("_", " "),
    required: asNumber(required),
    available: asNumber(availableResources[name], asNumber(required) * 0.8),
  }));

  const simulationRows = asArray(simulationOutput.scenario_results);
  const intelligenceOutput = asRecord(getAgent(orchestration, "intelligence")?.output);
  const groundedContext = asRecord(intelligenceOutput.grounded_context);
  const sourceRows = asArray(intelligenceOutput.data_sources);
  const weatherEvidence = getAgent(orchestration, "intelligence")?.evidence.find(
    (item) => String(item.title).toLowerCase().includes("weather"),
  );
  const sourceScenario = asRecord(logisticsOutput.source_scenario);
  const hasResults = Boolean(orchestration);

  function updateScenario<K extends keyof Scenario>(key: K, value: Scenario[K]) {
    setScenario((current) => ({ ...current, [key]: value }));
  }

  function scrollToSection(id: string) {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    setActiveNav(navItems.find((item) => item.id === id)?.label ?? activeNav);
  }

  async function runAgents() {
    if (!canRun) {
      return;
    }

    setIsRunning(true);
    setServiceState("ready");
    setCompletedAgents(0);
    setActiveAgent(0);
    setOrchestration(null);
    setReport("Agents are running. The report will appear after validation and decision synthesis.");

    for (let index = 0; index < agents.length; index += 1) {
      setActiveAgent(index);
      await new Promise((resolve) => setTimeout(resolve, 130));
      setCompletedAgents(index + 1);
    }

    try {
      const response = await orchestrateScenario({
        ...intake,
      });
      setOrchestration(response);
      setReport(response.final_report);
      setServiceState(response.provider === "nvidia-build" ? "live" : "offline");
      setActiveNav("Dashboard");
      setTimeout(() => scrollToSection("dashboard"), 100);
    } catch {
      setServiceState("offline");
      setReport("Orchestration failed. Check backend logs, NVIDIA key, and network access.");
    } finally {
      setActiveAgent(-1);
      setIsRunning(false);
    }
  }

  async function askRealtimeAgents() {
    if (!canRun || !question.trim()) {
      return;
    }
    setIsAnswering(true);
    setAgentAnswer("");
    try {
      const response = await askAgents(intake, question.trim());
      setAgentAnswer(response.answer);
    } catch {
      setAgentAnswer("The real-time agent answer service failed. Check backend logs and NVIDIA access.");
    } finally {
      setIsAnswering(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="mb-7 flex items-center gap-3">
          <div className="brand-mark">
            <Siren size={22} aria-hidden="true" />
          </div>
          <div>
            <h1 className="text-lg font-bold">DisasterMind AI</h1>
            <p className="text-sm text-[var(--muted)]">Agent operations</p>
          </div>
        </div>

        <nav className="nav-list grid gap-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const disabled = !item.always && !hasResults && !isRunning;
            return (
              <button
                className="nav-button"
                data-active={activeNav === item.label}
                key={item.label}
                onClick={() => {
                  if (disabled) {
                    scrollToSection("scenario");
                    return;
                  }
                  scrollToSection(item.id);
                }}
                type="button"
                title={disabled ? "Run agents first" : item.label}
              >
                <Icon size={18} aria-hidden="true" />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="mt-8 rounded-lg border border-[var(--line)] bg-[var(--panel-muted)] p-3">
          <div className="flex items-center gap-2 text-sm font-bold">
            <RadioTower size={16} aria-hidden="true" />
            NVIDIA Build
          </div>
          <p className="mt-2 text-sm text-[var(--muted)]">
            {serviceState === "live"
              ? "Live agent report"
              : serviceState === "offline"
                ? "Fallback or error"
                : "Ready"}
          </p>
        </div>
      </aside>

      <section className="main">
        <header className="topbar">
          <div>
            <p className="text-sm font-bold uppercase text-[var(--green)]">
              {hasResults ? "Orchestrated response" : "Scenario intake"}
            </p>
            <h2 className="text-2xl font-bold">
              {hasResults
                ? `${scenario.disasterType} response plan`
                : "Tell the agents what happened"}
            </h2>
          </div>
          <button
            className="primary-button"
            disabled={!canRun || isRunning}
            onClick={runAgents}
            type="button"
          >
            <Play size={17} aria-hidden="true" />
            {isRunning ? "Running agents" : "Run agents"}
          </button>
        </header>

        <section className="panel" id="scenario">
          <div className="panel-header">
            <h3 className="font-bold">Disaster scenario</h3>
            <span className={`status-pill ${canRun ? "" : "warning"}`}>
              {canRun ? "Ready to run" : "Select disaster and region"}
            </span>
          </div>
          <div className="panel-body grid gap-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="field">
                <label htmlFor="disasterType">Disaster type</label>
                <select
                  id="disasterType"
                  value={scenario.disasterType}
                  onChange={(event) => updateScenario("disasterType", event.target.value)}
                >
                  <option value="">Select disaster</option>
                  <option>Cyclone</option>
                  <option>Flood</option>
                  <option>Earthquake</option>
                  <option>Wildfire</option>
                  <option>Hospital outage</option>
                  <option>Power grid failure</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="severity">Severity</label>
                <select
                  id="severity"
                  value={scenario.severity}
                  onChange={(event) => updateScenario("severity", event.target.value as Severity)}
                >
                  <option>Low</option>
                  <option>Moderate</option>
                  <option>High</option>
                  <option>Critical</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="field">
                <label htmlFor="region">Region</label>
                <input
                  id="region"
                  placeholder="City, district, or operating zone"
                  value={scenario.region}
                  onChange={(event) => updateScenario("region", event.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="timeHorizon">Planning horizon</label>
                <input
                  id="timeHorizon"
                  value={scenario.timeHorizon}
                  onChange={(event) => updateScenario("timeHorizon", event.target.value)}
                />
              </div>
            </div>

            <div className="field">
              <label htmlFor="notes">Operational notes</label>
              <textarea
                id="notes"
                placeholder="Road closures, hospitals at risk, shelter limits, evacuation concerns..."
                value={scenario.notes}
                onChange={(event) => updateScenario("notes", event.target.value)}
              />
            </div>

            {hasResults && (
              <div className="assumption-grid">
                <div>
                  <span>Population</span>
                  <strong>{numberFormat(asNumber(sourceScenario.population, affectedPopulation))}</strong>
                </div>
                <div>
                  <span>Hospital beds</span>
                  <strong>{numberFormat(asNumber(sourceScenario.hospital_capacity))}</strong>
                </div>
                <div>
                  <span>Shelter capacity</span>
                  <strong>{numberFormat(asNumber(sourceScenario.shelter_capacity))}</strong>
                </div>
                <div>
                  <span>Weather</span>
                  <strong>
                    {asNumber(sourceScenario.wind_speed)} km/h, {asNumber(sourceScenario.rainfall)} mm
                  </strong>
                </div>
                <div>
                  <span>Elevation</span>
                  <strong>{asNumber(sourceScenario.elevation)} m</strong>
                </div>
                <div>
                  <span>Density</span>
                  <strong>{numberFormat(asNumber(sourceScenario.population_density))}/km2</strong>
                </div>
                <div>
                  <span>Temp / humidity</span>
                  <strong>
                    {asNumber(sourceScenario.temperature)} C / {asNumber(sourceScenario.humidity)}%
                  </strong>
                </div>
                <div>
                  <span>Sources</span>
                  <strong>{sourceRows.length} checked</strong>
                </div>
              </div>
            )}
          </div>
        </section>

        {!hasResults && !isRunning && (
          <section className="empty-state">
            <ShieldCheck size={34} aria-hidden="true" />
            <h3>Dashboard waits for the agents</h3>
            <p>
              Enter disaster type, severity, and region. The system will fetch or estimate
              population, capacity, weather, elevation, density, and facilities from public
              global data sources before the agents generate charts and reports.
            </p>
          </section>
        )}

        {(hasResults || isRunning) && (
          <>
            <section className="metric-grid" id="dashboard" aria-label="Generated metrics">
              <div className="metric">
                <p className="text-sm font-bold text-[var(--muted)]">Risk score</p>
                <p className="mt-4 text-3xl font-bold">
                  {hasResults ? Math.round(riskScore * 100) : "--"}
                </p>
                <p className="mt-2 text-sm text-[var(--muted)]">
                  {hasResults ? String(riskOutput.risk_level) : "Waiting for Risk Agent"}
                </p>
              </div>
              <div className="metric">
                <p className="text-sm font-bold text-[var(--muted)]">Affected population</p>
                <p className="mt-4 text-3xl font-bold">
                  {hasResults ? numberFormat(affectedPopulation) : "--"}
                </p>
                <p className="mt-2 text-sm text-[var(--muted)]">Risk Agent output</p>
              </div>
              <div className="metric">
                <p className="text-sm font-bold text-[var(--muted)]">Evacuation need</p>
                <p className="mt-4 text-3xl font-bold">
                  {hasResults ? numberFormat(evacuationNeed) : "--"}
                </p>
                <p className="mt-2 text-sm text-[var(--muted)]">Prediction Agent output</p>
              </div>
              <div className="metric">
                <p className="text-sm font-bold text-[var(--muted)]">Validation</p>
                <p className="mt-4 text-2xl font-bold">
                  {orchestration?.validation_status ?? "Running"}
                </p>
                <p className="mt-2 text-sm text-[var(--muted)]">
                  Confidence {orchestration ? `${Math.round(orchestration.confidence * 100)}%` : "--"}
                </p>
              </div>
            </section>

            <section className="mt-4 panel" id="agents">
              <div className="panel-header">
                <h3 className="font-bold">Agent pipeline</h3>
                <span className="text-sm text-[var(--muted)]">
                  {completedAgents}/{agents.length} complete
                </span>
              </div>
              <div className="panel-body">
                <div className="agent-chain">
                  {agents.map((agent, index) => {
                    const Icon = agent.icon;
                    const state = getAgentState(index, activeAgent, completedAgents);
                    const result = getAgent(orchestration, agent.name);
                    return (
                      <div className="agent-step" data-state={state} key={agent.name}>
                        <div className="flex items-center justify-between gap-2">
                          <div className="agent-icon">
                            <Icon size={15} aria-hidden="true" />
                          </div>
                          <span className="text-xs font-bold uppercase text-[var(--muted)]">
                            {result?.status ?? state}
                          </span>
                        </div>
                        <p className="mt-3 truncate text-sm font-bold">{agent.name}</p>
                        <p className="mt-1 text-xs text-[var(--muted)]">
                          {result ? `${Math.round(result.confidence * 100)}%` : "waiting"}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>

            <section className="data-grid" id="risk">
              <div className="panel">
                <div className="panel-header">
                  <h3 className="font-bold">Risk forecast</h3>
                </div>
                <div className="panel-body h-[260px]">
                  {chartsReady && hasResults && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={riskTrend}>
                        <CartesianGrid stroke="#dfe5df" strokeDasharray="3 3" />
                        <XAxis dataKey="hour" stroke="#65736b" />
                        <YAxis stroke="#65736b" />
                        <Tooltip />
                        <Line dataKey="risk" stroke="#b42318" strokeWidth={3} type="monotone" />
                        <Line dataKey="impact" stroke="#245ba7" strokeWidth={3} type="monotone" />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <h3 className="font-bold">Risk map</h3>
                  <span className="status-pill warning">
                    <MapPinned size={14} aria-hidden="true" />
                    Generated
                  </span>
                </div>
                <div className="panel-body">
                  <div className="map-canvas" aria-label="Generated operational risk map">
                    <div className="map-zone zone-critical" />
                    <div className="map-zone zone-warning" />
                    <div className="road road-a" />
                    <div className="road road-b" />
                    <div className="road road-c" />
                    <div className="map-pin pin-shelter">
                      <Warehouse size={15} aria-hidden="true" />
                      Shelter
                    </div>
                    <div className="map-pin pin-hospital">
                      <Hospital size={15} aria-hidden="true" />
                      Hospital
                    </div>
                    <div className="map-pin pin-staging">
                      <Ambulance size={15} aria-hidden="true" />
                      Staging
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className="data-grid" id="resources">
              <div className="panel">
                <div className="panel-header">
                  <h3 className="font-bold">Resource plan</h3>
                </div>
                <div className="panel-body h-[280px]">
                  {chartsReady && hasResults && (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={resourceChart}>
                        <CartesianGrid stroke="#dfe5df" strokeDasharray="3 3" />
                        <XAxis dataKey="name" stroke="#65736b" />
                        <YAxis stroke="#65736b" />
                        <Tooltip />
                        <Bar dataKey="required" fill="#b46c0a" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="available" fill="#166b4f" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              <div className="panel" id="simulation">
                <div className="panel-header">
                  <h3 className="font-bold">Simulation center</h3>
                </div>
                <div className="panel-body">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Scenario</th>
                        <th>Status</th>
                        <th>Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {simulationRows.map((row) => (
                        <tr key={String(row.scenario)}>
                          <td>{String(row.scenario).replaceAll("_", " ")}</td>
                          <td>{String(row.status)}</td>
                          <td>{String(row.delta)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            <section className="data-grid" id="reports">
              <div className="panel">
                <div className="panel-header">
                  <h3 className="font-bold">AI report</h3>
                  <span className={`status-pill ${serviceState === "offline" ? "warning" : ""}`}>
                    <FileText size={14} aria-hidden="true" />
                    {serviceState === "live" ? "NVIDIA" : "Local"}
                  </span>
                </div>
                <div className="panel-body">
                  <div className="report-output">{report}</div>
                </div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <h3 className="font-bold">Ask the agents</h3>
                  <span className="text-sm text-[var(--muted)]">
                    Casualties {hasResults ? numberFormat(casualtyEstimate) : "--"}
                  </span>
                </div>
                <div className="panel-body grid gap-3">
                  <div className="field">
                    <label htmlFor="agentQuestion">Real-time question</label>
                    <textarea
                      id="agentQuestion"
                      placeholder="Ask about routes, resources, validation warnings, casualty estimate, or what to do next..."
                      value={question}
                      onChange={(event) => setQuestion(event.target.value)}
                    />
                  </div>
                  <button
                    className="secondary-button"
                    disabled={!canRun || !question.trim() || isAnswering}
                    onClick={askRealtimeAgents}
                    type="button"
                  >
                    {isAnswering ? "Asking agents" : "Ask agents"}
                  </button>
                  {agentAnswer && <div className="answer-output">{agentAnswer}</div>}

                  <h3 className="mt-2 font-bold">Structured outputs</h3>
                  {(orchestration?.agent_results ?? []).map((result) => (
                    <details
                      className="rounded-lg border border-[var(--line)] bg-white p-3"
                      key={result.agent_name}
                    >
                      <summary className="cursor-pointer font-bold capitalize">
                        {result.agent_name} - {result.status}
                      </summary>
                      <p className="mt-2 text-sm text-[var(--muted)]">
                        {result.reasoning_summary}
                      </p>
                      <pre className="mt-3 max-h-72 overflow-auto rounded-lg bg-[var(--panel-muted)] p-3 text-xs">
                        {JSON.stringify(result.output, null, 2)}
                      </pre>
                    </details>
                  ))}
                </div>
              </div>
            </section>

            <section className="mt-4 panel">
              <div className="panel-header">
                <h3 className="font-bold">Public data sources</h3>
                <span className="text-sm text-[var(--muted)]">
                  {String(groundedContext.resolved_location ?? scenario.region)}
                </span>
              </div>
              <div className="panel-body">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Source</th>
                      <th>Status</th>
                      <th>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sourceRows.map((source) => (
                      <tr key={String(source.name)}>
                        <td>{String(source.name)}</td>
                        <td>{String(source.status)}</td>
                        <td>{String(source.detail)}</td>
                      </tr>
                    ))}
                    {weatherEvidence && (
                      <tr>
                        <td>{String(weatherEvidence.source)}</td>
                        <td>used</td>
                        <td>{String(weatherEvidence.detail)}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </section>
    </main>
  );
}
