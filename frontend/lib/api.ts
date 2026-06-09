export type GenerationMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type GenerationResponse = {
  provider: string;
  model: string;
  content: string;
  reasoning_content?: string | null;
  request_id?: string | null;
};

export type DisasterScenarioRequest = {
  disaster_type: string;
  region: string;
  time_horizon: string;
  severity: "Low" | "Moderate" | "High" | "Critical";
  notes: string;
};

export type AgentResult = {
  request_id: string;
  agent_name: string;
  timestamp: string;
  confidence: number;
  reasoning_summary: string;
  evidence: Array<Record<string, unknown>>;
  recommendations: string[];
  next_actions: string[];
  status: string;
  output: Record<string, unknown>;
};

export type RetrievedDisaster = {
  id?: string;
  event_name?: string;
  disaster_type?: string;
  disaster_subtype?: string;
  country?: string;
  region?: string;
  location?: string;
  start_year?: number;
  total_deaths?: number;
  total_affected?: number;
  total_damage?: number;
  similarity?: number;
};

export type EvidenceItem = {
  id?: string;
  event_name?: string;
  disaster_type?: string;
  disaster_subtype?: string;
  country?: string;
  region?: string;
  location?: string;
  start_year?: number;
  year?: number;
  total_deaths?: number;
  total_affected?: number;
  total_damage?: number;
  similarity?: number;
};

export type OrchestrationResponse = {
  request_id: string;
  status: "completed" | "completed_with_warnings" | "failed";
  agent_sequence: string[];
  agent_results: AgentResult[];
  final_report: string;
  validation_status: string;
  confidence: number;
  provider: string;
  model?: string | null;
  warnings: string[];
  retrieved_disasters?: RetrievedDisaster[];
  evidence_used?: EvidenceItem[];
  insufficient_data?: { status: string; reason?: string };
};

export type AgentQuestionResponse = {
  request_id: string;
  answer: string;
  provider: string;
  model?: string | null;
  cited_agents: string[];
  confidence: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "";

export async function generatePlan(
  messages: GenerationMessage[],
): Promise<GenerationResponse> {
  const endpoint = API_BASE_URL ? `${API_BASE_URL}/generate` : "/api/generate";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messages,
      max_tokens: 900,
      temperature: 0.2,
      top_p: 0.95,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Generation failed with status ${response.status}`);
  }

  return response.json();
}

export async function orchestrateScenario(
  scenario: DisasterScenarioRequest,
): Promise<OrchestrationResponse> {
  const endpoint = API_BASE_URL ? `${API_BASE_URL}/orchestrate` : "/api/orchestrate";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(scenario),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Orchestration failed with status ${response.status}`);
  }

  return response.json();
}

export async function askAgents(
  scenario: DisasterScenarioRequest,
  question: string,
): Promise<AgentQuestionResponse> {
  const endpoint = API_BASE_URL ? `${API_BASE_URL}/agent-answer` : "/api/agent-answer";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ intake: scenario, question }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Agent answer failed with status ${response.status}`);
  }

  return response.json();
}
