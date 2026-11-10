import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { after, before, test } from "node:test";
import { chromium } from "playwright";

const baseURL = "http://127.0.0.1:3187";
let server;
let browser;

before(async () => {
  server = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "-p", "3187"], {
    cwd: process.cwd(), stdio: "ignore", windowsHide: true,
  });
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(baseURL);
      if (response.ok) break;
    } catch { /* wait for Next.js startup */ }
    await delay(500);
  }
  browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH }
      : {}),
  });
});

after(async () => {
  await browser?.close();
  if (server && !server.killed) server.kill();
});

test("renders only SSE-backed agents and displays the verified briefing", async () => {
  const page = await browser.newPage();
  const briefing = {
    status: "partial", label: "hypothetical_scenario", run_id: "ui-run-1",
    executive_summary: "A test briefing from the stream.", action_plan: ["Confirm local conditions."],
    risk: { level: "moderate", premise_valid: true, summary: "Evidence is limited." },
    evacuation_recommendation: { recommendation: "standby_advisory", requires_human_approval: true },
    logistics_summary: null, citations: [{ record_id: "DEV-1", event_name: "Sample event", country: "Test", start_year: 2001 }],
    data_gaps: ["Current road status"], disclaimers: ["Historical data is not a forecast."],
    human_review: { required: true, reasons: ["Verify current conditions."] }, conflicts: [], abstentions: [],
  };
  const frames = ["run_started", "agent_started", "agent_verified", "briefing_ready"].map((name, i) => {
    const event = { event: name, run_id: "ui-run-1", agent: i < 3 ? ["coordinator", "risk", "logistics"][i] : null,
      data: name === "briefing_ready" ? { briefing } : {} };
    return `event: ${name}\ndata: ${JSON.stringify(event)}\n\n`;
  }).join("");
  await page.route("**/api/orchestrate/stream", (route) => route.fulfill({
    status: 200, contentType: "text/event-stream", body: frames,
  }));
  const pageResponse = await page.goto(baseURL);
  const headers = await pageResponse.allHeaders();
  assert.equal(headers["x-frame-options"], "DENY");
  assert.match(headers["content-security-policy"], /connect-src 'self'/);
  assert.doesNotMatch(headers["content-security-policy"], /unsafe-eval/);
  await page.getByLabel("Disaster type").selectOption("Flood");
  await page.getByLabel("Region").fill("Test district");
  const notes = page.getByLabel(/Operational notes/);
  await notes.fill("x".repeat(700));
  assert.equal(await notes.inputValue().then((value) => value.length), 500);
  await page.getByRole("button", { name: "Generate briefing" }).click();
  await page.getByText("A test briefing from the stream.").waitFor();
  assert.equal(await page.locator('section[aria-labelledby="agents-heading"] ol li').count(), 4);
  await page.getByText("Human approval required before any operational action.").waitFor();
  await page.getByText("Sample event").waitFor();
  await page.close();
});

test("shows a backend-down error and does not invent a briefing", async () => {
  const page = await browser.newPage();
  await page.route("**/api/orchestrate/stream", (route) => route.fulfill({
    status: 502, contentType: "application/json", body: JSON.stringify({ detail: "service unavailable" }),
  }));
  await page.goto(baseURL);
  await page.getByLabel("Disaster type").selectOption("Flood");
  await page.getByLabel("Region").fill("Test district");
  await page.getByRole("button", { name: "Generate briefing" }).click();
  await page.getByText("Service unavailable or run incomplete:").waitFor();
  assert.equal(await page.getByText("Verified briefing").count(), 0);
  await page.close();
});
