import { NextResponse } from "next/server";

const BACKEND_API_BASE_URL =
  process.env.BACKEND_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export async function POST(request: Request) {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (process.env.DISASTERMIND_API_KEY) {
      headers["X-DisasterMind-API-Key"] = process.env.DISASTERMIND_API_KEY;
    }
    const response = await fetch(`${BACKEND_API_BASE_URL}/orchestrate/stream`, {
      method: "POST",
      headers,
      body: await request.text(),
      cache: "no-store",
      signal: request.signal,
    });
    if (!response.ok || !response.body) {
      return new NextResponse(await response.text(), {
        status: response.status,
        headers: { "Content-Type": response.headers.get("Content-Type") || "application/json" },
      });
    }
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return NextResponse.json({ detail: "Unable to reach orchestration service." }, { status: 502 });
  }
}
