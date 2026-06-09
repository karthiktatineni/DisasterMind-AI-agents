import { NextResponse } from "next/server";

const BACKEND_API_BASE_URL =
  process.env.BACKEND_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (process.env.DISASTERMIND_API_KEY) {
      headers["X-DisasterMind-API-Key"] = process.env.DISASTERMIND_API_KEY;
    }

    const response = await fetch(`${BACKEND_API_BASE_URL}/orchestrate-intake`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const text = await response.text();
    if (!response.ok) {
      return NextResponse.json(
        { detail: "Backend orchestration failed." },
        { status: response.status },
      );
    }

    return new NextResponse(text, {
      status: response.status,
      headers: {
        "Content-Type": "application/json",
      },
    });
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach orchestration service." },
      { status: 502 },
    );
  }
}
