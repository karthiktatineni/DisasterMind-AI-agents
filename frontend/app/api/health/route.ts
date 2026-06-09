import { NextResponse } from "next/server";

const BACKEND_API_BASE_URL =
  process.env.BACKEND_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_API_BASE_URL}/health`, {
      cache: "no-store",
    });

    const text = await response.text();
    if (!response.ok) {
      return NextResponse.json(
        { detail: "Backend health check failed." },
        { status: response.status },
      );
    }

    return new NextResponse(text, {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach backend service." },
      { status: 502 },
    );
  }
}
