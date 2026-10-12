// Proxies the browser to the GPU server so the tunnel URL and token never reach the client.
import type { NextRequest } from "next/server";

const ALLOWED = new Set(["health", "plan", "render", "generate"]);

async function forward(req: NextRequest, path: string[], method: "GET" | "POST") {
  const target = path.join("/");
  if (!ALLOWED.has(target)) return Response.json({ detail: "not found" }, { status: 404 });

  const base = process.env.LAM_API_URL;
  const token = process.env.LAM_TOKEN;
  if (!base || !token) {
    return Response.json(
      { detail: "GPU server not configured. Run scripts/colab_up.sh first." },
      { status: 503 },
    );
  }

  try {
    const res = await fetch(`${base}/${target}`, {
      method,
      headers: { "content-type": "application/json", "x-lam-token": token, "bypass-tunnel-reminder": "true" },
      body: method === "POST" ? await req.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(180_000),
    });
    const body = await res.text();
    const isJson = (res.headers.get("content-type") ?? "").includes("application/json");
    if (!isJson && !res.ok) {
      // Cloudflare answers 530/502 with an HTML page when the Colab VM behind the tunnel is gone.
      return Response.json(
        { detail: `GPU server offline (HTTP ${res.status}). The Colab session probably ended; run scripts/colab_up.sh.` },
        { status: 503 },
      );
    }
    return new Response(body, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return Response.json({ detail: `GPU server unreachable (${msg}). Is the Colab session running?` }, { status: 502 });
  }
}

export async function GET(req: NextRequest, ctx: RouteContext<"/api/lam/[...path]">) {
  return forward(req, (await ctx.params).path, "GET");
}

export async function POST(req: NextRequest, ctx: RouteContext<"/api/lam/[...path]">) {
  return forward(req, (await ctx.params).path, "POST");
}
