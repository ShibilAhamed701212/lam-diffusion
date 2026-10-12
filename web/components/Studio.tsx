"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { PoseCanvas } from "./PoseCanvas";
import { SpecEditor } from "./SpecEditor";
import { type BBox, type GenerateResult, type Health, type PlanResult, type Spec, api, png } from "@/lib/lam";

const EXAMPLES = [
  "an anime girl with long pink hair and green eyes, school uniform, standing in a cherry blossom park",
  "a knight girl with long blonde hair and blue eyes in silver armor holding a sword, castle at sunset",
  "an anime boy with short black hair and red eyes, wearing a dark jacket, standing on a rooftop at sunset",
  "an anime girl with short blue hair and yellow eyes, summer dress, standing on a beach",
];

type HistoryItem = GenerateResult & { userPrompt: string };

export function Studio() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [planner, setPlanner] = useState<"llm" | "rules">("llm");
  const [spec, setSpec] = useState<Spec | null>(null);
  const [tagPrompt, setTagPrompt] = useState("");
  const [poseImage, setPoseImage] = useState<string | null>(null);
  const [plannerUsed, setPlannerUsed] = useState<string | null>(null);
  const [busy, setBusy] = useState<"plan" | "render" | "generate" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seed, setSeed] = useState(-1);
  const [steps, setSteps] = useState(28);
  const [cfg, setCfg] = useState(5);
  const [controlScale, setControlScale] = useState(0.8);
  const [withBaseline, setWithBaseline] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selected, setSelected] = useState(0);
  const renderSeq = useRef(0);

  // Poll the GPU server while it loads models.
  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const h = await api<Health>("health");
        if (stop) return;
        setHealth(h);
        setHealthErr(null);
        timer = setTimeout(tick, h.status === "ready" ? 30_000 : 5_000);
      } catch (e) {
        if (stop) return;
        setHealthErr((e as Error).message);
        timer = setTimeout(tick, 8_000);
      }
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, []);

  // Re-render prompt + pose image after spec edits. Only bboxes are taken back from the
  // server so in-progress typing in the editor is never overwritten.
  const rerender = useCallback(async (next: Spec) => {
    const seq = ++renderSeq.current;
    setBusy((b) => b ?? "render");
    try {
      const r = await api<PlanResult>("render", { spec: next });
      if (seq !== renderSeq.current) return;
      setTagPrompt(r.prompt);
      setPoseImage(r.pose_image);
      setSpec((cur) =>
        cur && cur.characters.length === r.spec.characters.length
          ? { ...cur, characters: cur.characters.map((c, i) => ({ ...c, bbox: r.spec.characters[i].bbox })) }
          : cur,
      );
      setError(null);
    } catch (e) {
      if (seq === renderSeq.current) setError((e as Error).message);
    } finally {
      if (seq === renderSeq.current) setBusy((b) => (b === "render" ? null : b));
    }
  }, []);

  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  function editSpec(next: Spec) {
    setSpec(next);
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => rerender(next), 350);
  }

  function moveBox(index: number, bbox: BBox) {
    if (!spec) return;
    const next = { ...spec, characters: spec.characters.map((c, i) => (i === index ? { ...c, bbox } : c)) };
    setSpec(next);
    rerender(next);
  }

  async function runPlan() {
    setBusy("plan");
    setError(null);
    try {
      const r = await api<PlanResult>("plan", { prompt, planner });
      renderSeq.current++;
      setSpec(r.spec);
      setTagPrompt(r.prompt);
      setPoseImage(r.pose_image);
      setPlannerUsed(`${r.planner_used} · ${r.seconds}s`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function runGenerate() {
    if (!spec) return;
    setBusy("generate");
    setError(null);
    try {
      const r = await api<GenerateResult>("generate", {
        spec,
        user_prompt: prompt,
        seed,
        steps,
        cfg,
        control_scale: controlScale,
        with_baseline: withBaseline,
      });
      setHistory((h) => [{ ...r, userPrompt: prompt }, ...h].slice(0, 24));
      setSelected(0);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const ready = health?.status === "ready";
  const current = history[selected];

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-1 flex-col gap-4 px-4 py-4 lg:px-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-100">
            LAM Studio <span className="font-normal text-zinc-500">· plan → pose → diffuse</span>
          </h1>
          <p className="text-xs text-zinc-500">Action-planned anime character generation (LAM-Diffusion demo)</p>
        </div>
        <StatusPill health={health} error={healthErr} />
      </header>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">{error}</div>
      )}

      <div className="grid flex-1 gap-4 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.8fr)_minmax(0,1.15fr)]">
        {/* 1. Prompt + spec */}
        <section className="flex flex-col gap-3 rounded-xl border border-white/10 bg-zinc-950/60 p-4">
          <StepTitle n={1} title="Describe" hint="The planner turns this into an editable spec" />
          <textarea
            className="min-h-24 w-full resize-y rounded-lg border border-white/10 bg-zinc-900 p-3 text-sm text-zinc-100 focus:border-violet-400/60 focus:outline-none"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                className="max-w-full truncate rounded-full border border-white/10 px-2.5 py-1 text-[11px] text-zinc-400 hover:border-violet-400/50 hover:text-zinc-200"
                onClick={() => setPrompt(ex)}
                title={ex}
              >
                {ex.slice(0, 38)}…
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg border border-white/10 p-0.5 text-xs">
              {(["llm", "rules"] as const).map((p) => (
                <button
                  key={p}
                  className={`rounded-md px-2.5 py-1 ${planner === p ? "bg-zinc-800 text-zinc-100" : "text-zinc-500"}`}
                  onClick={() => setPlanner(p)}
                >
                  {p === "llm" ? "LLM planner" : "Rule planner"}
                </button>
              ))}
            </div>
            <button
              className="ml-auto rounded-lg bg-violet-500 px-4 py-1.5 text-sm font-medium text-white hover:bg-violet-400 disabled:opacity-40"
              disabled={!ready || busy !== null || !prompt.trim()}
              onClick={runPlan}
            >
              {busy === "plan" ? "Planning…" : "Plan"}
            </button>
          </div>
          {plannerUsed && <p className="text-[11px] text-zinc-500">planner: {plannerUsed}</p>}
          {spec ? (
            <div className="-mx-1 overflow-y-auto px-1 lg:max-h-[calc(100vh-420px)]">
              <SpecEditor spec={spec} poses={health?.poses ?? []} framings={health?.framings ?? []} onChange={editSpec} />
            </div>
          ) : (
            <Empty text="Press Plan to get a structured character spec you can edit." />
          )}
        </section>

        {/* 2. Layout */}
        <section className="flex flex-col gap-3 rounded-xl border border-white/10 bg-zinc-950/60 p-4">
          <StepTitle n={2} title="Layout & pose" hint="Drag boxes to move, corner to resize" />
          {spec ? (
            <>
              <PoseCanvas
                aspect={spec.aspect}
                characters={spec.characters}
                poseImage={poseImage}
                loading={busy === "render"}
                onBoxChange={moveBox}
              />
              <button
                className="self-start text-xs text-zinc-500 hover:text-zinc-300"
                onClick={() => editSpec({ ...spec, characters: spec.characters.map((c) => ({ ...c, bbox: null })) })}
              >
                Reset layout
              </button>
              <div>
                <div className="mb-1 text-[11px] font-medium tracking-wide text-zinc-500 uppercase">Conditioning prompt</div>
                <p className="rounded-lg bg-zinc-900 p-2.5 font-mono text-[11px] leading-relaxed text-zinc-300">{tagPrompt}</p>
              </div>
            </>
          ) : (
            <Empty text="The planned pose skeletons and layout appear here." />
          )}
        </section>

        {/* 3. Generate */}
        <section className="flex flex-col gap-3 rounded-xl border border-white/10 bg-zinc-950/60 p-4">
          <StepTitle n={3} title="Generate" hint="LAM-Diffusion: trained full U-Net with dual-stream action conditioning" />
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs text-zinc-400 sm:grid-cols-3">
            <Num label="Seed (-1 random)" value={seed} step={1} onChange={setSeed} />
            <Num label="Steps" value={steps} step={1} min={8} max={50} onChange={setSteps} />
            <Num label="CFG" value={cfg} step={0.5} min={1} max={12} onChange={setCfg} />
          </div>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-xs text-zinc-400">
              <input type="checkbox" className="accent-violet-500" checked={withBaseline} onChange={(e) => setWithBaseline(e.target.checked)} />
              Also render the plain Animagine prompt-only baseline (same seed)
            </label>
            <button
              className="ml-auto rounded-lg bg-pink-500 px-4 py-1.5 text-sm font-medium text-white hover:bg-pink-400 disabled:opacity-40"
              disabled={!ready || !spec || busy !== null}
              onClick={runGenerate}
            >
              {busy === "generate" ? "Generating…" : "Generate"}
            </button>
          </div>

          {current ? (
            <>
              <div className={`grid gap-3 ${current.baseline ? "grid-cols-2" : "grid-cols-1"}`}>
                <Figure title="LAM-Diffusion" sub={`seed ${current.seed} · ${current.seconds}s`} src={png(current.image)} />
                {current.baseline && <Figure title="Prompt-only baseline" sub="same seed, no plan" src={png(current.baseline.image)} />}
              </div>
              <button className="self-start text-xs text-zinc-500 hover:text-zinc-300" onClick={() => setSeed(current.seed)}>
                Lock this seed
              </button>
            </>
          ) : (
            <Empty text={busy === "generate" ? "Generating… (~20–40s on an L4)" : "Results appear here."} />
          )}

          {history.length > 1 && (
            <div className="flex gap-2 overflow-x-auto pb-1">
              {history.map((h, i) => (
                <button
                  key={i}
                  onClick={() => setSelected(i)}
                  className={`h-16 shrink-0 overflow-hidden rounded-md border ${i === selected ? "border-violet-400" : "border-white/10"}`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={png(h.image)} alt="" className="h-full w-auto" />
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function StatusPill({ health, error }: { health: Health | null; error: string | null }) {
  const [color, text] = error
    ? ["bg-red-500", error.length > 70 ? `${error.slice(0, 70)}…` : error]
    : !health
      ? ["bg-zinc-500", "connecting…"]
      : health.status === "ready"
        ? ["bg-emerald-500", `GPU ready${health.llm_planner ? " · LLM planner" : " · rule planner only"}`]
        : health.status === "loading"
          ? ["bg-amber-500 animate-pulse", "GPU loading models…"]
          : ["bg-red-500", `GPU error: ${health.error ?? ""}`.slice(0, 90)];
  return (
    <div className="flex items-center gap-2 rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400" title={error ?? health?.error ?? ""}>
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {text}
    </div>
  );
}

function StepTitle({ n, title, hint }: { n: number; title: string; hint: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-800 text-[11px] font-semibold text-zinc-300">{n}</span>
      <h2 className="text-sm font-semibold text-zinc-100">{title}</h2>
      <span className="truncate text-xs text-zinc-500">{hint}</span>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed border-white/10 p-8 text-center text-sm text-zinc-500">
      {text}
    </div>
  );
}

function Figure({ title, sub, src }: { title: string; sub: string; src: string }) {
  return (
    <figure className="flex flex-col gap-1.5">
      <a href={src} download={`${title.replace(/\s+/g, "-").toLowerCase()}.png`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={title} className="w-full rounded-lg border border-white/10" />
      </a>
      <figcaption className="flex justify-between text-[11px]">
        <span className="font-medium text-zinc-300">{title}</span>
        <span className="text-zinc-500">{sub}</span>
      </figcaption>
    </figure>
  );
}

function Num({
  label,
  value,
  step,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  step: number;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] text-zinc-500">{label}</span>
      <input
        type="number"
        className="w-full rounded-md border border-white/10 bg-zinc-900 px-2 py-1 text-sm text-zinc-100 focus:border-violet-400/60 focus:outline-none"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
