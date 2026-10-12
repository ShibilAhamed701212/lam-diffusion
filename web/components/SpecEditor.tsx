"use client";

import { AGES, type Age, type Aspect, type Character, type Gender, type Position, type Spec, MAX_CHARACTERS, blankCharacter, label } from "@/lib/lam";

const BOX_COLORS = ["#f472b6", "#60a5fa", "#facc15"];

const input =
  "w-full rounded-md border border-white/10 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-400/60 focus:outline-none";

function Field({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium tracking-wide text-zinc-500 uppercase">{name}</span>
      {children}
    </label>
  );
}

function Select<T extends string>({ value, options, onChange }: { value: T; options: readonly T[]; onChange: (v: T) => void }) {
  return (
    <select className={input} value={value} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => (
        <option key={o} value={o}>
          {label(o)}
        </option>
      ))}
    </select>
  );
}

/** Edits the structured spec. Layout-affecting edits clear bboxes so the server re-lays them out. */
export function SpecEditor({
  spec,
  poses,
  framings,
  onChange,
}: {
  spec: Spec;
  poses: string[];
  framings: string[];
  onChange: (spec: Spec) => void;
}) {
  const clearBoxes = (chars: Character[]) => chars.map((c) => ({ ...c, bbox: null }));

  function setChar(i: number, patch: Partial<Character>) {
    const relayout = "position" in patch || "framing" in patch;
    const characters = spec.characters.map((c, j) => (j === i ? { ...c, ...patch, ...(relayout ? { bbox: null } : {}) } : c));
    onChange({ ...spec, characters });
  }

  function addChar() {
    const used = new Set(spec.characters.map((c) => c.position));
    const pos = (["left", "right", "center"] as Position[]).find((p) => !used.has(p)) ?? "center";
    onChange({ ...spec, characters: clearBoxes([...spec.characters, blankCharacter(pos)]) });
  }

  function removeChar(i: number) {
    onChange({ ...spec, characters: clearBoxes(spec.characters.filter((_, j) => j !== i)) });
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <Field name="Scene">
          <input className={input} value={spec.scene} placeholder="cherry blossoms, park" onChange={(e) => onChange({ ...spec, scene: e.target.value })} />
        </Field>
        <Field name="Style">
          <input className={input} value={spec.style} placeholder="soft lighting" onChange={(e) => onChange({ ...spec, style: e.target.value })} />
        </Field>
        <Field name="Props">
          <input
            className={input}
            value={spec.props.join(", ")}
            placeholder="book, cat"
            onChange={(e) => onChange({ ...spec, props: e.target.value.split(",").map((s) => s.trimStart()) })}
          />
        </Field>
        <Field name="Aspect">
          <Select<Aspect>
            value={spec.aspect}
            options={["portrait", "square", "landscape"]}
            onChange={(aspect) => onChange({ ...spec, aspect, characters: clearBoxes(spec.characters) })}
          />
        </Field>
      </div>

      {spec.characters.map((c, i) => (
        <div key={i} className="rounded-lg border border-white/10 bg-zinc-900/50 p-3">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: BOX_COLORS[i % 3] }} />
              Character {i + 1}
            </div>
            {spec.characters.length > 1 && (
              <button className="text-xs text-zinc-500 hover:text-red-400" onClick={() => removeChar(i)}>
                Remove
              </button>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
            <Field name="Gender">
              <Select<Gender> value={c.gender} options={["female", "male", "other"]} onChange={(gender) => setChar(i, { gender })} />
            </Field>
            <Field name="Age">
              <Select<Age> value={c.age} options={AGES} onChange={(age) => setChar(i, { age })} />
            </Field>
            <Field name="Pose">
              <Select value={c.pose} options={poses} onChange={(pose) => setChar(i, { pose })} />
            </Field>
            <Field name="Framing">
              <Select value={c.framing} options={framings} onChange={(framing) => setChar(i, { framing })} />
            </Field>
            <Field name="Hair color">
              <input className={input} value={c.hair_color} placeholder="silver" onChange={(e) => setChar(i, { hair_color: e.target.value })} />
            </Field>
            <Field name="Hair style">
              <input className={input} value={c.hair_style} placeholder="long hair" onChange={(e) => setChar(i, { hair_style: e.target.value })} />
            </Field>
            <Field name="Eyes">
              <input className={input} value={c.eye_color} placeholder="blue" onChange={(e) => setChar(i, { eye_color: e.target.value })} />
            </Field>
            <Field name="Outfit">
              <input className={input} value={c.outfit} placeholder="school uniform" onChange={(e) => setChar(i, { outfit: e.target.value })} />
            </Field>
            <Field name="Expression">
              <input className={input} value={c.expression} placeholder="smile" onChange={(e) => setChar(i, { expression: e.target.value })} />
            </Field>
            <Field name="Action">
              <input className={input} value={c.action} placeholder="reading book" onChange={(e) => setChar(i, { action: e.target.value })} />
            </Field>
            <Field name="Position">
              <Select<Position> value={c.position} options={["left", "center", "right"]} onChange={(position) => setChar(i, { position })} />
            </Field>
            <Field name="Extra tags">
              <input className={input} value={c.extra} placeholder="hair ribbon" onChange={(e) => setChar(i, { extra: e.target.value })} />
            </Field>
            <label className="flex items-end gap-2 pb-2 text-sm text-zinc-400">
              <input type="checkbox" className="accent-violet-500" checked={c.mirror} onChange={(e) => setChar(i, { mirror: e.target.checked })} />
              Mirror pose
            </label>
          </div>
        </div>
      ))}

      {spec.characters.length < MAX_CHARACTERS && (
        <button
          className="w-full rounded-lg border border-dashed border-white/15 py-2 text-sm text-zinc-400 hover:border-violet-400/50 hover:text-zinc-200"
          onClick={addChar}
        >
          + Add character
        </button>
      )}
    </div>
  );
}
