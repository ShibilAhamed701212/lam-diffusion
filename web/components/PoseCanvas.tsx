"use client";

import { useRef, useState } from "react";
import { ASPECT_SIZE, type Aspect, type BBox, type Character, png } from "@/lib/lam";

const BOX_COLORS = ["#f472b6", "#60a5fa", "#facc15"];

type Drag = { index: number; mode: "move" | "resize"; startX: number; startY: number; start: BBox };

/** Layout + pose preview. Each character box can be dragged or resized; the skeleton
 *  image refreshes from the server once the pointer is released. */
export function PoseCanvas({
  aspect,
  characters,
  poseImage,
  loading,
  onBoxChange,
}: {
  aspect: Aspect;
  characters: Character[];
  poseImage: string | null;
  loading: boolean;
  onBoxChange: (index: number, bbox: BBox) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [live, setLive] = useState<BBox | null>(null);
  const [w, h] = ASPECT_SIZE[aspect];

  function begin(e: React.PointerEvent, index: number, mode: Drag["mode"]) {
    const bbox = characters[index].bbox;
    if (!bbox) return;
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setDrag({ index, mode, startX: e.clientX, startY: e.clientY, start: bbox });
    setLive(bbox);
  }

  function move(e: React.PointerEvent) {
    if (!drag || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const dx = (e.clientX - drag.startX) / rect.width;
    const dy = (e.clientY - drag.startY) / rect.height;
    const [x0, y0, x1, y1] = drag.start;
    const clamp = (v: number) => Math.min(1, Math.max(0, v));
    if (drag.mode === "move") {
      const bw = x1 - x0, bh = y1 - y0;
      const nx = Math.min(1 - bw, Math.max(0, x0 + dx));
      const ny = Math.min(1 - bh, Math.max(0, y0 + dy));
      setLive([nx, ny, nx + bw, ny + bh]);
    } else {
      setLive([x0, y0, Math.max(x0 + 0.08, clamp(x1 + dx)), Math.max(y0 + 0.08, clamp(y1 + dy))]);
    }
  }

  function end() {
    if (drag && live) onBoxChange(drag.index, live.map((v) => Math.round(v * 1000) / 1000) as BBox);
    setDrag(null);
    setLive(null);
  }

  return (
    <div
      ref={ref}
      className="relative mx-auto w-full overflow-hidden rounded-xl border border-white/10 bg-black select-none touch-none"
      style={{ aspectRatio: `${w} / ${h}`, maxHeight: "60vh", maxWidth: `calc(60vh * ${w / h})` }}
      onPointerMove={move}
      onPointerUp={end}
      onPointerCancel={end}
    >
      {poseImage && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={png(poseImage)} alt="OpenPose layout" className="absolute inset-0 h-full w-full" draggable={false} />
      )}
      {characters.map((c, i) => {
        const b = drag?.index === i && live ? live : c.bbox;
        if (!b) return null;
        const color = BOX_COLORS[i % BOX_COLORS.length];
        return (
          <div
            key={i}
            className="absolute cursor-move rounded-md border-2 border-dashed"
            style={{
              left: `${b[0] * 100}%`,
              top: `${b[1] * 100}%`,
              width: `${(b[2] - b[0]) * 100}%`,
              height: `${(b[3] - b[1]) * 100}%`,
              borderColor: color,
              background: drag?.index === i ? `${color}14` : "transparent",
            }}
            onPointerDown={(e) => begin(e, i, "move")}
          >
            <span
              className="absolute -top-px left-1 rounded-b px-1.5 py-0.5 text-[10px] font-semibold text-black"
              style={{ background: color }}
            >
              #{i + 1} {c.gender}
            </span>
            <span
              className="absolute -right-1.5 -bottom-1.5 h-3.5 w-3.5 cursor-se-resize rounded-sm border-2 border-black"
              style={{ background: color }}
              onPointerDown={(e) => begin(e, i, "resize")}
            />
          </div>
        );
      })}
      {loading && (
        <div className="absolute right-2 bottom-2 rounded-full bg-black/70 px-2 py-0.5 text-[11px] text-zinc-300">
          updating…
        </div>
      )}
    </div>
  );
}
