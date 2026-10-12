export type Gender = "female" | "male" | "other";
export type Age = "unspecified" | "child" | "teen" | "young_adult" | "adult" | "elderly";
export const AGES: readonly Age[] = ["unspecified", "child", "teen", "young_adult", "adult", "elderly"];
export type Position = "left" | "center" | "right";
export type Aspect = "portrait" | "square" | "landscape";
export type BBox = [number, number, number, number];

export interface Character {
  gender: Gender;
  age: Age;
  hair_color: string;
  hair_style: string;
  eye_color: string;
  outfit: string;
  expression: string;
  action: string;
  extra: string;
  pose: string;
  framing: string;
  position: Position;
  mirror: boolean;
  bbox: BBox | null;
}

export interface Spec {
  scene: string;
  style: string;
  props: string[];
  aspect: Aspect;
  characters: Character[];
}

export interface Rendered {
  spec: Spec;
  prompt: string;
  negative: string;
  pose_image: string;
}

export interface PlanResult extends Rendered {
  planner_used: string;
  seconds: number;
}

export interface GenerateResult extends Rendered {
  image: string;
  baseline: { image: string; prompt: string } | null;
  seed: number;
  seconds: number;
}

export interface Health {
  status: "loading" | "ready" | "error";
  error: string | null;
  llm_planner: boolean;
  poses: string[];
  framings: string[];
  aspects: Aspect[];
}

export const ASPECT_SIZE: Record<Aspect, [number, number]> = {
  portrait: [832, 1216],
  square: [1024, 1024],
  landscape: [1216, 832],
};

export const MAX_CHARACTERS = 3;

export function blankCharacter(position: Position = "center"): Character {
  return {
    gender: "female",
    age: "unspecified",
    hair_color: "",
    hair_style: "",
    eye_color: "",
    outfit: "",
    expression: "",
    action: "",
    extra: "",
    pose: "standing",
    framing: "full_body",
    position,
    mirror: false,
    bbox: null,
  };
}

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/lam/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? res.statusText);
    throw new Error(detail);
  }
  return data as T;
}

export const png = (b64: string) => `data:image/png;base64,${b64}`;

export const label = (s: string) => s.replace(/_/g, " ");
