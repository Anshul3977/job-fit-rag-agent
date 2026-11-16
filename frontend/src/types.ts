export type Verdict = "strong" | "maybe" | "skip";
export type MatchStatus = "match" | "partial" | "gap";

export interface EvidenceQuote {
  chunk_id: string;
  quote: string;
  section: string;
  score: number;
  page: number | null;
}

export interface RequirementResult {
  requirement: string;
  kind: "required" | "preferred";
  status: MatchStatus;
  severity: "high" | "medium" | "low";
  rationale: string;
  evidence: EvidenceQuote | null;
}

export interface FitReport {
  overall_score: number;
  verdict: Verdict;
  summary: string;
  matching_skills: { skill: string; evidence_chunk_id: string; quote: string }[];
  gaps: { requirement: string; why_it_is_a_gap: string; severity: string }[];
  stretch_flags: string[];
  recommended_resume_edits: string[];
  citations: { chunk_id: string; quote: string; section: string | null }[];
  requirement_results: RequirementResult[];
  model: string;
  provider: "groq" | "deterministic";
  provider_note?: string;
  guardrail_flags?: string[];
}

export interface FitResponse {
  run_id: string;
  resume_id: string;
  job_id: string;
  job_title: string;
  latency_ms: number;
  retrieve_ms: number;
  generate_ms: number;
  model: string;
  provider: string;
  provider_note?: string;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  report: FitReport;
}

export interface ReadyResponse {
  status: string;
  llm_provider: string;
  llm_model: string;
  embedding_backend: string;
  embedding_model: string;
  vector_store: string;
}

export interface SamplesResponse {
  label: "synthetic";
  note: string;
  resume_filename: string;
  resume_text: string;
  jobs: { title: string; text: string }[];
}

export interface ApiErrorBody {
  error_code?: string;
  message?: string;
}
