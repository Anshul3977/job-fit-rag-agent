import type { ApiErrorBody, FitResponse, ReadyResponse, SamplesResponse } from "./types";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  errorCode: string;

  constructor(status: number, errorCode: string, message: string) {
    super(message);
    this.status = status;
    this.errorCode = errorCode;
  }
}

async function parse<T>(response: Response): Promise<T> {
  const text = await response.text();
  let data: (T & ApiErrorBody) | null = null;
  if (text) {
    try {
      data = JSON.parse(text) as T & ApiErrorBody;
    } catch {
      data = null;
    }
  }
  if (!response.ok) {
    throw new ApiError(
      response.status,
      data?.error_code ?? "HTTP_ERROR",
      data?.message ?? `Request failed (${response.status}).`,
    );
  }
  return data as T;
}

export function getReady(): Promise<ReadyResponse> {
  return fetch(`${BASE}/ready`).then((response) => parse<ReadyResponse>(response));
}

export function getSamples(): Promise<SamplesResponse> {
  return fetch(`${BASE}/samples`).then((response) => parse<SamplesResponse>(response));
}

export function ingestResume(file: File): Promise<{ resume_id: string; filename: string; chunk_count: number }> {
  const body = new FormData();
  body.append("file", file);
  return fetch(`${BASE}/ingest/resume`, { method: "POST", body }).then((response) =>
    parse(response),
  );
}

export function ingestResumeText(
  text: string,
  filename: string,
): Promise<{ resume_id: string; filename: string; chunk_count: number }> {
  const body = new FormData();
  body.append("text", text);
  body.append("filename", filename);
  return fetch(`${BASE}/ingest/resume`, { method: "POST", body }).then((response) =>
    parse(response),
  );
}

export function ingestJob(text: string, title: string): Promise<{ job_id: string; title: string }> {
  return fetch(`${BASE}/ingest/job`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, title: title || null }),
  }).then((response) => parse(response));
}

export function fitBatch(resumeId: string, jobIds: string[]): Promise<{ results: FitResponse[] }> {
  return fetch(`${BASE}/fit/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_id: resumeId, job_ids: jobIds }),
  }).then((response) => parse(response));
}
