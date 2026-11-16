import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  fitOne,
  getReady,
  getSamples,
  ingestJob,
  ingestResume,
  ingestResumeText,
} from "./api";
import type { FitResponse, ReadyResponse } from "./types";

interface JobDraft {
  key: string;
  title: string;
  text: string;
}

const EMPTY_JOB = (): JobDraft => ({
  key: Math.random().toString(36).slice(2, 8),
  title: "",
  text: "",
});

function summaryText(result: FitResponse): string {
  const report = result.report;
  const matches = report.matching_skills
    .map((item) => `- ${item.skill}\n  "${item.quote}" (${item.evidence_chunk_id})`)
    .join("\n");
  const gaps = report.gaps
    .map((item) => `- ${item.requirement} [${item.severity}]\n  ${item.why_it_is_a_gap}`)
    .join("\n");
  return [
    `Job Fit — ${result.job_title}`,
    `Score: ${report.overall_score}/100 (${report.verdict})`,
    `Scorer: ${result.provider} (${result.model})`,
    result.report.provider_note || "",
    "",
    report.summary,
    "",
    "Matches:",
    matches || "- none",
    "",
    "Gaps:",
    gaps || "- none",
    "",
    `Latency: ${result.latency_ms} ms (retrieve ${result.retrieve_ms} ms, graph ${result.generate_ms} ms)`,
    `Run: ${result.run_id}`,
  ].join("\n");
}

export function App() {
  const [ready, setReady] = useState<ReadyResponse | null>(null);
  const [readyError, setReadyError] = useState<string | null>(null);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumeText, setResumeText] = useState("");
  const [resumeName, setResumeName] = useState("pasted-resume.txt");
  const [jobs, setJobs] = useState<JobDraft[]>([EMPTY_JOB()]);
  const [sampleNote, setSampleNote] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<FitResponse[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [active, setActive] = useState(0);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    getReady()
      .then(setReady)
      .catch(() => setReadyError("API is not reachable yet."));
  }, []);

  const canScore = useMemo(() => {
    const hasResume = Boolean(resumeFile) || resumeText.trim().length >= 40;
    const hasJob = jobs.some((job) => job.text.trim().length >= 20);
    return hasResume && hasJob && status !== "loading";
  }, [resumeFile, resumeText, jobs, status]);

  async function loadSample() {
    setError(null);
    setSampleNote(null);
    try {
      const sample = await getSamples();
      setResumeFile(null);
      setResumeText(sample.resume_text);
      setResumeName(sample.resume_filename);
      setJobs(
        sample.jobs.map((job) => ({
          key: Math.random().toString(36).slice(2, 8),
          title: job.title,
          text: job.text,
        })),
      );
      setSampleNote(sample.note);
      setStatus("idle");
      setResults([]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the sample.");
    }
  }

  async function score() {
    setStatus("loading");
    setError(null);
    setCopied(false);
    try {
      const resume = resumeFile
        ? await ingestResume(resumeFile)
        : await ingestResumeText(resumeText, resumeName || "pasted-resume.txt");
      const filled = jobs.filter((job) => job.text.trim().length >= 20);
      const ingested = [];
      for (const job of filled) {
        ingested.push(await ingestJob(job.text, job.title));
      }
      const scored: FitResponse[] = [];
      for (let index = 0; index < ingested.length; index += 1) {
        if (ingested.length > 1) {
          setProgress(
            `Scoring job ${index + 1} of ${ingested.length}. Jobs run one at a time so a free Groq key is not asked for several fits at once.`,
          );
        }
        scored.push(await fitOne(resume.resume_id, ingested[index].job_id));
      }
      setProgress(null);
      setResults(scored);
      setActive(0);
      setStatus("done");
    } catch (err) {
      setStatus("error");
      setProgress(null);
      setResults([]);
      if (err instanceof ApiError) {
        setError(`${err.errorCode}: ${err.message}`);
      } else {
        setError("Scoring failed before the API returned a report.");
      }
    }
  }

  async function copySummary() {
    const current = results[active];
    if (!current) return;
    const text = summaryText(current);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      setCopied(false);
      setError("Clipboard is blocked in this browser. Select the summary and copy it manually.");
    }
  }

  const current = results[active];

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="eyebrow">Job Fit RAG Agent</p>
          <h1>Score the fit. Cite the line.</h1>
          <p className="lede">
            Upload a resume, paste one or more job posts, and read a score that only counts
            skills the retrieval step can point at in the resume.
          </p>
        </div>
        <div className="status-pill" data-state={ready ? "up" : "down"}>
          {ready ? (
            <>
              <strong>{ready.llm_provider === "groq" ? "Groq" : "Deterministic scorer"}</strong>
              <span>
                {ready.embedding_backend === "hash" ? "hash embeddings" : ready.embedding_model}
                {" · "}
                {ready.vector_store}
              </span>
            </>
          ) : (
            <span>{readyError ?? "Checking the API…"}</span>
          )}
        </div>
      </header>

      <main className="layout">
        <section className="panel inputs" aria-label="Resume and jobs">
          <div className="panel-head">
            <h2>Resume</h2>
            <button type="button" className="text-button" onClick={loadSample}>
              Load synthetic sample
            </button>
          </div>
          <label className="drop">
            <input
              type="file"
              accept=".pdf,.docx,.txt,.md,application/pdf,text/plain"
              onChange={(event) => {
                const file = event.target.files?.[0] ?? null;
                setResumeFile(file);
                if (file) setResumeText("");
              }}
            />
            <span>{resumeFile ? resumeFile.name : "Drop a PDF, DOCX, or TXT — or paste below"}</span>
          </label>
          <textarea
            aria-label="Resume text"
            placeholder="Or paste the resume text. PDF upload is best-effort; a text paste is the reliable demo path."
            value={resumeText}
            onChange={(event) => {
              setResumeText(event.target.value);
              if (event.target.value) setResumeFile(null);
            }}
          />
          {sampleNote ? <p className="note">{sampleNote}</p> : null}

          <div className="panel-head jobs-head">
            <h2>Job posts</h2>
            <button
              type="button"
              className="text-button"
              onClick={() => setJobs((prev) => (prev.length >= 3 ? prev : [...prev, EMPTY_JOB()]))}
              disabled={jobs.length >= 3}
            >
              Add another
            </button>
          </div>
          {jobs.map((job, index) => (
            <div className="job" key={job.key}>
              <div className="job-bar">
                <label>
                  Title
                  <input
                    value={job.title}
                    placeholder={index === 0 ? "Backend engineer" : "Role title"}
                    onChange={(event) =>
                      setJobs((prev) =>
                        prev.map((item) =>
                          item.key === job.key ? { ...item, title: event.target.value } : item,
                        ),
                      )
                    }
                  />
                </label>
                {jobs.length > 1 ? (
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => setJobs((prev) => prev.filter((item) => item.key !== job.key))}
                  >
                    Remove
                  </button>
                ) : null}
              </div>
              <textarea
                aria-label={`Job post ${index + 1}`}
                placeholder="Paste the posting, including requirements and nice-to-haves."
                value={job.text}
                onChange={(event) =>
                  setJobs((prev) =>
                    prev.map((item) =>
                      item.key === job.key ? { ...item, text: event.target.value } : item,
                    ),
                  )
                }
              />
            </div>
          ))}
          <button type="button" className="primary" disabled={!canScore} onClick={score}>
            {status === "loading" ? "Scoring…" : "Score fit"}
          </button>
          {status === "loading" ? (
            <p className="note" role="status">
              {progress ??
                "Reading the resume, retrieving chunks per requirement, and checking that every match quotes the source."}
            </p>
          ) : null}
          {error ? (
            <p className="error" role="alert">
              {error}
            </p>
          ) : null}
        </section>

        <section className="panel results" aria-live="polite" aria-label="Fit report">
          {status === "idle" && results.length === 0 ? (
            <EmptyState />
          ) : null}
          {status === "loading" ? <LoadingState /> : null}
          {status === "error" && results.length === 0 ? (
            <p className="empty-title">No report yet.</p>
          ) : null}
          {current ? (
            <Report
              results={results}
              active={active}
              onSelect={setActive}
              onCopy={copySummary}
              copied={copied}
            />
          ) : null}
        </section>
      </main>
      <footer>
        Citations are exact substrings of retrieved resume chunks. A public deploy has no accounts —
        treat uploads as shared. The deterministic scorer runs when <code>GROQ_API_KEY</code> is
        unset, and when Groq returns invalid JSON or stays rate-limited. The report names which
        scorer produced the score.
      </footer>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty">
      <p className="empty-kicker">Nothing scored yet</p>
      <p className="empty-title">A fit report shows up here.</p>
      <ol>
        <li>Add a resume.</li>
        <li>Paste a posting with a requirements list.</li>
        <li>Read matches, gaps, and the quoted lines.</li>
      </ol>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="empty">
      <p className="empty-kicker">Working</p>
      <p className="empty-title">Retrieving evidence…</p>
      <div className="skeleton" />
      <div className="skeleton short" />
    </div>
  );
}

function Report({
  results,
  active,
  onSelect,
  onCopy,
  copied,
}: {
  results: FitResponse[];
  active: number;
  onSelect: (index: number) => void;
  onCopy: () => void;
  copied: boolean;
}) {
  const result = results[active];
  const report = result.report;
  return (
    <article>
      {results.length > 1 ? (
        <div className="tabs" role="tablist">
          {results.map((item, index) => (
            <button
              key={item.run_id}
              type="button"
              role="tab"
              aria-selected={index === active}
              className={index === active ? "tab active" : "tab"}
              onClick={() => onSelect(index)}
            >
              <span>{item.job_title}</span>
              <strong>{item.report.overall_score}</strong>
            </button>
          ))}
        </div>
      ) : null}
      <div className="score-row">
        <div>
          <p className="score-kicker">{result.job_title}</p>
          <p className="score">
            {report.overall_score}
            <span>/100</span>
          </p>
        </div>
        <div className="score-side">
          <span className={`verdict verdict-${report.verdict}`}>{report.verdict}</span>
          <button type="button" className="text-button" onClick={onCopy}>
            {copied ? "Copied" : "Copy summary"}
          </button>
        </div>
      </div>
      <p className="summary">{report.summary}</p>
      {report.provider_note ? <p className="note">{report.provider_note}</p> : null}
      {report.guardrail_flags && report.guardrail_flags.length > 0 ? (
        <div className="note guard" role="status">
          <strong>Ignored {report.guardrail_flags.length} line{report.guardrail_flags.length > 1 ? "s" : ""} addressed to the scoring model.</strong>
          <ul>
            {report.guardrail_flags.map((flag) => (
              <li key={flag}>{flag}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="meta">
        {result.provider === "groq" ? "Groq" : "Deterministic scorer"} · {result.model} ·{" "}
        {result.latency_ms} ms total · retrieve {result.retrieve_ms} ms · graph {result.generate_ms}{" "}
        ms · {result.prompt_tokens}+{result.completion_tokens} tokens · $
        {result.cost_usd.toFixed(4)} · run {result.run_id}
      </p>

      <h3>Requirements</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Requirement</th>
              <th>Level</th>
              <th>Result</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {report.requirement_results.map((item) => (
              <tr key={`${item.kind}-${item.requirement}`}>
                <td>{item.requirement}</td>
                <td>{item.kind}</td>
                <td>
                  <span className={`status status-${item.status}`}>{item.status}</span>
                </td>
                <td>
                  {item.evidence ? (
                    <blockquote>
                      “{item.evidence.quote}”
                      <cite>
                        {item.evidence.section} · {item.evidence.chunk_id}
                      </cite>
                    </blockquote>
                  ) : (
                    <span className="gap-copy">{item.rationale}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {report.stretch_flags.length > 0 ? (
        <>
          <h3>Almost</h3>
          <ul className="plain">
            {report.stretch_flags.map((flag) => (
              <li key={flag}>{flag}</li>
            ))}
          </ul>
        </>
      ) : null}

      {report.recommended_resume_edits.length > 0 ? (
        <>
          <h3>Edits, only if true</h3>
          <ul className="plain">
            {report.recommended_resume_edits.map((edit) => (
              <li key={edit}>{edit}</li>
            ))}
          </ul>
        </>
      ) : null}

      <h3>Citations</h3>
      {report.citations.length === 0 ? (
        <p className="note">No cited matches. The gaps above were not supported by retrieved text.</p>
      ) : (
        <ol className="citations">
          {report.citations.map((citation) => (
            <li key={`${citation.chunk_id}-${citation.quote}`}>
              <blockquote>“{citation.quote}”</blockquote>
              <cite>
                {citation.section ?? "section"} · {citation.chunk_id}
              </cite>
            </li>
          ))}
        </ol>
      )}
    </article>
  );
}
