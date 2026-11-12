"""Synthetic demo documents. These are not real people."""

from pathlib import Path

from app.schemas import SampleJob, SamplesResponse

DATASET = Path(__file__).resolve().parents[1] / "eval" / "dataset"


def load_samples() -> SamplesResponse:
    resume = (DATASET / "resumes" / "avery_chen.txt").read_text(encoding="utf-8")
    backend = (DATASET / "jobs" / "backend_python.txt").read_text(encoding="utf-8")
    platform = (DATASET / "jobs" / "platform.txt").read_text(encoding="utf-8")
    return SamplesResponse(
        label="synthetic",
        note=(
            "Sample resume and postings are fictional, written for this demo. "
            "Do not upload a real resume to a public deploy."
        ),
        resume_filename="avery_chen_synthetic.txt",
        resume_text=resume,
        jobs=[
            SampleJob(title="Backend Engineer", text=backend),
            SampleJob(title="Platform Engineer", text=platform),
        ],
    )
