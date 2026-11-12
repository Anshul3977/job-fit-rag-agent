"""Two labeled pairs through the same fit path the eval script uses. Not the full 20."""

import json
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "eval" / "dataset"
LABELS = json.loads((Path(__file__).resolve().parents[1] / "eval" / "labels.json").read_text())


def test_labels_are_synthetic_and_cover_twenty_pairs():
    assert LABELS["meta"]["synthetic"] is True
    assert len(LABELS["pairs"]) == 20
    assert {pair["verdict"] for pair in LABELS["pairs"]} == {"strong", "maybe", "skip"}


def test_two_obvious_pairs(client):
    expectations = {"p01": "strong", "p12": "skip"}
    chosen = [pair for pair in LABELS["pairs"] if pair["id"] in expectations]
    resume_ids = {}
    job_ids = {}
    for pair in chosen:
        if pair["resume"] not in resume_ids:
            text = (DATASET / "resumes" / f"{pair['resume']}.txt").read_text(encoding="utf-8")
            response = client.post(
                "/ingest/resume",
                files={"file": (f"{pair['resume']}.txt", text.encode(), "text/plain")},
            )
            assert response.status_code == 200
            resume_ids[pair["resume"]] = response.json()["resume_id"]
        if pair["job"] not in job_ids:
            text = (DATASET / "jobs" / f"{pair['job']}.txt").read_text(encoding="utf-8")
            response = client.post("/ingest/job", json={"text": text, "title": pair["job"]})
            assert response.status_code == 200
            job_ids[pair["job"]] = response.json()["job_id"]
        fit = client.post(
            "/fit",
            json={"resume_id": resume_ids[pair["resume"]], "job_id": job_ids[pair["job"]]},
        )
        assert fit.status_code == 200
        report = fit.json()["report"]
        assert report["verdict"] == expectations[pair["id"]]
        if report["verdict"] == "strong":
            assert report["citations"]
            assert all(item["quote"] for item in report["matching_skills"])
