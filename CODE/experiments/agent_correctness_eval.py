from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from experiments.schemes import (
    SchemeConfig,
    assign_artifact_paths,
    destroy_scheme_persistent_state,
    make_scheme,
    rollback_restorable_aux_state_paths,
    scheme_artifact_paths,
)
from agent import AgentConfig, SealedMemoryAgent
from agent_research import ResearchAgentConfig, ResearchAssistantAgent
from agent_team import TeamAgentConfig, TeamAssistantAgent
from llm import OllamaLLM, make_llm

try:
    from experiments.utils import clean
except Exception:
    def clean(paths: list[str]) -> None:
        for p in paths:
            if p and os.path.exists(p):
                os.remove(p)


class Progress:
    def __init__(self, total: int, desc: str = ""):
        self.total = max(1, int(total))
        self.desc = desc
        self.start = time.perf_counter()

    def update(self, i: int):
        pct = 100 * i / self.total
        elapsed = time.perf_counter() - self.start
        print(f"\r{self.desc} {pct:6.2f}% ({i}/{self.total}) elapsed={elapsed:0.1f}s", end="", flush=True)

    def done(self):
        print("")


@dataclass(frozen=True)
class ArtifactSpec:
    source_kind: str
    source_payload: dict[str, Any]
    derived_kind: str
    derived_payload: dict[str, Any]
    question: str
    expected_token: str


SPECS = {
    "personal": ArtifactSpec(
        source_kind="fact",
        source_payload={"fact": "my favorite drink is tea"},
        derived_kind="profile_summary",
        derived_payload={"summary": "User profile summary: favorite drink is tea"},
        question="what is my favorite drink?",
        expected_token="tea",
    ),
    "team": ArtifactSpec(
        source_kind="team_note",
        source_payload={"note": "the sprint deadline is Friday"},
        derived_kind="team_digest",
        derived_payload={"digest": "Team digest: the sprint deadline is Friday"},
        question="when is the sprint deadline?",
        expected_token="friday",
    ),
    "research": ArtifactSpec(
        source_kind="research_doc",
        source_payload={"doc": "SAGE uses epoch-based key derivation for rollback-resilient deletion."},
        derived_kind="research_summary",
        derived_payload={"summary": "Research summary: SAGE uses epoch-based key derivation for rollback-resilient deletion."},
        question="what does SAGE use for rollback-resilient deletion?",
        expected_token="epoch",
    ),
}


def safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def safe_close(obj: Any) -> None:
    if hasattr(obj, "close"):
        try:
            obj.close()
        except Exception:
            pass


def snapshot_file(src: str) -> str | None:
    if src and os.path.exists(src):
        dst = src + ".snapshot"
        shutil.copy2(src, dst)
        return dst
    return None


def restore_file(dst: str, snap: str) -> None:
    if snap and os.path.exists(snap):
        shutil.copy2(snap, dst)


def snapshot_scheme_state(cfg: Any) -> dict[str, str]:
    """
    Snapshot attacker-controlled local scheme state only.

    KMS state is external and is NOT rolled back.
    SAGE trusted state (epochs/root key/dev master) is also NOT rolled back.
    """
    snaps: dict[str, str] = {}
    for p in rollback_restorable_aux_state_paths(cfg):
        if isinstance(p, str) and os.path.exists(p):
            snap = snapshot_file(p)
            if snap:
                snaps[p] = snap
    return snaps


def restore_scheme_state(snaps: dict[str, str]) -> None:
    for orig, snap in snaps.items():
        restore_file(orig, snap)


def cleanup_snapshots(snaps: dict[str, str]) -> None:
    for snap in snaps.values():
        safe_remove(snap)


def snapshot(db_path: str, snap_path: str) -> None:
    if os.path.exists(db_path):
        shutil.copy2(db_path, snap_path)


def rollback(db_snap: str, db_path: str) -> None:
    if os.path.exists(db_snap):
        shutil.copy2(db_snap, db_path)


def make_trial_cfg(scheme: str, db_path: str, trial_tag: str) -> SchemeConfig:
    cfg = SchemeConfig(db_path=db_path, scheme=scheme)
    return assign_artifact_paths(cfg, f"{db_path}.{trial_tag}")


def cleanup_trial_files(cfg: SchemeConfig, snap_path: str) -> None:
    destroy_scheme_persistent_state(cfg)
    safe_remove(snap_path)
    for p in scheme_artifact_paths(cfg, include_wal=True):
        safe_remove(p)


def store_item(s, scope_id: str, payload: dict[str, Any], kind: str) -> str:
    try:
        return s.put(scope_id, payload, kind=kind)
    except TypeError:
        return s.put(scope_id, payload)


def store_derived(
    s,
    scope_id: str,
    payload: dict[str, Any],
    kind: str,
    derived_from_item_ids: Iterable[str],
    source_scope_ids: Iterable[str],
) -> str:
    if hasattr(s, "put_derived"):
        try:
            return s.put_derived(
                scope_id,
                payload,
                kind=kind,
                derived_from_item_ids=list(derived_from_item_ids),
                source_scope_ids=list(source_scope_ids),
            )
        except Exception:
            pass
    if hasattr(s, "mem") and hasattr(s.mem, "put_derived"):
        return s.mem.put_derived(
            scope_id,
            payload,
            kind=kind,
            derived_from_item_ids=list(derived_from_item_ids),
            source_scope_ids=list(source_scope_ids),
        )
    return store_item(s, scope_id, payload, kind)


def safe_forget_scope(s, scope_id: str) -> None:
    """
    Prefer physical deletion when the scheme supports it, but gracefully fall
    back for baselines with narrower forget_scope signatures.
    """
    try:
        s.forget_scope(scope_id, delete_ciphertext_rows=True, propagate=True)
        return
    except TypeError:
        pass
    except Exception:
        # Some baselines may reject extra kwargs with a different exception type.
        pass

    try:
        s.forget_scope(scope_id, delete_ciphertext_rows=True)
        return
    except TypeError:
        pass
    except Exception:
        pass

    try:
        s.forget_scope(scope_id)
    except TypeError:
        # Last-ditch compatibility: some wrappers may expose underlying mem object.
        if hasattr(s, "mem") and hasattr(s.mem, "forget_scope"):
            try:
                s.mem.forget_scope(scope_id, delete_ciphertext_rows=True, propagate=True)
                return
            except TypeError:
                try:
                    s.mem.forget_scope(scope_id, delete_ciphertext_rows=True)
                    return
                except TypeError:
                    s.mem.forget_scope(scope_id)
                    return
        raise


def make_ollama_llm(model: str = "llama3.2", force_cpu: bool = False) -> OllamaLLM:
    options = {"num_gpu": 0} if force_cpu else {}
    return make_llm(
        "ollama",
        model=model,
        temperature=0.2,
        max_tokens=128,
        options=options,
    )


def build_agent(agent_kind: str, llm: OllamaLLM, mem: Any, scope_id: str):
    if agent_kind == "personal":
        return SealedMemoryAgent(
            llm,
            mem,
            AgentConfig(
                scope_id=scope_id,
                enable_llm_memory_summarizer=False,
                store_message_artifacts=False,
            ),
        )
    if agent_kind == "team":
        return TeamAssistantAgent(llm, mem, TeamAgentConfig(scope_id=scope_id))
    if agent_kind == "research":
        return ResearchAssistantAgent(llm, mem, ResearchAgentConfig(scope_id=scope_id))
    raise ValueError(f"Unknown agent kind: {agent_kind}")


def answer_matches(agent_kind: str, out: str, expected: str) -> bool:
    text = (out or "").strip().lower()
    token = expected.lower()
    if agent_kind == "research":
        return token in text
    return text == token or token in text


def safe_agent_chat(agent_kind: str, agent: Any, question: str, expected: str) -> bool:
    try:
        return answer_matches(agent_kind, agent.chat(question), expected)
    except KeyError:
        return False
    except Exception as e:
        msg = str(e).lower()
        if "deleted" in msg or "not found" in msg or "missing key" in msg:
            return False
        raise


@dataclass
class TrialResult:
    source_visible_before: bool
    derived_visible_before: bool
    source_hidden_after_forget: bool
    derived_hidden_after_forget: bool
    source_rollback_recovery: bool
    derived_rollback_recovery: bool


def run_one_trial(
    agent_kind: str,
    scheme: str,
    db_path: str,
    snap_path: str,
    trial_tag: str,
    ollama_model: str,
    ollama_force_cpu: bool,
) -> TrialResult:
    spec = SPECS[agent_kind]
    cfg = make_trial_cfg(scheme, db_path, trial_tag)
    cleanup_trial_files(cfg, snap_path)

    source_scope = f"{agent_kind}:{scheme}:{trial_tag}:source"
    derived_scope = f"{agent_kind}:{scheme}:{trial_tag}:derived"

    s = make_scheme(cfg)
    llm = make_ollama_llm(ollama_model, force_cpu=ollama_force_cpu)
    source_id = store_item(s, source_scope, spec.source_payload, spec.source_kind)
    _ = store_derived(
        s,
        derived_scope,
        spec.derived_payload,
        spec.derived_kind,
        derived_from_item_ids=[source_id],
        source_scope_ids=[source_scope],
    )

    source_agent = build_agent(agent_kind, llm, s, source_scope)
    derived_agent = build_agent(agent_kind, llm, s, derived_scope)
    source_visible_before = safe_agent_chat(
        agent_kind, source_agent, spec.question, spec.expected_token
    )
    derived_visible_before = safe_agent_chat(
        agent_kind, derived_agent, spec.question, spec.expected_token
    )

    safe_close(s)
    snapshot(db_path, snap_path)
    state_snaps = snapshot_scheme_state(cfg)

    s = make_scheme(cfg)
    safe_forget_scope(s, source_scope)
    source_agent = build_agent(agent_kind, llm, s, source_scope)
    derived_agent = build_agent(agent_kind, llm, s, derived_scope)

    source_hidden_after_forget = not safe_agent_chat(
        agent_kind, source_agent, spec.question, spec.expected_token
    )
    derived_hidden_after_forget = not safe_agent_chat(
        agent_kind, derived_agent, spec.question, spec.expected_token
    )
    safe_close(s)

    rollback(snap_path, db_path)
    if rollback_restorable_aux_state_paths(cfg):
        restore_scheme_state(state_snaps)

    s2 = make_scheme(cfg)
    source_agent = build_agent(agent_kind, llm, s2, source_scope)
    derived_agent = build_agent(agent_kind, llm, s2, derived_scope)
    source_rollback_recovery = safe_agent_chat(
        agent_kind, source_agent, spec.question, spec.expected_token
    )
    derived_rollback_recovery = safe_agent_chat(
        agent_kind, derived_agent, spec.question, spec.expected_token
    )
    safe_close(s2)

    cleanup_snapshots(state_snaps)
    cleanup_trial_files(cfg, snap_path)

    return TrialResult(
        source_visible_before=source_visible_before,
        derived_visible_before=derived_visible_before,
        source_hidden_after_forget=source_hidden_after_forget,
        derived_hidden_after_forget=derived_hidden_after_forget,
        source_rollback_recovery=source_rollback_recovery,
        derived_rollback_recovery=derived_rollback_recovery,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/agent_correctness.csv")
    ap.add_argument("--workdir", default="results/agent_correctness_runs")
    ap.add_argument("--schemes", default="plain,static,sealed_no_rp,sqlite_envelope,kms,sage")
    ap.add_argument("--agents", default="personal,team,research")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--ollama-model", default="llama3.2")
    ap.add_argument(
        "--ollama-force-cpu",
        action="store_true",
        help="Force Ollama to run on CPU by setting num_gpu=0.",
    )
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    rows: list[dict[str, Any]] = []
    total = len(schemes) * len(agents)
    progress = Progress(total, "[agent_correctness] configs")
    step = 0

    for agent_kind in agents:
        for scheme in schemes:
            step += 1
            progress.update(step)

            source_before = 0
            derived_before = 0
            source_hidden = 0
            derived_hidden = 0
            source_rollback_hits = 0
            derived_rollback_hits = 0

            trial_prog = Progress(args.trials, " trials")
            for t in range(args.trials):
                trial_tag = uuid.uuid4().hex
                db_path = os.path.join(
                    args.workdir,
                    f"{agent_kind}_{scheme}_{t}_{trial_tag}.db",
                )
                snap_path = f"{db_path}.snap"
                tr = run_one_trial(
                    agent_kind,
                    scheme,
                    db_path,
                    snap_path,
                    trial_tag,
                    args.ollama_model,
                    args.ollama_force_cpu,
                )

                source_before += int(tr.source_visible_before)
                derived_before += int(tr.derived_visible_before)
                source_hidden += int(tr.source_hidden_after_forget)
                derived_hidden += int(tr.derived_hidden_after_forget)
                source_rollback_hits += int(tr.source_rollback_recovery)
                derived_rollback_hits += int(tr.derived_rollback_recovery)
                trial_prog.update(t + 1)
            trial_prog.done()

            row = dict(
                agent=agent_kind,
                scheme=scheme,
                trials=args.trials,
                acc_source_visible_before=source_before / args.trials,
                acc_derived_visible_before=derived_before / args.trials,
                acc_source_hidden_after_forget=source_hidden / args.trials,
                acc_derived_hidden_after_forget=derived_hidden / args.trials,
                source_rollback_recovery=source_rollback_hits / args.trials,
                derived_rollback_recovery=derived_rollback_hits / args.trials,
            )
            rows.append(row)
            print(
                f"[{agent_kind}:{scheme}] "
                f"before(source/derived)=({row['acc_source_visible_before']:.2f}/{row['acc_derived_visible_before']:.2f}) "
                f"after_hide(source/derived)=({row['acc_source_hidden_after_forget']:.2f}/{row['acc_derived_hidden_after_forget']:.2f}) "
                f"rollback(source/derived)=({row['source_rollback_recovery']:.2f}/{row['derived_rollback_recovery']:.2f})"
            )

    progress.done()
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
