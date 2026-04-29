"""Tests for the openvibe skills system.

Covers:
- SkillRegistry: register, get, all, user_invocable, search, find_best
- SimulateSkill: match_intent, get_prompt, flag parsing
- BuildAndEvalSkill: match_intent, extract_args, get_prompt
- init_bundled_skills: exactly two skills registered, removed skills absent
"""

from __future__ import annotations

import pytest

from openvibe.skill.base import CostTier, SkillDefinition, SkillExample, SkillResult, SkillStatus
from openvibe.skill.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Helpers — minimal concrete skill for registry tests
# ---------------------------------------------------------------------------


class _SimpleSkill(SkillDefinition):
    name = "simple"
    description = "A simple test skill."
    aliases = ["s", "smp"]
    tags = ["test", "simple"]
    capabilities = ["testing"]
    cost_estimate = CostTier.LOW
    user_invocable = True

    def get_prompt(self, args: str) -> str:
        return f"Do simple: {args}"


class _HiddenSkill(SkillDefinition):
    name = "hidden"
    description = "Not user-facing."
    user_invocable = False

    def get_prompt(self, args: str) -> str:
        return "hidden"


# ---------------------------------------------------------------------------
# SkillRegistry tests
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def _reg(self) -> SkillRegistry:
        """Return a fresh registry for each test."""
        r = SkillRegistry()
        r.register(_SimpleSkill())
        return r

    def test_register_and_get_by_name(self):
        r = self._reg()
        skill = r.get("simple")
        assert skill is not None
        assert skill.name == "simple"

    def test_get_by_alias(self):
        r = self._reg()
        assert r.get("s") is not None
        assert r.get("smp") is not None

    def test_get_case_insensitive(self):
        r = self._reg()
        assert r.get("Simple") is not None
        assert r.get("SIMPLE") is not None

    def test_get_missing_returns_none(self):
        r = self._reg()
        assert r.get("nonexistent") is None

    def test_all_returns_registered(self):
        r = self._reg()
        names = [s.name for s in r.all()]
        assert "simple" in names

    def test_user_invocable_filters_hidden(self):
        r = SkillRegistry()
        r.register(_SimpleSkill())
        r.register(_HiddenSkill())
        visible = [s.name for s in r.user_invocable()]
        assert "simple" in visible
        assert "hidden" not in visible

    def test_register_overwrites_existing(self):
        r = self._reg()
        original = r.get("simple")

        class _Replacement(_SimpleSkill):
            description = "replaced"

        r.register(_Replacement())
        replaced = r.get("simple")
        assert replaced.description == "replaced"

    def test_search_by_name_exact(self):
        r = self._reg()
        results = r.search("simple")
        assert len(results) > 0
        assert results[0][0].name == "simple"

    def test_search_by_tag(self):
        r = self._reg()
        results = r.search("testing")
        assert any(s.name == "simple" for s, _ in results)

    def test_search_returns_scores(self):
        r = self._reg()
        results = r.search("simple", top_k=5)
        for skill, score in results:
            assert score > 0

    def test_find_best_returns_skill(self):
        r = self._reg()
        best = r.find_best("simple")
        assert best is not None
        assert best.name == "simple"

    def test_find_best_empty_registry(self):
        r = SkillRegistry()
        assert r.find_best("anything") is None

    def test_search_top_k_respected(self):
        r = SkillRegistry()
        for i in range(10):
            class _S(_SimpleSkill):
                name = f"skill_{i}"
            _S.name = f"skill_{i}"
            r.register(_S())
        results = r.search("simple", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# SkillDefinition.match_intent default implementation
# ---------------------------------------------------------------------------


class TestSkillDefinitionMatchIntent:
    def test_alias_match_boosts_score(self):
        s = _SimpleSkill()
        score = s.match_intent("run smp on this file")
        assert score > 0.3

    def test_short_input_penalised(self):
        s = _SimpleSkill()
        long_score = s.match_intent("run the simple test skill here")
        short_score = s.match_intent("simple")
        assert long_score >= short_score

    def test_no_match_returns_zero(self):
        s = _SimpleSkill()
        score = s.match_intent("completely unrelated query about databases")
        # No overlap → should be 0 or very low
        assert score < 0.2

    def test_score_bounded_to_one(self):
        s = _SimpleSkill()
        score = s.match_intent(
            "simple smp s test testing testing testing testing testing"
        )
        assert score <= 1.0


# ---------------------------------------------------------------------------
# SimulateSkill tests
# ---------------------------------------------------------------------------


class TestSimulateSkill:
    def _skill(self):
        from openvibe.skill.bundled.simulate import SimulateSkill
        return SimulateSkill()

    def test_name(self):
        assert self._skill().name == "simulate"

    def test_user_invocable(self):
        assert self._skill().user_invocable is True

    def test_get_prompt_contains_simulate_tool(self):
        prompt = self._skill().get_prompt("billing support")
        assert "simulate" in prompt.lower()
        assert "billing support" in prompt

    def test_get_prompt_empty_args(self):
        prompt = self._skill().get_prompt("")
        assert isinstance(prompt, str)
        assert len(prompt) > 10

    def test_get_prompt_parses_n_flag(self):
        prompt = self._skill().get_prompt("--n=3 billing support")
        assert "3" in prompt

    def test_get_prompt_parses_n_space_flag(self):
        prompt = self._skill().get_prompt("--n 7 customer service")
        assert "7" in prompt

    def test_get_prompt_parses_output_flag(self):
        prompt = self._skill().get_prompt("--output=/tmp/out billing")
        assert "/tmp/out" in prompt

    def test_get_prompt_parses_mode_flag(self):
        prompt = self._skill().get_prompt("--mode=design billing")
        assert "design" in prompt

    def test_match_intent_simulation_keyword(self):
        skill = self._skill()
        score = skill.match_intent("simulate a workflow with harness")
        assert score > 0.2

    def test_match_intent_alias_sim(self):
        skill = self._skill()
        score = skill.match_intent("run sim on this support context")
        assert score > 0.2


# ---------------------------------------------------------------------------
# BuildAndEvalSkill tests
# ---------------------------------------------------------------------------


class TestBuildAndEvalSkill:
    def _skill(self):
        from openvibe.skill.bundled.build_and_eval import BuildAndEvalSkill
        return BuildAndEvalSkill()

    def test_name(self):
        assert self._skill().name == "build-eval"

    def test_user_invocable(self):
        assert self._skill().user_invocable is True

    def test_aliases(self):
        assert "beval" in self._skill().aliases

    def test_extract_args_finds_md_path(self):
        skill = self._skill()
        assert skill.extract_args("build and evaluate examples/github_dashboard/TASK.md") \
            == "examples/github_dashboard/TASK.md"

    def test_extract_args_empty_when_no_path(self):
        skill = self._skill()
        assert skill.extract_args("build and evaluate something") == ""

    def test_get_prompt_uses_task_file(self):
        skill = self._skill()
        prompt = skill.get_prompt("examples/myapp/TASK.md")
        assert "examples/myapp/TASK.md" in prompt

    def test_get_prompt_defaults_to_task_md(self):
        skill = self._skill()
        prompt = skill.get_prompt("")
        assert "TASK.md" in prompt

    def test_get_prompt_includes_simulate_tool(self):
        skill = self._skill()
        prompt = skill.get_prompt("examples/myapp/TASK.md")
        assert "simulate" in prompt

    def test_get_prompt_includes_working_dir(self):
        skill = self._skill()
        prompt = skill.get_prompt("examples/myapp/TASK.md")
        assert "working_dir" in prompt

    def test_get_prompt_output_path_sibling(self):
        skill = self._skill()
        prompt = skill.get_prompt("examples/myapp/TASK.md")
        assert "eval_output" in prompt

    def test_match_intent_task_md_build(self):
        skill = self._skill()
        score = skill.match_intent("build and evaluate examples/myapp/TASK.md")
        assert score >= 0.7

    def test_match_intent_build_eval_keywords(self):
        skill = self._skill()
        score = skill.match_intent("build and evaluate the pipeline")
        assert score > 0.3

    def test_match_intent_task_md_alone_low(self):
        skill = self._skill()
        score = skill.match_intent("task.md")
        # Short input, no build/eval context
        assert score < 0.7

    def test_match_intent_pipeline_harness(self):
        skill = self._skill()
        score = skill.match_intent("run the pipeline harness")
        assert score > 0.1


# ---------------------------------------------------------------------------
# init_bundled_skills — registry completeness
# ---------------------------------------------------------------------------


class TestInitBundledSkills:
    def _fresh_registry(self):
        """Return a new registry with bundled skills registered."""
        from openvibe.skill.registry import SkillRegistry
        r = SkillRegistry()
        # Import and register directly so we don't pollute the global singleton
        from openvibe.skill.bundled.simulate import SimulateSkill
        from openvibe.skill.bundled.build_and_eval import BuildAndEvalSkill
        r.register(SimulateSkill())
        r.register(BuildAndEvalSkill())
        return r

    def test_exactly_two_skills_registered(self):
        r = self._fresh_registry()
        assert len(r.all()) == 2

    def test_simulate_skill_registered(self):
        r = self._fresh_registry()
        assert r.get("simulate") is not None

    def test_build_eval_skill_registered(self):
        r = self._fresh_registry()
        assert r.get("build-eval") is not None

    def test_build_eval_alias_beval(self):
        r = self._fresh_registry()
        assert r.get("beval") is not None

    def test_simulate_alias_sim(self):
        r = self._fresh_registry()
        assert r.get("sim") is not None

    def test_removed_brainstorm_not_present(self):
        r = self._fresh_registry()
        assert r.get("brainstorm") is None

    def test_removed_draft_not_present(self):
        r = self._fresh_registry()
        assert r.get("draft") is None

    def test_removed_explain_not_present(self):
        r = self._fresh_registry()
        assert r.get("explain") is None

    def test_removed_summarize_not_present(self):
        r = self._fresh_registry()
        assert r.get("summarize") is None

    def test_removed_skills_not_importable(self):
        """The deleted skill modules must not exist."""
        import importlib
        for mod in (
            "openvibe.skill.bundled.brainstorm",
            "openvibe.skill.bundled.draft",
            "openvibe.skill.bundled.explain",
            "openvibe.skill.bundled.summarize",
        ):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                importlib.import_module(mod)

    def test_all_bundled_user_invocable(self):
        r = self._fresh_registry()
        for skill in r.all():
            assert skill.user_invocable is True


# ---------------------------------------------------------------------------
# SkillResult / SkillStatus data objects
# ---------------------------------------------------------------------------


class TestSkillDataObjects:
    def test_skill_result_defaults(self):
        r = SkillResult(skill_name="test", status=SkillStatus.SUCCESS, output="ok")
        assert r.attempt == 1
        assert r.error is None
        assert r.metadata == {}

    def test_skill_status_values(self):
        assert SkillStatus.SUCCESS == "success"
        assert SkillStatus.FAILED == "failed"
        assert SkillStatus.RETRIED == "retried"
        assert SkillStatus.FALLBACK == "fallback"
        assert SkillStatus.PARTIAL == "partial"

    def test_cost_tier_values(self):
        assert CostTier.LOW == "low"
        assert CostTier.MEDIUM == "medium"
        assert CostTier.HIGH == "high"

    def test_skill_example_fields(self):
        ex = SkillExample(input="some code", description="Review this code")
        assert ex.input == "some code"
        assert ex.description == "Review this code"
