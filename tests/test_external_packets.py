"""Evidence boundaries and financial unit mistakes, using synthetic source rows only."""
import copy

import pytest

from benchmarks.agent_study.external_packets import ScalarPolicy, packets, prompt


def fixture():
    row = dict(task="CodeFinQA", question="What percentage of 80 is 20?",
               context="The part is 20 and the total is 80.", context_type="text",
               options=[], answer="25", program="SECRET_REFERENCE_PROGRAM",
               hidden_rationale="SECRET_RATIONALE")
    policy = ScalarPolicy("Percentages use the percentage number, e.g. 12 for 12%.",
                          1e-8, 0, "synthetic test convention, not official BizBench policy")
    return row, policy


def test_public_packet_and_prompt_exclude_answers_programs_and_scoring_metadata():
    row, policy = fixture()
    public, private = packets(row, revision="a"*40, split="train", row_index=0,
                              source_cluster="synthetic-document", policy=policy)
    assert "25" not in str(prompt(public))
    assert "SECRET" not in str(public) and "answer" not in public and "program" not in public
    assert private["original_answer"] == "25"
    contaminated = copy.deepcopy(public)
    contaminated["answer"] = "25"
    with pytest.raises(ValueError, match="unexpected"):
        prompt(contaminated)


def test_percentage_scaling_and_sign_errors_are_not_accepted():
    _, policy = fixture()
    assert policy.correct(25.0, "25")
    for wrong in (0.25, -25, 2500, True, "25", float("nan"), [25]):
        assert not policy.correct(wrong, "25")


@pytest.mark.parametrize("answer", ["25%", "$25", "1,000", "NaN", "[25]", None])
def test_ambiguous_reference_formats_need_review_before_release(answer):
    row, policy = fixture()
    row["answer"] = answer
    with pytest.raises(ValueError, match="reviewed"):
        packets(row, revision="a"*40, split="train", row_index=0,
                source_cluster="synthetic-document", policy=policy)


def test_cluster_and_frozen_revision_are_required():
    row, policy = fixture()
    with pytest.raises(ValueError, match="revision"):
        packets(row, revision="main", split="train", row_index=0,
                source_cluster="synthetic-document", policy=policy)
    with pytest.raises(ValueError, match="cluster"):
        packets(row, revision="a"*40, split="train", row_index=0,
                source_cluster="", policy=policy)


def test_output_instruction_comes_after_the_library_interface():
    """The frozen study put the instruction first; models then lost it behind ~26k characters."""
    row, policy = fixture()
    public, _ = packets(row, revision="a"*40, split="train", row_index=0,
                        source_cluster="synthetic-document", policy=policy)
    plain = prompt(public)[0]["content"]
    with_library = prompt(public, interface_text="LIBRARY INTERFACE TEXT")[0]["content"]
    assert plain.startswith("Write Python code") and "LIBRARY" not in plain
    assert with_library.startswith("Available library interface:")
    assert with_library.endswith(plain)
    assert with_library.index("LIBRARY INTERFACE TEXT") < with_library.index("Write Python code")
