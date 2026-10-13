"""A sampler failure must preserve diagnostics without inventing probabilities."""
import pytest
from benchmarks.agent_study.transformers_chat import FiniteScores, InvalidSamplingScores


def test_valid_scores_including_grammar_masks_are_returned_unchanged():
    torch = pytest.importorskip('torch')
    scores = torch.tensor([[1., float('-inf'), -3.]])
    original = scores.clone()
    assert FiniteScores(2)(torch.tensor([[9, 8, 7]]), scores) is scores
    assert torch.equal(scores, original)


@pytest.mark.parametrize('values',[[float('nan'), 1.], [float('inf'), 1.], [float('-inf'),float('-inf')]])
def test_invalid_distribution_raises_before_sampling_and_retains_generated_ids(values):
    torch = pytest.importorskip('torch')
    with pytest.raises(InvalidSamplingScores) as caught:
        FiniteScores(2)(torch.tensor([[9, 8, 7, 6]]), torch.tensor([values]))
    assert caught.value.diagnostic['generated_token_ids'] == [[7,6]]


def test_one_empty_row_cannot_hide_behind_another_valid_batch_row():
    torch = pytest.importorskip('torch')
    scores = torch.tensor([[1.,2.], [float('-inf'),float('-inf')]])
    with pytest.raises(InvalidSamplingScores) as caught:
        FiniteScores(1)(torch.tensor([[1,2],[3,4]]), scores)
    assert caught.value.diagnostic['finite_per_row'] == [2,0]