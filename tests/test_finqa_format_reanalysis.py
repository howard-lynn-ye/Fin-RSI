from benchmarks.agent_study.finqa_format_reanalysis import numeric_final


def test_percentage_recovery_uses_units_not_gold():
    parsed=numeric_final('The percentage change from 2011 to 2012 is approximately 41.43%.')
    assert parsed['accepted'] and parsed['value']=='0.4143'


def test_conflicting_candidates_are_not_selected_by_expected_answer():
    assert not numeric_final('The answer is 12. The answer was 14.')['accepted']


def test_missing_refusal_and_currency_scale_stay_unresolved():
    for text in (None,'It cannot be calculated; the 2012 value was 12.',
                 'The answer is 12 million.','2012'):
        assert not numeric_final(text)['accepted']


def test_ratio_and_bare_answer():
    assert numeric_final('The debt to equity ratio is 2.57:1.')['value']=='2.57'
    assert numeric_final('0.25')['value']=='0.25'
