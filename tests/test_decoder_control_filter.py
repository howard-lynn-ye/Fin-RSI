"""Exercise the real LMFE token tree, including merged tokens and escapes."""
import pytest

from benchmarks.agent_study import trading_schema_v8 as schema
from benchmarks.agent_study import trading_research_v9 as research


def token_filter(parser, *, harden=True):
    pytest.importorskip('lmformatenforcer')
    from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData
    texts = [chr(c) for c in range(128)] + [
        prefix + chr(c) + suffix for c in range(32)
        for prefix, suffix in [('a', ''), ('a', '"'), ('"', '"'), ('"a', '"')]]
    texts += ['"a\\t"', '"a\\u0000"']
    ids = {s: i for i, s in enumerate(texts)}
    stop = len(texts)
    data = TokenEnforcerTokenizerData([(i, s, True) for i, s in enumerate(texts)],
        lambda tokens: ''.join(texts[i] for i in tokens), stop, False, stop + 1)
    if harden:
        schema.harden_string_token_cache(data)
    return TokenEnforcer(data, parser), ids, [stop], stop


def advance(enforcer, ids, sequence, text):
    for c in text:
        assert ids[c] in enforcer.get_allowed_tokens(sequence).allowed_tokens
        sequence.append(ids[c])
    return enforcer.get_allowed_tokens(sequence).allowed_tokens


@pytest.mark.parametrize('stage', ['final', 'research'])
def test_all_raw_control_tokens_blocked_inside_strings_without_blocking_escapes(stage):
    pytest.importorskip('lmformatenforcer')
    if stage == 'final':
        parser = schema.parser_factory()
        prefix = '{"tool":"submit","arguments":{"allocation":{"CASH":1},"rationale":'
    else:
        parser = research.parser_factory(research.output_contract('library'))
        prefix = '{"tool":"research_context","arguments":{"query":'
    enforcer, ids, sequence, stop = token_filter(parser)
    allowed = advance(enforcer, ids, sequence, prefix)
    # Merged tokens can enter and leave a string in one step, bypassing its
    # freetext shortcut. Both the alphabet and cache must reject raw controls.
    for c in range(32):
        assert ids['"' + chr(c) + '"'] not in allowed
        assert ids['"a' + chr(c) + '"'] not in allowed
    assert ids['"a\\t"'] in allowed
    allowed = advance(enforcer, ids, sequence, '"a')
    for c in range(32):
        for token in (chr(c), 'a' + chr(c), 'a' + chr(c) + '"'):
            assert ids[token] not in allowed, (stage, repr(token))
    allowed = advance(enforcer, ids, sequence, '\\t\\n\\u0000"}}')
    assert stop in allowed


def test_unhardened_pinned_token_cache_reproduces_the_tab_leak():
    pytest.importorskip('lmformatenforcer')
    enforcer, ids, sequence, _ = token_filter(schema.parser_factory(), harden=False)
    allowed = advance(enforcer, ids, sequence,
        '{"tool":"submit","arguments":{"allocation":{"CASH":1},"rationale":"a')
    assert ids['a\t'] in allowed


def test_json_whitespace_outside_strings_stays_available():
    pytest.importorskip('lmformatenforcer')
    enforcer, ids, sequence, stop = token_filter(schema.parser_factory())
    allowed = advance(enforcer, ids, sequence, '\t{\n"tool":"hold",\r"arguments":{}}')
    assert stop in allowed


def test_output_receipts_identify_both_hardened_stages():
    assert schema.output_contract(True)['string_token_filter'] == schema.STRING_TOKEN_FILTER
    assert research.output_contract('library')['string_token_filter'] == schema.STRING_TOKEN_FILTER
    assert schema.output_contract(False) == {'transport': 'text'}


def test_research_query_limits_are_visible_and_fail_before_retrieval():
    from benchmarks.agent_study.trading_runtime_v9 import Tools, RAG_GUIDE
    tools = object.__new__(Tools)
    tools.arm = 'library'
    assert '1..512' in RAG_GUIDE
    assert tools.help_tool('research_context')['query_contract'] == dict(
        type='string', minLength=1, maxLength=512, whitespace_only=False)
    with pytest.raises(ValueError, match='received 922 characters'):
        tools.research_context('x' * 922)
    with pytest.raises(ValueError, match='nonblank string'):
        tools.research_context('   ')
