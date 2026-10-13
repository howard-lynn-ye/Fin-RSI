"""Final-stage formatting only: the model chooses the action and every amount.

The grammar does not enforce financial validity. Existing submission validation
still rejects negative, nonfinite and all-zero parts. No values are defaulted.
"""
import hashlib
import json

from benchmarks.agent_study.market_data import TICKERS
from benchmarks.agent_study.trading_submission_v8 import unique_keys

DECODER_VERSION = '0.11.3'
TRANSPORT = 'lm-format-enforcer'
MAX_RATIONALE = 256
STRING_TOKEN_FILTER = 'exclude-raw-controls-v1'
PARSER_OPTIONS = dict(max_consecutive_whitespaces=12, force_json_field_order=False)


class StrictJSONFailure(ValueError):
    pass


class DecoderFailure(RuntimeError):
    def __init__(self, response, prior_turns, final_output, request_messages=None):
        super().__init__('infrastructure qualification failed: broken structured decoder')
        self.diagnostic = dict(response=response, prior_turns=prior_turns,
                               final_output=final_output, request_messages=request_messages, status='infrastructure_failure')


def persist_failure(root, name, failure):
    from benchmarks.agent_study import trading_study as old
    # Immutable, private diagnostic. It is never an accepted decision or a return.
    old.write(root / 'infrastructure-failures' / (name + '.json'), failure.diagnostic)


def decision_schema():
    def obj(properties, required):
        return dict(type='object', properties=properties, required=required,
                    additionalProperties=False)
    allocation = obj({ticker: {'type': 'number'} for ticker in (*TICKERS, 'CASH')}, ['CASH'])
    submit = obj({'tool': {'type': 'string', 'enum': ['submit']},
                  'arguments': obj({'allocation': allocation,
                      'rationale': {'type': 'string', 'maxLength': MAX_RATIONALE}}, ['allocation'])},
                 ['tool', 'arguments'])
    hold = obj({'tool': {'type': 'string', 'enum': ['hold']},
                'arguments': obj({}, [])}, ['tool', 'arguments'])
    return {'anyOf': [submit, hold]}


def output_contract(structured=False):
    if not structured:
        return {'transport': 'text'}
    digest = hashlib.sha256(json.dumps(decision_schema(), sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    return dict(transport=TRANSPORT, version=DECODER_VERSION, schema_sha256=digest,
                stage='final_only', string_token_filter=STRING_TOKEN_FILTER, **PARSER_OPTIONS)


def check_contract(contract):
    if contract not in (output_contract(False), output_contract(True)):
        raise ValueError('unqualified final output contract')


def check_format(text):
    """Independently reject a broken decoder; do not strip wrappers or repair fields."""
    def non_json_constant(value):
        raise ValueError('non-JSON numeric constant: ' + value)
    try:
        value = json.loads(text, object_pairs_hook=unique_keys, parse_constant=non_json_constant)
    except (ValueError, RecursionError) as exc:
        raise StrictJSONFailure(str(exc)) from exc
    if not isinstance(value, dict) or set(value) != {'tool', 'arguments'}:
        raise ValueError('structured output envelope changed')
    args = value['arguments']
    if value['tool'] == 'hold' and args == {}:
        return
    if (value['tool'] != 'submit' or not isinstance(args, dict) or
            'allocation' not in args or set(args) - {'allocation', 'rationale'}):
        raise ValueError('structured output arguments changed')
    parts = args['allocation']
    if (not isinstance(parts, dict) or 'CASH' not in parts or
            set(parts) - set(TICKERS) - {'CASH'} or
            any(type(x) not in (int, float) for x in parts.values())):
        raise ValueError('structured allocation shape changed')
    if 'rationale' in args and (not isinstance(args['rationale'], str) or
                               len(args['rationale']) > MAX_RATIONALE):
        raise ValueError('structured rationale shape changed')


def check_turn(turn, contract):
    if (turn.get('backend_metadata') or {}).get('final_output') != contract:
        raise ValueError('structured decoder receipt absent or changed')
    if turn['finish_reason'] != 'length':
        try:
            check_format(turn['response'])
        except StrictJSONFailure:
            # A filter is not a strict JSON parser. Preserve known parser leniency
            # as an explicit output-interface error, with the same correction budget.
            if grammar_permits(turn['response']):
                return 'structured-json-rejected'
            raise
    return None


def prepare_decoder(tokenizer, stop_token_ids):
    from importlib.metadata import version
    if version('lm-format-enforcer') != DECODER_VERSION:
        raise ValueError('unqualified lm-format-enforcer version')
    if tokenizer.eos_token_id not in stop_token_ids:
        raise ValueError('decoder EOS differs from qualified generation stop tokens')
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    return harden_string_token_cache(build_token_enforcer_tokenizer_data(tokenizer))


def harden_string_token_cache(data):
    """Keep LMFE's fast path, but exclude every raw JSON control character.

    LMFE 0.11.3 checks newlines/CR in its freetext cache, but misses tabs and
    other U+0000..001F characters. A real Nemo token passed that cache while
    the character parser rejected it. Rebuild only the string shortcut cache;
    the ordinary token tree still accepts valid whitespace outside strings.
    Escaped control sequences remain available, and no model text is repaired.
    """
    from lmformatenforcer.tokenizerprefixtree import JsonFreetextTokenCache
    cache = JsonFreetextTokenCache(data.use_bitmask, data.vocab_size)
    for token_id, text, _ in data.regular_tokens:
        if not any(ord(character) < 32 for character in text):
            cache.add_token(text, token_id)
    cache.freeze()
    data.tokenizer_tree.json_freetext_tokens = cache
    # Also cover tokens that open a string and contain controls in one step.
    data.tokenizer_alphabet = ''.join(c for c in data.tokenizer_alphabet if ord(c) >= 32)
    return data


def prefix_filter(tokenizer_data, prompt):
    # Do not erase reasoning or change the publisher's chat template to force JSON.
    if prompt.rstrip().endswith('<think>'):
        raise ValueError('structured final output requires qualification of this reasoning template')
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser_factory())


def parser_factory():
    from importlib.metadata import version
    if version('lm-format-enforcer') != DECODER_VERSION:
        raise ValueError('unqualified lm-format-enforcer version')
    from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
    return JsonSchemaParser(decision_schema(), config=CharacterLevelParserConfig(**PARSER_OPTIONS))


def grammar_permits(text):
    parser = parser_factory()
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def backend_identity(protocol):
    if protocol.get('batch') == 'personal':
        return None
    return dict(model=protocol['model'][1], revision=protocol['model'][2],
                max_tokens=protocol['max_tokens'])


def check_backend(record, protocol):
    expected = backend_identity(protocol)
    if expected is not None and any((turn.get('backend_metadata') or {}).get('identity') != expected
                                    for turn in record['turns']):
        raise ValueError('actual backend identity differs from frozen model/revision/token budget')
