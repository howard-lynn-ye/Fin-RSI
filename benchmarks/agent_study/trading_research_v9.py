"""Research JSON transport: constrain format and tool names, never investments.

The model may choose any available reader, library tool, Python, submit or hold.
Arguments stay model-authored and are validated by the existing controller.
"""
import hashlib
import json
from benchmarks.agent_study import trading_schema_v8 as final
from benchmarks.agent_study.trading_submission_v8 import unique_keys

# Do not inherit LMFE's small default array cap for free-form tool arguments.
# This declared cap exceeds practical arrays within the 1024-token response budget.
PARSER_OPTIONS = dict(final.PARSER_OPTIONS, max_json_array_length=4096)


def call_schema(names):
    return dict(type='object', properties={
        'tool': {'type': 'string', 'enum': list(names)},
        # Omitting additionalProperties means any JSON value. The pinned LMFE
        # version mishandles literal True as a schema object; its default uses
        # the intended recursive ANY_JSON_OBJECT_SCHEMA instead.
        'arguments': {'type': 'object'}},
        required=['tool', 'arguments'], additionalProperties=False)


def output_contract(arm, interface='v9', structured=True):
    if not structured:
        return {'transport': 'text'}
    if arm not in ('raw', 'library') or interface not in ('v8', 'v9'):
        raise ValueError('undeclared research interface')
    from benchmarks.agent_study.trading_runtime_v8 import Tools
    if interface == 'v9':
        from benchmarks.agent_study.trading_runtime_v9 import Tools
    names = sorted(set(Tools.COMMON + (Tools.LIBRARY if arm == 'library' else ()) +
                       ('run_python', 'submit', 'hold')))
    digest = hashlib.sha256(json.dumps(call_schema(names), sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    return dict(transport=final.TRANSPORT, version=final.DECODER_VERSION,
        stage='research_only', revision='single-json-call-v1', arm=arm, interface=interface,
        tool_names=names, schema_sha256=digest, **PARSER_OPTIONS)


def protocol_contract(interface, batch):
    structured = batch not in ('personal', 'matrix_native', 'matrix_reasoning')
    return {arm: output_contract(arm, interface, structured) for arm in ('raw', 'library')}


def arm_contract(protocol, arm):
    # Small unit fixtures may omit the field. Real frozen runs are checked by
    # study.verify against protocol_contract before inference or accounting.
    return protocol.get('research_output', {}).get(arm, {'transport': 'text'})


def check_contract(contract):
    if contract == {'transport': 'text'}:
        return
    if not isinstance(contract, dict) or contract != output_contract(
            contract.get('arm'), contract.get('interface')):
        raise ValueError('unqualified research output contract')


def parser_factory(contract):
    check_contract(contract)
    if contract['transport'] != final.TRANSPORT:
        raise ValueError('structured research contract required')
    from importlib.metadata import version
    if version('lm-format-enforcer') != final.DECODER_VERSION:
        raise ValueError('unqualified research decoder version')
    from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
    return JsonSchemaParser(call_schema(contract['tool_names']),
        config=CharacterLevelParserConfig(**PARSER_OPTIONS))


def prefix_filter(tokenizer_data, prompt, contract):
    if prompt.rstrip().endswith('<think>'):
        raise ValueError('reasoning template is not qualified for structured research')
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser_factory(contract))


def grammar_permits(text, contract):
    parser = parser_factory(contract)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def check_turn(turn, contract):
    check_contract(contract)
    metadata = turn.get('backend_metadata') or {}
    if metadata.get('research_output') != contract or metadata.get('final_output') is not None:
        raise ValueError('structured research receipt absent or changed')
    if turn['finish_reason'] == 'length':
        return None  # The runtime rejects length before executing any response.
    def reject_constant(value):
        raise ValueError('non-JSON constant: ' + value)
    try:
        value = json.loads(turn['response'], object_pairs_hook=unique_keys,
                           parse_constant=reject_constant)
    except (ValueError, RecursionError):
        if grammar_permits(turn['response'], contract):
            return 'structured-json-rejected'
        raise
    if (not isinstance(value, dict) or set(value) != {'tool', 'arguments'} or
            not isinstance(value['tool'], str) or value['tool'] not in contract['tool_names'] or
            not isinstance(value['arguments'], dict)):
        raise ValueError('structured research envelope or tool scope changed')
    return None
