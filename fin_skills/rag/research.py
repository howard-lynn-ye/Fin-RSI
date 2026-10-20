"""One query for reference knowledge, dated evidence and exact tool contracts."""
from __future__ import annotations

import json

from .documents import Document, documents_from_skills, positive
from .index import RAGIndex
from .pipeline import RAGPipeline


def compact(prepared):
    return {k: prepared[k] for k in ('context', 'citations', 'no_evidence')}


def research_context(query, documents=None, skills=None, as_of=None, top_k=3,
                     max_context_chars=4000, tool_k=2):
    """Retrieve without choosing investments or executing the returned tools.

    Reference knowledge/contracts describe the installed library, not historical
    information. Only caller-supplied evidence is filtered by ``as_of``. Documents
    need actual availability timestamps for a historical query.
    """
    positive(top_k, 'top_k')
    positive(tool_k, 'tool_k')
    if top_k > 10 or tool_k > 5:
        raise ValueError('top_k must be 1..10 and tool_k 1..5')
    if type(max_context_chars) is not int or not 800 <= max_context_chars <= 12000:
        raise ValueError('max_context_chars must be 800..12000')
    if documents is not None and not isinstance(documents, list):
        raise TypeError('documents must be a list')
    if skills is not None and not isinstance(skills, list):
        raise TypeError('skills must be a list')
    supplied = documents or []
    evidence_budget = max_context_chars // 2 if supplied else 0
    knowledge_budget = max_context_chars - evidence_budget
    knowledge = RAGPipeline(RAGIndex.from_documents(
        documents_from_skills(skills), chunk_size=600, overlap=100)).prepare(
            query, top_k=top_k, max_context_chars=knowledge_budget)
    evidence = RAGPipeline(RAGIndex.from_documents(
        supplied, chunk_size=600, overlap=100)).prepare(
            query, top_k=top_k, max_context_chars=max(1, evidence_budget), as_of=as_of)

    # Import here: tools.runner registers this function while loading the registry.
    from fin_skills.tools import list_tools
    specs = {s['name']: s for s in list_tools()}
    cards = [Document(id=name, source='fin-skills:tool/' + name,
                      text=name + '\n' + spec['description']) for name, spec in specs.items()]
    ranked = RAGIndex.from_documents(cards).search(query, top_k=tool_k)
    contracts, seen = [], set()
    for hit in ranked:
        name = hit['document_id']
        if name in seen:
            continue
        seen.add(name)
        spec = specs[name]
        contracts.append({'name': name, 'description': spec['description'],
                          'input_schema': spec['input_schema'],
                          'call_template': {'tool': name, 'arguments': {
                              key: '<supply ' + key + ' according to input_schema>'
                              for key in spec['input_schema'].get('required', [])}},
                          'template_requires_values': True})
    return dict(query=query, knowledge=compact(knowledge), evidence=compact(evidence),
                tool_contracts=contracts, content_is_untrusted=True,
                reference_policy='Current installed library knowledge/contracts; not a historical corpus.',
                evidence_as_of=as_of, citation_scope='Each section has its own citation labels.',
                method='bm25', note='Relevance is not a trading recommendation. No tools were executed.')


def documents_from_events(events):
    """Index the observed revision, never backdate it to a transaction or filing date."""
    from fin_skills.collect.model import Event
    from .documents import timestamp
    fields = tuple(Event.__dataclass_fields__)
    documents = []
    for row in events:
        event = Event(**{k: row[k] for k in fields if k in row})
        observed = timestamp(event.observed_at, 'observed_at')
        published = (timestamp(event.published_at, 'published_at')
                     if event.published_at is not None else observed)
        metadata = dict(watch_id=row.get('watch_id'), event_id=event.id,
                        sequence=row.get('seq'), kind=event.kind,
                        published_at=event.published_at, observed_at=event.observed_at,
                        availability_basis='max(observed_at, published_at); observation if publication unknown',
                        serialization='title followed by complete source event encoded as JSON')
        identity = json.dumps([row.get('watch_id'), event.id, row.get('seq')], ensure_ascii=True)
        documents.append(Document(id=identity, source=event.url or event.source,
            text=event.title + '\n' + json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True),
            available_at=max(observed, published).isoformat(), metadata=metadata))
    return documents
