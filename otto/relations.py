"""Complementary directed co-visitation and inspectable ranking features.

All timestamps are milliseconds. Fit only on events available before a query
window; never use a query's suffix to build its graph or features.
"""
from collections import Counter, defaultdict

RELATIONS = ('adjacent', 'forward_buy', 'buy2buy')
SOURCES = ('clicks', 'carts', 'orders') + RELATIONS
FEATURES = [
    'base_rank', 'seen', 'history_weight', 'click_count', 'cart_count',
    'order_count', 'last_position', 'last_age_hours', 'first_age_hours',
    'session_events', 'session_unique', 'session_hours', 'item_pop_rank',
] + [f'{source}_{stat}' for source in SOURCES for stat in ('score', 'rank', 'last_score')]


class Relations:
    def __init__(self, neighbors=40):
        self.neighbors = neighbors
        self.graph = {k: defaultdict(Counter) for k in RELATIONS}

    def add(self, events):
        events = events[-30:]
        pairs = {k: {} for k in RELATIONS}
        for i, a in enumerate(events):
            for j in range(i + 1, len(events)):
                b = events[j]
                delta = b['ts'] - a['ts']
                if a['aid'] == b['aid'] or not 0 < delta <= 86400000:
                    continue
                edge = (a['aid'], b['aid'])
                decay = 0.5 ** (delta / 3600000)
                if j == i + 1:
                    pairs['adjacent'][edge] = max(pairs['adjacent'].get(edge, 0), decay)
                if b['type'] in ('carts', 'orders'):
                    if delta <= 7200000:
                        weight = decay * (2 if b['type'] == 'orders' else 1)
                        pairs['forward_buy'][edge] = max(pairs['forward_buy'].get(edge, 0), weight)
                    if a['type'] in ('carts', 'orders'):
                        pairs['buy2buy'][edge] = max(pairs['buy2buy'].get(edge, 0), decay)
        for kind, values in pairs.items():
            for (a, b), score in values.items():
                self.graph[kind][a][b] += score

    def finish(self):
        self.graph = {k: {a: sorted(v.items(), key=lambda x: (-x[1], x[0]))[:self.neighbors]
                          for a, v in graph.items()} for k, graph in self.graph.items()}


class CandidateFeatures:
    def __init__(self, base, relations):
        self.base = base
        self.base.policy = 'pooled'
        self.relations = relations
        self.pop_ranks = {k: {a: i + 1 for i, a in enumerate(v)} for k, v in base.pop.items()}

    def evidence(self, events):
        anchors = list(dict.fromkeys(e['aid'] for e in reversed(events)))[:20]
        graphs = dict(self.base.graph, **self.relations.graph)
        scores, ranks, last = {}, {}, {}
        for source in SOURCES:
            score = Counter(); last_score = {}
            for pos, aid in enumerate(anchors):
                neighbors = graphs[source].get(aid, [])
                norm = max((s for _, s in neighbors), default=1)
                for target, strength in neighbors:
                    value = strength / norm
                    score[target] += value / (1 + pos)
                    if pos == 0:
                        last_score[target] = value
            scores[source] = score
            ranks[source] = {a: i + 1 for i, a in enumerate(sorted(score, key=lambda a: (-score[a], a)))}
            last[source] = last_score
        return scores, ranks, last

    def make(self, events, kind, evidence=None, extra=True):
        scores, ranks, last = evidence or self.evidence(events)
        baseline = self.base.predict(events, kind, budget=200)
        candidates = baseline[:100]
        if extra:
            # 100 established candidates + up to 30 from each independent source.
            for source in RELATIONS:
                candidates += list(ranks[source])[:30]
        candidates = list(dict.fromkeys(candidates + baseline))[:200]
        base_ranks = {a: i + 1 for i, a in enumerate(baseline)}
        history = {}
        now = events[-1]['ts']
        for pos, e in enumerate(reversed(events)):
            info = history.setdefault(e['aid'], [0., 0, 0, 0, pos, (now-e['ts'])/3600000, 0.])
            info[0] += {'clicks': 1, 'carts': 6, 'orders': 3}[e['type']] * 0.95**pos
            info[1 + ('clicks', 'carts', 'orders').index(e['type'])] += 1
            info[6] = (now-e['ts'])/3600000
        rows = []
        for aid in candidates:
            h = history.get(aid, [0., 0, 0, 0, 1000, 1000., 1000.])
            row = [base_ranks.get(aid, 1000), int(aid in history), *h,
                   len(events), len(history), (now-events[0]['ts'])/3600000,
                   self.pop_ranks[kind].get(aid, 1000)]
            for source in SOURCES:
                row += [scores[source].get(aid, 0), ranks[source].get(aid, 1000), last[source].get(aid, 0)]
            assert len(row) == len(FEATURES)
            rows.append(row)
        return candidates, rows, baseline
