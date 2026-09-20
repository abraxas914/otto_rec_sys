"""OTTO adaptation of PCTM log-pooling (Petrov et al., RecSys 2026).

Independent implementation of arXiv:2608.19833 Algorithm 2. Adaptations:
three target behavior types, bounded last-30 history, strict positive time
within a day, distance cap 5, row pruning 80, and repeats remain eligible.
This is not a reproduction of the paper's datasets or reported scores.
"""
from collections import Counter, defaultdict
import math
from otto.evaluate import WEIGHTS

PCTM_FEATURES = [f'pctm_{k}_{s}' for k in WEIGHTS for s in ('score', 'rank')]
PCTM_FEATURES += ['log_item_frequency', 'log_target_frequency']


class TypedPCTM:
    def __init__(self, distance=5, decay=.7, history_decay=.7, tau=15000., neighbors=80):
        if tau <= 0 or distance < 1 or not 0 < history_decay <= 1:
            raise ValueError('Invalid PCTM configuration')
        self.distance=distance;self.decay=decay;self.history_decay=history_decay
        self.tau=tau;self.neighbors=neighbors
        self.graph={k:defaultdict(Counter) for k in WEIGHTS}
        self.frequency=Counter();self.typed_frequency={k:Counter() for k in WEIGHTS}
        self.max_fit_ts=-1

    def add(self, events):
        if not events:return
        self.max_fit_ts=max(self.max_fit_ts,max(e['ts'] for e in events))
        for e in events:
            self.frequency[e['aid']]+=1;self.typed_frequency[e['type']][e['aid']]+=1
        events=events[-30:]
        for i,a in enumerate(events):
            for j in range(i+1,min(len(events),i+self.distance+1)):
                b=events[j];dt=b['ts']-a['ts']
                if 0 < dt <= 86400000:
                    # Repeated item transitions are valid in OTTO, unlike seen-filtered benchmarks.
                    self.graph[b['type']][a['aid']][b['aid']]+=self.decay**(j-i-1)

    def finish(self):
        self.catalog_size=len(self.frequency)
        self.graph={k:{a:sorted(c.items(),key=lambda x:(-x[1],x[0]))[:self.neighbors]
                       for a,c in g.items()} for k,g in self.graph.items()}

    def evidence(self, events):
        history=list(reversed(events[-30:]))
        weights=[self.history_decay**i for i in range(len(history))]
        norm=sum(weights) or 1.
        evidence={}
        for kind in WEIGHTS:
            scores=Counter()
            for e,w in zip(history,weights):
                for aid,count in self.graph[kind].get(e['aid'],[]):
                    scores[aid]+=w/norm*math.log1p(self.catalog_size*count/self.tau)
            ranks={aid:i+1 for i,aid in enumerate(sorted(scores,key=lambda a:(-scores[a],a)))}
            evidence[kind]=(scores,ranks)
        return evidence

    def features(self, aids, kind, evidence):
        rows=[]
        for aid in aids:
            row=[]
            for source in WEIGHTS:
                scores,ranks=evidence[source]
                row.extend([scores.get(aid,0.),ranks.get(aid,10000)])
            row.extend([math.log1p(self.frequency[aid]),math.log1p(self.typed_frequency[kind][aid])])
            rows.append(row)
        return rows
