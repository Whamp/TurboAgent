# Turbo Agent

An LLM proxy that turns one client request into several candidate completions and selects one to return.

## Language

**Client request**:
The inbound Anthropic Messages or OpenAI chat-completions call. The proxy ignores the requested model id for routing and always runs the configured backend models.

**Backend model**:
A model listed under `backend.models` in `turbo-agent.yaml`. The actual completion provider for candidates.

**Candidate**:
One completion produced for a single client request. Several candidates may come from the same backend model (`num_candidates`) or from different backend models.

**Model execution**:
Producing one candidate with one backend model. It determines how that backend model is reached, but not how many candidates run or which candidate wins.

**Concurrent inference**:
Producing the candidate set for one client request, in parallel.

**Context refinement**:
An optional rewrite of the conversation before concurrent inference. It can change the messages sent to backend models. It does not choose among candidates.

**Verification**:
Selection of one candidate as the response. That includes the majority-vote shortcut, dropping empty completions, the pivot tournament, and fallback when the judge fails. It does not include progress scoring.

**Judge**:
The LLM that scores directed pairs during the pivot tournament.

**Majority voting**:
A verification shortcut: if more than half the candidates are identical, the tournament does not run and that completion wins.

**Pivot tournament**:
The comparison procedure verification uses when majority voting does not apply. The judge scores directed pairs. The winner is the selected candidate.
_Avoid_: round-robin, pairwise (legacy visualizer field)

**Fallback**:
Verification when the judge fails. The first valid candidate is selected. The request trace records that the tournament did not run.

**Progress monitor**:
A post-hoc score of the already-selected response. Observability only. It never changes which candidate the client receives.
_Avoid_: verification, judge (for this score)

**Request trace**:
The per-request record written for the visualizer: original request, candidates, verification, optional progress, and the response returned to the client. A verification winner is identified by its position in the original candidate list, empty completions included.
_Avoid_: reflection (removed; the field still exists in old logs as a stub)
