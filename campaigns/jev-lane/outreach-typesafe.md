# Outreach — TypeSafe (DRAFT, hold until first findings; send before ANY publication)

Context: guard-lane publishes nothing about Jev; preview ToS §1(v) requires
their OK for publishing benchmarks/performance info about the Interfaces.
They invite failure-mode reports (jaggedness page: "Found a failure mode that
belongs on this list? ... Reach us on Discord"; launch post: "where Jev works,
and where it falls short").

Status: NOT SENT. Decide send-timing: (a) now as a heads-up, or (b) after the
first bank readout + one robustness cell (better opener — findings in hand).

---

**Subject:** Independent guardrail robustness characterization — heads-up + a publication question

Hi TypeSafe team,

I'm Pepijn Frenken, a student researcher (Eindhoven). I've been happily
using Jev in a supply-chain replication (I rebuilt an LLM beer-game agent as
a decision-native one — your calibrated probabilities made the confidence
gate actually work), and it's now up for a second role in a different project:
an audit campaign on abliterated open-weight safety guards, where I need a
closed-model comparison arm.

Concretely, I'm reading harmful/benign prompt–response pairs (public eval
banks) with Jev in the guardrail role, and running sensitivity checks on how
the input state is framed — your jaggedness page notes state isn't treated as
hostile by default, and I'd like to measure, defensively, how far that goes
(think guard-robustness measurement, not evasion).

Two questions:

1. Is this a fine use of my eval account? (Normal API use, small volume, my key.)
2. If the results are interesting, how would you like to handle this — and is
   there anything you'd prefer I not publish? I'm asking before writing
   anything up, not after.

Either way, I'll send you the findings first. Happy to keep it completely
private if that's your preference.

Thanks — and congratulations on the launch.

Pepijn Frenken
[email] · [GitHub]

---

Notes for us:
- Keep every number out of this email until they reply (no details about the
  bank contents or the specific probe cells — the ask is about permission,
  not a findings dump).
- If they green-light publication, mirror guard-lane's defensive framing and
  credit/coordinate the release timing with them.
- Their Discord (from the jaggedness page) is the faster channel if email
  stalls; the DevRel hire (red-teaming programs) suggests they will route
  this positively.
