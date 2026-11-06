"""Task result policy; never renews an existing attempt or input deadline."""
from dataclasses import dataclass
from enum import Enum


class Outcome(str, Enum):
    COMPLETE='complete'
    RETRY='retry'
    SKIPPED='skipped'
    BLOCKED='blocked'


@dataclass(frozen=True)
class TaskOutcome:
    kind: Outcome
    detail: str = ''
    retry_seconds: float | None = None
    checked: bool = False

    def delay(self, interval):
        if self.retry_seconds is not None:return max(0.,float(self.retry_seconds))
        if self.kind is Outcome.COMPLETE or self.kind is Outcome.SKIPPED and self.checked:
            return max(0.,float(interval))
        return min(1800.,max(60.,float(interval)))


def result(success=True, *, retry_seconds=None, outcome=None, detail=''):
    kind=Outcome(outcome) if outcome is not None else (Outcome.COMPLETE if success else Outcome.RETRY)
    return TaskOutcome(kind,detail,retry_seconds,bool(success))


def controller_result(controller,success,*,retry_seconds=None,detail=''):
    """Classify old controller results without claiming an absent action ran."""
    journal=getattr(controller,'journal',{}) or {}
    if not isinstance(journal,dict):journal={}
    if success:
        counts=[journal[k] for k in ('upgraded','submitted') if type(journal.get(k)) is int]
        if not counts and type(getattr(controller,'holds',None)) is int:counts=[controller.holds]
        if counts and not any(counts):
            return TaskOutcome(Outcome.SKIPPED,detail or '已检查，本轮无可执行项',retry_seconds,True)
        return TaskOutcome(Outcome.COMPLETE,detail or '已确认完成',retry_seconds,True)
    last=getattr(controller,'last_result',None) or {}
    if not isinstance(last,dict):last={}
    message=detail or journal.get('failure') or getattr(controller,'hold',None) or last.get('reason') or '本轮未确认完成'
    blocked=bool(journal.get('retirement') or journal.get('phase')=='blocked'
                 or getattr(controller,'recovery_required',False)
                 or any(word in str(message) for word in ('未知','无法识别','到期','原有时限','用完','超过','保留')))
    return TaskOutcome(Outcome.BLOCKED if blocked else Outcome.RETRY,str(message),retry_seconds,False)
