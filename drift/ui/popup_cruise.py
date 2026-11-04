"""Read-only return from an identified task notice to native cruise settings."""
from drift.ui.popup_raid import PopupRaidReturn


class PopupCruiseReturn:
    def __init__(self):
        self.clear()

    def clear(self):
        self.origin=self.key=self.since=self.last_seen=self.frame=None
        self.frames=0

    def arm(self, st, frame, setup, *, stopped=False):
        self.clear()
        view=getattr(setup,'view',None)
        if (stopped or st.handling_interrupt or st.stage!='main'
                or not callable(getattr(view,'observe_setup',None))
                or not callable(getattr(view,'values',None))):
            return False
        self.origin=PopupRaidReturn.notice_identity(frame)
        return self.origin is not None

    def observe(self, st, frame, setup, *, notice_gone, now, stopped=False):
        if (self.origin is None or stopped or not st.handling_interrupt
                or st.stage!='task_popup_return' or st.stage_deadline is None
                or now>=st.stage_deadline):
            self.clear();return False
        if frame is self.frame:
            return False
        self.frame=frame
        view=getattr(setup,'view',None)
        if (not notice_gone or now<st.stage_ready_at
                or PopupRaidReturn.notice_identity(frame)!=self.origin or view is None):
            self.key=self.since=self.last_seen=None;self.frames=0;return False
        context=view.observe_setup(frame)
        values=view.values(frame) if context and context['menu'] else None
        if (not context or not context['menu'] or values is None
                or not view.owner._off_icon(frame)):
            self.key=self.since=self.last_seen=None;self.frames=0;return False
        key=(context['scene'],context['signature'],str(values))
        if key!=self.key or self.last_seen is None or now-self.last_seen>8:
            self.key,self.since,self.frames=key,now,1
        else:
            self.frames+=1
        self.last_seen=now
        return self.frames>=3 and now-self.since>=3
