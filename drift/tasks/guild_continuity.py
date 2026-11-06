"""Departure guard for aligned legacy guild controllers.

The deployed controller may come from an existing packaged module. Keep its
task/teleport/return logic and add only the missing departure precondition.
"""
from drift.core.continuity import reconcile_late_departure


def install(cls):
    if getattr(cls,'_departure_guard_installed',False):return
    previous=cls.step

    def step(self,frame,*,due,period):
        if self.active and not self.allowed():
            return previous(self,frame,due=due,period=period)
        check=(not self.active and due and not self.collection_only
               or self.active and self.phase=='stop' and self.journal.get('origin') is None)
        if not check:return previous(self,frame,due=due,period=period)
        reader=self.origin_screen
        origin=reader(frame)
        if not self.active and origin is None:
            self._confirmed(None,frame)
            # Keep the controller's independent read-only return reconciliation.
            return previous(self,frame,due=False,period=period)
        if self.active and origin is not None:
            reconcile_late_departure(self.journal,origin,valid=self.valid_farm_origin)
        overridden='origin_screen' in self.__dict__
        # Reuse only this same frame's read, never a later input-guard screenshot.
        self.origin_screen=lambda fresh:origin if fresh is frame else reader(fresh)
        try:return previous(self,frame,due=due,period=period)
        finally:
            if overridden:self.origin_screen=reader
            else:del self.origin_screen

    cls.step=step
    cls._departure_guard_installed=True
