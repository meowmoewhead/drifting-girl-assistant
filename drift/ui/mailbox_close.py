"""One bounded retry for an unchanged, fully identified mailbox close page."""

CLOSES = ('driftinggirl_mailbox_close', 'driftinggirl_mailbox_limit_close')


class MailboxCloseGuard:
    def __init__(self):
        self.origin = None
        self.attempts = 0
        self.frames, self.since = 0, None

    @staticmethod
    def page_visible(frame, matchers):
        """An obscured cross is not evidence of return: retain text-only absence."""
        if frame.shape != (1280, 720, 3):
            return True
        for key in CLOSES:
            entry = matchers.get(key)
            if entry is None or not entry[0].get('enabled'):
                continue
            rule, button, _ = entry
            if not getattr(button, 'mailbox_close_guarded', False):
                continue
            for primary, offset, _ in button.variants:
                if primary.match(frame, offset=offset, similarity=rule['threshold']):
                    return True
        return False

    @staticmethod
    def context(frame, matchers, rule_id=None):
        if frame.shape != (1280, 720, 3):
            return None
        for key in (CLOSES if rule_id is None else (rule_id,)):
            entry = matchers.get(key)
            if entry is None or not entry[0].get('enabled'):
                continue
            rule, button, offset = entry
            if (getattr(button, 'mailbox_close_guarded', False)
                    and button.match(frame, offset=offset, similarity=rule['threshold'])):
                # Natural battle progress beneath the modal cannot authorize
                # a click. Its own text, tabs, claim button and cross must match.
                return key, tuple(button.button)
        return None

    def issued(self, context, *, retry):
        if retry:
            if context != self.origin or self.attempts != 1:
                raise RuntimeError('邮箱关闭重试上下文或次数无效')
            self.attempts = 2
        else:
            self.origin, self.attempts = context, 1
        self.frames, self.since = 0, None

    def ready(self, frame, state, *, now, stopped=False, interrupted=False):
        entry = state.matchers.get(self.origin[0]) if self.origin is not None else None
        valid = (not stopped and not interrupted and not state.handling_interrupt
                 and entry is not None and self.attempts == 1
                 and state.stage == 'mailbox_return'
                 and state.stage_trigger == entry[0]['name']
                 and state.stage_deadline is not None
                 and state.stage_ready_at <= now < state.stage_deadline
                 and self.context(frame, state.matchers, self.origin[0]) == self.origin)
        if not valid:
            self.frames, self.since = 0, None
            return False
        if self.since is None:
            self.since = now
        self.frames += 1
        return self.frames >= 3 and now - self.since >= 3
