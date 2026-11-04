"""Abandon only a product selection that visibly left the guild list unchanged.

No selection is repeated, and cleanup must finish within the original selection
deadline. An opened quantity dialog, changed stock, or uncertain frame closes
this recovery path.
"""
import numpy as np

SELECTORS = {
    'driftinggirl_guild_potion_select': 'guild_potion_quantity',
    'driftinggirl_guild_material_select': 'guild_material_quantity',
}


class GuildReturnGuard:
    """Read-only native-fishing return when combat dims the shop sidebar icon."""
    def __init__(self):
        self.key, self.frames, self.since = None, 0, None

    def observe(self, frame, state, probe, *, now, stopped=False, interrupted=False):
        close = state.matchers.get('driftinggirl_guild_close')
        returning = state.matchers.get('driftinggirl_guild_return')
        context = None
        if (not stopped and not interrupted and not state.handling_interrupt
                and close is not None and returning is not None
                and close[0].get('enabled') and returning[0].get('enabled')
                and returning[0].get('action') == 'observe'
                and returning[0].get('next_stage') == 'main'
                and state.stage == 'guild_return' and state.stage_trigger == close[0]['name']
                and state.stage_deadline is not None
                and state.stage_ready_at <= now < state.stage_deadline
                and frame.shape == (1280,720,3)
                and not close[1].match(frame,offset=close[2],similarity=close[0]['threshold'])
                and not returning[1].match(frame,offset=returning[2],similarity=returning[0]['threshold'])):
            observer = probe()
            if not observer.defense.encounter(frame,39):
                actual = observer.native_entry_recovery_context(frame)
                if actual is not None:
                    context = state.stage_trigger, actual
        if context is None:
            self.key, self.frames, self.since = None, 0, None
            return False
        if context != self.key:
            self.key, self.frames, self.since = context, 0, now
        self.frames += 1
        return self.frames >= max(3,returning[0]['stable_frames']) and now-self.since >= 3


class GuildSelectionGuard:
    def __init__(self):
        self.clear()

    def clear(self, reason='idle'):
        self.rule_id = self.origin = self.context_key = None
        self.frames = 0
        self.since = None
        self.reason = reason

    def status(self):
        return dict(rule_id=self.rule_id, reason=self.reason, frames=self.frames,
                    stock=list(self.origin[0]) if self.origin is not None else None)

    @staticmethod
    def list_context(frame, matchers):
        """Identify only the bright guild list, never its quantity/confirm modal."""
        if frame.shape != (1280, 720, 3):
            return None
        entry = matchers.get('driftinggirl_guild_close')
        if entry is None or not entry[0].get('enabled'):
            return None
        rule, button, offset = entry
        if not button.match(frame, offset=offset, similarity=rule['threshold']):
            return None
        title = frame[337:374, 175:328]
        if float(((title.max(axis=2) >= 180) & (title.min(axis=2) >= 100)).mean()) < .5:
            return None
        rank = frame[10:62, 72:126].min(axis=2) > 130
        if int(rank.sum()) < 50:
            return None
        stage = frame[46:66, 35:56].astype(np.int16)
        world = ((stage[:, :, 1] >= 100) & (stage[:, :, 1] > stage[:, :, 0] + 15)
                 & (stage[:, :, 2] > stage[:, :, 0] + 15))
        return rank.tobytes(), world.tobytes(), tuple(button.button)

    @staticmethod
    def context(frame, matchers, rule_id, purchase_check):
        if rule_id not in SELECTORS or frame.shape != (1280, 720, 3):
            return None
        entries = [matchers.get(key) for key in
                   (rule_id, 'driftinggirl_guild_close')]
        if any(entry is None or not entry[0].get('enabled') for entry in entries):
            return None
        for rule, button, offset in entries:
            if not button.match(frame, offset=offset, similarity=rule['threshold']):
                return None
        rule, button, _ = entries[0]
        passed, stock = purchase_check(rule, button, frame)
        if not passed:
            return None
        page = GuildSelectionGuard.list_context(frame, matchers)
        return (tuple(stock), *page) if page is not None else None

    def arm(self, rule_id, context):
        self.clear()
        if rule_id in SELECTORS and context is not None:
            self.rule_id, self.origin = rule_id, context
            self.reason = 'armed'

    def observe(self, frame, matchers, purchase_check, *, stage, trigger,
                started, deadline, now, stopped=False):
        if self.rule_id is None:
            return False
        if (stopped or deadline is None or now >= deadline
                or stage != SELECTORS[self.rule_id]
                or trigger != matchers[self.rule_id][0]['name']):
            self.clear('inactive_or_context_ended')
            return False
        prefix = self.rule_id.removesuffix('_select')
        for suffix in ('_plus', '_buy'):
            entry = matchers.get(prefix + suffix)
            if entry and entry[1].match(frame, offset=entry[2], similarity=entry[0]['threshold']):
                self.clear('quantity_dialog_seen')
                return False
        context = self.context(frame, matchers, self.rule_id, purchase_check)
        if context is None or context != self.origin:
            self.frames, self.since, self.context_key = 0, None, None
            self.reason = 'list_or_stock_changed'
            return False
        if self.context_key != context:
            self.frames, self.since, self.context_key = 0, now, context
        self.frames += 1
        ready = self.frames >= 3 and now - self.since >= 3 and now - started >= 5
        self.reason = 'unchanged_list_confirmed' if ready else 'confirming_unchanged_list'
        return ready
