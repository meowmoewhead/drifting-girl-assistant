"""Bounded optional-reward cleanup; navigation hints never authorize inputs."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from drift.ui.button import CroppedTemplateButton

PREFIXES = ('driftinggirl_achievements_', 'driftinggirl_mailbox_', 'driftinggirl_event_potion_')
CLOSES = {
    'achievements': {'driftinggirl_achievements_close', 'driftinggirl_achievements_fold'},
    'mailbox': {'driftinggirl_mailbox_close', 'driftinggirl_mailbox_limit_close'},
    'event_potion': {'driftinggirl_event_potion_close'},
}
RETURNS = {
    'achievements': {'driftinggirl_achievements_return', 'driftinggirl_achievements_recover',
                     'driftinggirl_achievements_battle_recover', 'driftinggirl_achievements_parallel_entry_recover'},
    'mailbox': {'driftinggirl_mailbox_return', 'driftinggirl_mailbox_parallel_return',
                'driftinggirl_mailbox_recover', 'driftinggirl_mailbox_battle_recover',
                'driftinggirl_mailbox_parallel_battle_recover'},
    'event_potion': {'driftinggirl_event_potion_return'},
}


class PotionAdRecovery:
    """Cancel only the identified wrong-world prompt; never confirm a return."""
    ENTRY = 'driftinggirl_potion_ad_open'
    RETURN = 'driftinggirl_potion_ad_entry_recover'
    LIMIT = 3
    THRESHOLD = .96

    def __init__(self, root):
        self.attempts = 0
        self.active = self.cancelled = False
        self.deadline = None
        self.frames = 0
        self.key = self.since = self.last_frame = None
        self.button = None
        path = Path(root)/'assets/potion-ad/return-world-confirm-source.png'
        if path.is_file():
            with Image.open(path) as source:
                reference = np.asarray(source.convert('RGB'))
            title = (222,556,493,595)
            parts = ((195,597,524,626), (457,764,559,816))
            self.button = CroppedTemplateButton(reference[556:595,222:493],
                (202,526,513,625), [1], 'potion_ad_wrong_world_cancel', source_area=title,
                click_source_area=(480,775,540,805),
                required_templates=[(box, reference[box[1]:box[3],box[0]:box[2]]) for box in parts],
                color_condition=dict(source_area=[375,748,641,827],
                    hsv=[[0,100,100],[18,255,255]], minimum_fraction=.6))

    def visible(self, frame):
        return bool(self.button is not None and frame.shape == (1280,720,3)
                    and self.button.match(frame, similarity=self.THRESHOLD))

    def begin(self, deadline):
        if self.attempts >= self.LIMIT:
            raise RuntimeError('药水广告误入回城确认框已达本轮三次上限')
        self.attempts += 1
        self.active, self.cancelled, self.deadline = True, False, deadline
        self.key = self.since = self.last_frame = None
        self.frames = 0

    def status(self):
        return dict(attempts=self.attempts, limit=self.LIMIT, pending=self.active,
                    cancelled=self.cancelled, success=False, deadline=self.deadline)

    def observe_return(self, frame, matchers, now):
        item = matchers.get(self.RETURN)
        key = None
        if self.cancelled and item and not self.visible(frame):
            rule, button, offset = item
            if (rule.get('enabled') and rule.get('action') == 'observe'
                    and button.match(frame, offset=offset, similarity=rule['threshold'])):
                key = OptionalCleanup.context(frame)
        if key is None:
            self.key = self.since = self.last_frame = None
            self.frames = 0
            return False
        if frame is self.last_frame:
            return False
        self.last_frame = frame
        if key != self.key:
            self.key, self.since, self.frames = key, now, 1
        else:
            self.frames += 1
        return self.frames >= 3 and now-self.since >= 3


def apply_policy(rule):
    rule = dict(rule)
    # Explicit UI/custom configuration wins. Daily claim/bonus flags already
    # exist; the initial shop-open wait was the uncovered stopping boundary.
    safe = (rule.get('id') == 'driftinggirl_daily_open'
            or str(rule.get('id', '')).startswith(PREFIXES))
    # Closing an already open window has its own input budget and deadline.
    # Skipping a reward must not buy another attempt at a missed close/fold.
    safe = safe and rule.get('id') not in set().union(*CLOSES.values(), *RETURNS.values())
    if (safe and rule.get('next_stage') not in (None, 'main')
            and not any(rule.get(k) for k in ('purchase_condition', 'quantity_condition', 'confirmation_condition'))):
        rule.setdefault('skip_on_timeout', True)
    return rule


class OptionalCleanup:
    def __init__(self, stage, now, *, original_deadline=None, original_trigger=None):
        self.family = next((n for n in CLOSES if stage.startswith(n)), None)
        if self.family is None:
            raise ValueError('此步骤不支持安全跳过')
        self.stage = 'optional_cleanup'
        self.original_stage, self.original_deadline = stage, original_deadline
        self.original_trigger = original_trigger
        self.deadline = now+40
        self.key = self.since = self.last_frame = None
        self.frames = 0
        self.actions = None
        self.attempts = {}
        self.uncertain = False
        self.input_key = self.input_since = self.input_frame = None
        self.input_frames = 0

    def status(self):
        return dict(original_stage=self.original_stage, original_deadline=self.original_deadline,
                    original_trigger=self.original_trigger, deadline=self.deadline,
                    attempts=dict(self.attempts), uncertain=self.uncertain)

    def prepare(self, rules, load_rule):
        """Separate in-memory exit rules; saved normal reward guards stay intact."""
        self.actions = []
        for original, _, _ in rules:
            rid = original['id']
            if (rid not in CLOSES[self.family] or not original.get('enabled')
                    or any(original.get(k) for k in ('purchase_condition', 'quantity_condition',
                                                     'confirmation_condition'))):
                continue
            rule = deepcopy(original)
            alternatives = []
            if rid == 'driftinggirl_achievements_close':
                if (rule.get('asset') != 'assets/achievements-list-native.png'
                        or rule.get('source_crop') != [283,934,518,955]
                        or rule.get('click_source_area') != [630,225,710,294]):
                    continue
                # Explicit abandonment after timeout, not the normal "all claimed" rule.
                if rule.get('absent_rule') == 'driftinggirl_achievements_claim':
                    rule.pop('absent_rule')
                rule.setdefault('required_source_areas', []).append([643,235,695,279])
                rule.setdefault('color_condition', dict(source_area=[300,920,500,960],
                    hsv=[[0,0,160],[179,255,255]], minimum_fraction=.6))
            elif rid == 'driftinggirl_achievements_fold':
                if (rule.get('asset') != 'assets/achievements-menu-native.png'
                        or rule.get('click_source_area') != [650,550,690,587]):
                    continue
                rule.setdefault('required_source_areas', []).extend(
                    [[211,1140,266,1189],[542,1140,597,1183],[650,550,690,587]])
                rule.setdefault('color_condition', dict(source_area=[558,1147,585,1168],
                    hsv=[[0,0,210],[179,255,255]], minimum_fraction=.3))
            elif rid == 'driftinggirl_event_potion_close':
                if (rule.get('asset') != 'assets/event-potion/shop.png'
                        or rule.get('source_crop') != [557,521,597,564]
                        or rule.get('search_area') != [517,485,627,591]
                        or rule.get('click_source_area') != [583,390,650,446]
                        or not set(rule.get('asset_variants', [])).issubset({
                            'assets/event-potion/shop.png', 'assets/event-potion/shop-candy.png',
                            'assets/event-potion/shop-ghost.png'})):
                    continue
                if self.original_stage == 'event_potion_shop':
                    rule.update(source_crop=[143,621,206,694], search_area=[112,580,235,716])
                rule['threshold'] = max(.94, rule['threshold'])
                if rule.get('absent_rule') == 'driftinggirl_event_potion_confirm':
                    rule.pop('absent_rule')
                # The confirm modal covers the shop/help template. Recognize
                # its existing title/price/button, but replace the buy target
                # with the whitelisted close cross. No new click geometry.
                confirm = next((r for r, _, _ in rules
                                if r['id'] == 'driftinggirl_event_potion_confirm'), None)
                if (confirm and confirm.get('enabled')
                        and confirm.get('asset') == 'assets/event-potion/confirm.png'
                        and confirm.get('source_crop') == [267,574,513,608]
                        and confirm.get('search_area') == [162,531,552,650]
                        and confirm.get('click_source_area') == [95,761,321,821]
                        and set(confirm.get('asset_variants', [])).issubset({
                            'assets/event-potion/confirm.png', 'assets/event-potion/confirm-candy.png',
                            'assets/event-potion/confirm-ghost.png'})
                        and not any(confirm.get(k) for k in ('purchase_condition', 'quantity_condition',
                                                            'confirmation_condition'))):
                    cancel = deepcopy(confirm)
                    cancel.update(id=rid+'_confirm', click_source_area=list(rule['click_source_area']),
                                  threshold=max(.95, confirm['threshold']))
                    cancel.setdefault('required_source_areas', []).extend(
                        box for box in rule.get('required_source_areas', [])
                        if 583 <= box[0] < box[2] <= 650 and 390 <= box[1] < box[3] <= 446)
                    alternatives.append(cancel)
            else:
                # Fixed mail tab + button text + red close cross, independent of
                # whether a mail remains. Do not reuse claim_all as a graph exit.
                if (rid != 'driftinggirl_mailbox_close'
                        or rule.get('asset') != 'assets/mailbox-empty-native.png'
                        or rule.get('source_crop') != [298,570,408,596]
                        or rule.get('click_source_area') != [612,170,690,230]):
                    continue
                rule.update(source_crop=[144,228,197,253], search_area=[134,218,207,263])
                rule.setdefault('required_source_areas', []).extend(
                    [[488,987,582,1019],[615,175,685,225]])
                rule.setdefault('color_condition', dict(source_area=[400,550,520,590],
                    hsv=[[0,0,180],[179,255,255]], minimum_fraction=.7))
            for exit_rule in [rule, *alternatives]:
                exit_rule.update(id=exit_rule['id']+'_cleanup', name=original['name']+'（超时安全退出，不记完成）',
                                 stage=self.stage, next_stage=self.stage, action='click',
                                 stable_frames=max(3, exit_rule.get('stable_frames', 1)),
                                 next_delay=.8, cleanup_origin=rid)
                exit_rule.pop('recheck_before_click', None)
                exit_rule.pop('timeout_grace', None)
                exit_rule.pop('skip_on_timeout', None)
                self.actions.append(load_rule(exit_rule))
        return self.actions

    def rules(self, rules):
        actions = self.actions if self.actions is not None else [
            item for item in rules if item[0]['id'] in CLOSES[self.family]]
        # Return observations are handled only by observe_return: no single
        # normal stage transition may bypass its three-frame/three-second proof.
        return ([item for item in actions if self.allowed(item[0]['id'])]
                + [item for item in rules if item[0].get('interrupt')])

    def allowed(self, rid):
        return not self.uncertain and sum(self.attempts.values()) < 3 and self.attempts.get(rid, 0) < 2

    @staticmethod
    def context(frame):
        if frame.shape != (1280,720,3):
            return None
        rank = frame[10:62,72:126].min(axis=2) > 130
        world = frame[46:66,35:56].astype(np.int16)
        cyan = (world[:,:,1]>=100)&(world[:,:,1]>world[:,:,0]+15)&(world[:,:,2]>world[:,:,0]+15)
        return (rank.tobytes(), cyan.tobytes()) if int(rank.sum()) >= 50 else None

    def input_context(self, frame):
        # Event modals dim RANK. Their whitelisted templates, stable target
        # and fresh-frame rematch provide identity without the sidebar pixels.
        if self.family == 'event_potion':
            return (self.family,) if frame.shape == (1280,720,3) else None
        return self.context(frame)

    def ready(self, rid, button, frame, now):
        context = self.input_context(frame)
        key = (rid, tuple(button.button), context) if context is not None else None
        if key is None or now >= self.deadline or not self.allowed(rid):
            self.reset_input_proof()
            return False
        if frame is self.input_frame:
            return False
        self.input_frame = frame
        if key != self.input_key:
            self.input_key, self.input_since, self.input_frames = key, now, 1
        else:
            self.input_frames += 1
        return self.input_frames >= 3 and now-self.input_since >= 3

    def reset_input_proof(self):
        self.input_key = self.input_since = self.input_frame = None
        self.input_frames = 0

    def issued(self, rid):
        self.attempts[rid] = self.attempts.get(rid, 0)+1
        self.reset_input_proof()

    def observe_return(self, frame, matchers, now):
        if now >= self.deadline or frame.shape != (1280,720,3):
            return False
        # Footer templates can also be visible below the expanded sidebar.
        # That menu still needs its fold; never call it a completed return.
        event_potion = self.family == 'event_potion'
        blockers = ({'driftinggirl_achievements_resume', 'driftinggirl_achievements_ready',
                     'driftinggirl_achievements_close'}
                    if self.family == 'achievements' else CLOSES['mailbox'])
        if event_potion:
            blockers = {'driftinggirl_event_potion_'+name for name in ('select', 'confirm', 'close')}
            if not blockers.issubset(matchers):
                self.key = self.since = self.last_frame = None; self.frames = 0
                return False
        blockers |= {r['id'] for r, _, _ in self.actions or []}
        for name in blockers:
            item = matchers.get(name)
            if item:
                r, b, o = item
                if (event_potion or r.get('enabled')) and b.match(frame, offset=o, similarity=r['threshold']):
                    self.key = self.since = self.last_frame = None; self.frames = 0
                    return False
        if not event_potion and float((frame[1147:1168,558:585].min(axis=2)>=140).mean()) < .55:
            self.key = self.since = self.last_frame = None; self.frames = 0
            return False
        matched = None
        observations = {'driftinggirl_event_potion_open'} if event_potion else RETURNS[self.family]
        for name in sorted(observations):
            item = matchers.get(name)
            if item:
                r, b, o = item
                if (r.get('enabled') and (event_potion or r.get('action') == 'observe' and r.get('next_stage') == 'main')
                        and b.match(frame, offset=o, similarity=r['threshold'])):
                    absent = matchers.get(r.get('absent_rule'))
                    if absent and absent[1].match(frame, offset=absent[2], similarity=absent[0]['threshold']):
                        continue
                    matched = name
                    break
        key = None
        if matched and event_potion:
            key = matched
        elif matched:
            rank = frame[10:62,72:126].min(axis=2) > 130
            world = frame[46:66,35:56].astype(np.int16)
            cyan = (world[:,:,1]>=100)&(world[:,:,1]>world[:,:,0]+15)&(world[:,:,2]>world[:,:,0]+15)
            if int(rank.sum()) >= 50:
                key = matched, rank.tobytes(), cyan.tobytes()
        if key is None:
            self.key = self.since = self.last_frame = None
            self.frames = 0
            return False
        if frame is self.last_frame:
            return False
        self.last_frame = frame
        if key != self.key:
            self.key, self.since, self.frames = key, now, 1
        else:
            self.frames += 1
        return self.frames >= 3 and now-self.since >= 3
