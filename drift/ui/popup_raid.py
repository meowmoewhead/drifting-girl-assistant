"""Read-only return from a task notice to the RAID wait it interrupted."""
from pathlib import Path

import numpy as np
from PIL import Image

from drift.ui.button import CroppedTemplateButton


class PopupRaidReturn:
    CLOSE = 'driftinggirl_task_popup_close'
    PREPARE = 'driftinggirl_raid_prepare'
    START = 'driftinggirl_raid_start'
    RETURN = 'driftinggirl_task_popup_return'
    FIELDS = ('stage', 'stage_started_at', 'stage_deadline', 'stage_trigger',
              'stage_ready_at', 'skip_on_timeout', 'previous', 'previous_progress',
              'consecutive', 'absent_frames')

    def __init__(self, root):
        self.root = Path(root)
        self.buttons = None
        self.clear()

    def clear(self):
        self.saved = self.origin = self.key = self.since = None
        self.frames = 0

    @staticmethod
    def identity(frame):
        if frame.shape != (1280, 720, 3):
            return None
        # Fixed digit strokes only; normalize modal dimming, not the scene.
        gray = frame[10:62, 72:126].min(axis=2)
        peak = float(np.percentile(gray, 95))
        if peak < 100:
            return None
        rank = gray > max(70, peak * .75)
        if not 50 < int(rank.sum()) < rank.size * .65:
            return None
        stage = frame[46:66, 35:56].astype(np.int16)
        world = ((stage[:, :, 1] > stage[:, :, 0] + 15)
                 & (stage[:, :, 2] > stage[:, :, 0] + 15)
                 & (stage[:, :, 1] >= 70))
        return rank.tobytes(), world.tobytes()

    @staticmethod
    def notice_identity(frame):
        """Bind a separately recognized notice to its dimmed header.

        Closing a known notice can cover an already dimmed settings page.
        RAID arrival/return keep identity()'s original brightness requirement.
        The caller must still match the whole notice and its actual red X.
        """
        if frame.shape != (1280,720,3):
            return None
        gray=frame[10:62,72:126].min(axis=2)
        peak=float(np.percentile(gray,95))
        if (peak < 60
                or float((frame[490:640,300:610].max(axis=2)>=210).mean()) < .5):
            return None
        rank=gray > max(40,peak*.84)
        if not 50 < int(rank.sum()) < rank.size*.65:
            return None
        badge=frame[45:72,13:68].astype(np.int16)
        # Fixed high-chroma cores remain identical across the two modal
        # shades. Weak antialiased edge pixels do not identify the stage.
        world=((badge[:,:,1]-badge[:,:,0]>=peak*.25)
               & (badge[:,:,2]-badge[:,:,0]>=peak*.25)
               & (badge.max(axis=2)>=peak*.5))
        return rank.tobytes(),world.tobytes()

    def _waiting(self, frame):
        if self.buttons is None:
            self.buttons = []
            for name in ('waiting', 'ready'):
                path = self.root / f'assets/raid/{name}.png'
                if not path.is_file():
                    continue
                with Image.open(path) as image:
                    source = np.asarray(image.convert('RGB'))
                title = (310, 254, 410, 280)
                companions = ((420, 584, 466, 611), (325, 637, 396, 655),
                              (185, 240, 226, 279))
                self.buttons.append(CroppedTemplateButton(
                    source[254:280, 310:410], (308, 252, 412, 282), [1],
                    'popup-raid-' + name, source_area=title,
                    required_templates=[(area, source[area[1]:area[3], area[0]:area[2]])
                                        for area in companions],
                    color_condition={'source_area': [420, 584, 466, 611],
                                     'hsv': [[0, 0, 180], [179, 90, 255]],
                                     'minimum_fraction': .14}))
        return any(button.match(frame, similarity=.93) for button in self.buttons)

    def arm(self, st, frame, *, now, stopped=False):
        self.clear()
        prepare = st.matchers.get(self.PREPARE)
        start = st.matchers.get(self.START)
        returned = st.matchers.get(self.RETURN)
        if (stopped or st.handling_interrupt or st.stage != 'raid_wait'
                or prepare is None or start is None or returned is None
                or not prepare[0].get('enabled') or not start[0].get('enabled')
                or not returned[0].get('enabled')
                or prepare[0].get('next_stage') != 'raid_wait'
                or start[0].get('stage') != 'raid_wait'
                or start[0].get('next_stage') != 'raid_battle'
                or st.stage_trigger != prepare[0]['name']
                or st.stage_deadline is None or now >= st.stage_deadline):
            return False
        origin = self.identity(frame)
        if origin is None:
            return False
        self.saved = {field: getattr(st, field) for field in self.FIELDS}
        self.origin = origin
        return True

    def observe(self, st, frame, *, now, stopped=False):
        close = st.matchers.get(self.CLOSE)
        returned = st.matchers.get(self.RETURN)
        if (self.saved is None or stopped or not st.handling_interrupt
                or st.stage != 'task_popup_return' or close is None
                or returned is None or not returned[0].get('enabled')
                or not close[0].get('enabled') or st.stage_trigger != close[0]['name']
                or st.stage_deadline is None
                or now >= min(st.stage_deadline, self.saved['stage_deadline'])):
            self.clear()
            return False
        if (now < st.stage_ready_at
                or close[1].match(frame, offset=close[2], similarity=close[0]['threshold'])
                or self.identity(frame) != self.origin or not self._waiting(frame)):
            self.key = self.since = None
            self.frames = 0
            return False
        key = (st.stage_started_at, self.origin)
        if key != self.key:
            self.key, self.since, self.frames = key, now, 0
        self.frames += 1
        return self.frames >= 3 and now - self.since >= 3

    def restore(self, st, *, now, stopped=False):
        if (self.saved is None or stopped or self.frames < 3 or self.since is None
                or not st.handling_interrupt or st.stage != 'task_popup_return'
                or now - self.since < 3 or st.stage_deadline is None
                or now >= min(st.stage_deadline, self.saved['stage_deadline'])):
            return False
        for field, value in self.saved.items():
            setattr(st, field, value)
        st.handling_interrupt = False
        st.interrupt_started_at = None
        st.stable = {key: 0 for key in st.stable}
        st.number_values.clear()
        st.target_keys.clear()
        self.clear()
        return True
