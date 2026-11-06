"""Keep the shipped sidebar guards when an optional event icon disappears.

The two observed layouts differ by one 90 px row. Only sidebar fragments move;
bottom controls remain fixed, including in compound return/menu observations.
Every variant still matches the current image and retains all original guards.
"""
from copy import deepcopy
import math

from alas_core.button import Button


SUPPORTED = {
    'driftinggirl_daily_open': ('assets/daily-reward/main.png', (651,370,696,414)),
    'driftinggirl_guild_open': ('assets/guild-shop/main.png', (649,371,698,415)),
    'driftinggirl_guild_return': ('assets/guild-shop/main.png', (649,371,698,415)),
    'driftinggirl_mailbox_open': ('assets/main-ui-native.png', (659,479,701,505)),
    'driftinggirl_achievements_open': ('assets/main-ui-native.png', (650,550,690,587)),
    'driftinggirl_achievements_resume': ('assets/achievements-menu-native.png', (555,625,592,646)),
    'driftinggirl_achievements_crown': ('assets/achievements-menu-native.png', (555,625,592,646)),
    'driftinggirl_achievements_fold': ('assets/achievements-menu-native.png', (555,625,592,646)),
    'driftinggirl_achievements_claim_menu_recover': ('assets/achievements-menu-parallel.png', (555,625,592,646)),
    'driftinggirl_achievements_recover': ('assets/navigation/main.png', (211,1140,266,1189)),
    'driftinggirl_mailbox_recover': ('assets/navigation/main.png', (211,1140,266,1189)),
    'driftinggirl_mailbox_return': ('assets/navigation/main.png', (211,1140,266,1189)),
}


def floating(area):
    return 500 <= area[0] < area[2] <= 720 and 230 <= area[1] < area[3] <= 900


def translated(area, dy):
    if area is None:
        return None
    x1,y1,x2,y2 = area
    move = dy if floating(area) else 0
    return (x1,y1+move,x2,y2+move)


class SidebarLayoutButton(Button):
    sidebar_layout = True

    def __init__(self, original, rule):
        super().__init__(area=original.area, color=(0,0,0), button=original.button,
                         name=original.name)
        self.variants = original.variants  # Alternate assets are merged before wrapping.
        shifted = deepcopy(original)
        primary_move = -90 if floating(original.source_area) else 0
        shifted.source_area = translated(original.source_area, -90)
        shifted.click_source_area = translated(original.click_source_area, -90)
        if shifted.color_condition is not None:
            shifted.color_condition['source_area'] = translated(
                shifted.color_condition['source_area'], -90)
        variants=[]
        for candidate, search, companions in shifted.variants:
            guards=[]
            for area, (guard,dx,dy,gw,gh,tolerance) in zip(
                    rule.get('required_source_areas', []), companions, strict=True):
                companion_move = -90 if floating(area) else 0
                guards.append((guard,dx,dy+companion_move-primary_move,gw,gh,tolerance))
            x1,y1,x2,y2=search
            variants.append((candidate,(x1,y1+primary_move,x2,y2+primary_move),guards))
        shifted.variants=variants
        self.layouts=((0,original),(-90,shifted))
        self.active=None
        self.target_key=None
        self.last_similarity=-1.0
        self.matched_box=None

    def match(self, image, offset=0, similarity=.92):
        self.active=None
        self.target_key=None
        self.matched_box=None
        self._button_offset=None
        self.last_similarity=-1.0
        if image.shape != (1280,720,3):
            return False
        matches=[]
        for delta,button in self.layouts:
            passed=button.match(image,offset=offset,similarity=similarity)
            if math.isfinite(button.last_similarity):
                self.last_similarity=max(self.last_similarity,button.last_similarity)
            if passed:
                matches.append((delta,button))
        if not matches or len({tuple(button.button) for _,button in matches}) != 1:
            # Keep a positive score for ambiguity so absence checks stay closed.
            return False
        # Overlapping searches may locate the same real icon in both layouts.
        # Prefer the first valid hypothesis, avoiding score-rounding vote churn.
        self.active=matches[0]
        delta,button=self.active
        self._button_offset=tuple(button.button)
        self.matched_box=button.matched_box
        self.target_key=(delta,self._button_offset)
        self.last_similarity=button.last_similarity
        return True

    def source_to_screen(self, area):
        if self.active is None:
            raise ValueError('Sidebar target has not been confirmed in this frame')
        delta,button=self.active
        return button.source_to_screen(translated(area,delta))


def adapt_sidebar(rule, button):
    expected=SUPPORTED.get(rule.get('id'))
    if (expected is None or (rule.get('asset'),tuple(rule.get('source_crop',()))) != expected
            or rule.get('scales',[1]) != [1] or rule.get('kind','template') != 'template'):
        return button
    return SidebarLayoutButton(button,rule)


