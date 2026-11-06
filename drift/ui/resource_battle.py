"""Read-only crown recovery: movable hourglass, fixed bottom controls."""
import math
import cv2

from alas_core.button import Button
from drift.core.util import hsv_mask


class ResourceBattleRecoveryButton(Button):
    def __init__(self, original, reference):
        super().__init__(area=original.area,color=(0,0,0),button=original.button,name=original.name)
        self.original=original
        self.variants=original.variants
        self.last_similarity=-1.0
        self.whale=reference[156:205,116:184].copy()

    def match(self, image, offset=0, similarity=.93):
        self.last_similarity=-1.0
        if image.shape!=(1280,720,3):
            return False
        source=self.original
        if source.match(image,offset=offset,similarity=similarity):
            self.last_similarity=source.last_similarity
            return True
        # The added search is only for the actual whale resource encounter;
        # ordinary/quest bosses retain their separate existing recovery path.
        whale_score=cv2.minMaxLoc(cv2.matchTemplate(image[153:208,113:187],self.whale,cv2.TM_CCOEFF_NORMED))[1]
        if not math.isfinite(whale_score) or whale_score<max(.94,similarity):
            return False
        # The fish name changes the health bar width. Its hourglass can move
        # horizontally; the bottom currency icons and brightness cannot.
        for timer,_search,companions in source.variants:
            w,h=timer.area[2:]
            timer.match(image,offset=(188,125,610-w,165-h),similarity=similarity)
            score=timer.last_similarity
            if not math.isfinite(score) or score<=similarity:
                continue
            for guard,dx,dy,gw,gh,tolerance in companions:
                x,y=source.source_area[0]+dx,source.source_area[1]+dy
                guard.match(image,offset=(x-tolerance,y-tolerance,x+tolerance,y+tolerance),
                            similarity=similarity)
                score=min(score,guard.last_similarity) if math.isfinite(guard.last_similarity) else -1.0
                if score<=similarity:
                    break
            condition=source.color_condition
            x,y,X,Y=condition['source_area']
            if float((hsv_mask(image[y:Y,x:X],condition['hsv'])[:,:,0]>0).mean())<condition['minimum_fraction']:
                score=-1.0
            self.last_similarity=max(self.last_similarity,score)
        return self.last_similarity>similarity


def adapt_resource_battle(rule, button, reference):
    # Only the shipped observation rule. Never adapt an input, custom geometry,
    # or another task; saved settings, timeouts and retry limits stay unchanged.
    if (rule.get('id')=='driftinggirl_achievements_battle_recover'
            and rule.get('asset')=='assets/navigation/world-battle.png'
            and rule.get('source_crop')==[334,133,355,157]
            and rule.get('search_area')==[324,123,365,167]
            and rule.get('required_source_areas')==[[211,1140,266,1189],[542,1140,597,1183]]
            and rule.get('scales',[1])==[1] and rule.get('action')=='observe'
            and rule.get('stage')=='achievements_tab' and rule.get('next_stage')=='main'
            and rule.get('color_condition',{}).get('source_area')==[558,1147,585,1168]
            and not rule.get('click_source_area') and not rule.get('match_hsv')):
        return ResourceBattleRecoveryButton(button,reference)
    return button


