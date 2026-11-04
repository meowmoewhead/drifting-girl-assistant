"""Recognise the right-hand bonus card independently of the free reward."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

BONUS_ID = 'driftinggirl_daily_bonus'


def supported_bonus(rule):
    return (rule.get('id') == BONUS_ID
            and rule.get('asset') == 'assets/daily-reward/premium-ready.png'
            and rule.get('source_crop') == [177,336,310,366]
            and rule.get('click_source_area') == [410,430,605,500])


class DailyBonusVisual:
    """Fixed foreground marks only; neither voucher count nor paid plus is input."""
    def __init__(self, root):
        ready = np.asarray(Image.open(Path(root)/'assets/daily-reward/bonus-ready.png').convert('RGB'))
        self.parts = [(box, self.crop(ready,box)) for box in
                      ((171,250,263,280),(177,336,310,366),(637,251,688,294),
                       (474,399,570,429))]
        claimed = np.asarray(Image.open(Path(root)/'assets/daily-reward/bonus-claimed.png').convert('RGB'))
        self.claimed_label = self.crop(claimed,self.parts[-1][0])
        self.white = self.parts[-1][1].min(axis=2) >= 230
        if int(self.white.sum()) < 300:
            raise ValueError('附赠奖励亮色文字样本无效')

    @staticmethod
    def crop(frame, box):
        x,y,X,Y=box
        return frame[y:Y,x:X]

    def state(self, frame):
        if frame.shape != (1280,720,3):
            return None
        for index,(box,reference) in enumerate(self.parts):
            current=self.crop(frame,box)
            score=float(cv2.matchTemplate(cv2.cvtColor(current,cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),cv2.TM_CCOEFF_NORMED)[0,0])
            # The claimed card redraws the label's shadow. Its real fixed
            # strokes need a separate sample; keep the original threshold.
            if index == len(self.parts)-1:
                claimed_score=float(cv2.matchTemplate(cv2.cvtColor(current,cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(self.claimed_label,cv2.COLOR_RGB2GRAY),cv2.TM_CCOEFF_NORMED)[0,0])
                score=max(score,claimed_score)
            if not np.isfinite(score) or score < .94:
                return None
        # A modal dimming the entire page must never turn a ready card into gray.
        title=self.crop(frame,(177,336,310,366))
        if float(((title.max(axis=2)>=180)&(title.min(axis=2)>=100)).mean()) < .5:
            return None
        pixels=self.crop(frame,(474,399,570,429))[self.white]
        if float((pixels.min(axis=1)>=220).mean()) >= .9:
            return 'ready'
        if (float((pixels.max(axis=1)<=140).mean()) >= .9
                and float(pixels.mean()) >= 65):
            return 'gray'
        return None
