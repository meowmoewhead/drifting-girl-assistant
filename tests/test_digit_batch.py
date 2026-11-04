from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import unittest

import numpy as np
from PIL import Image

from digit_ocr import DigitReader

ROOT = Path(__file__).resolve().parents[1]


class Session:
    def __init__(self): self.batches = []
    def run(self, _, inputs):
        count = next(iter(inputs.values())).shape[0]
        self.batches.append(count)
        output = np.zeros((count, 3, 4), np.float32)
        output[:, :, 0] = 1.
        output[:, 1, 0] = 0.
        output[:, 1, 1] = .99
        return [output]


class DigitBatchTests(unittest.TestCase):
    def setUp(self):
        self.reader = DigitReader.__new__(DigitReader)
        self.reader.characters = ['', '1', 'S', ' ']
        self.reader.input_name = 'image'
        self.reader.session = Session()
        self.reader._frame = threading.local()
        self.reader._statistics_lock = threading.Lock()
        self.reader._statistics = dict(requests=0, model_calls=0, model_images=0,
                                       cache_hits=0, inference_ms=0.)
        self.image = np.full((20, 20, 3), 100, np.uint8)

    def test_exact_pixels_reuse_only_inside_the_current_frame(self):
        self.reader.begin_frame()
        first = self.reader.read(self.image)
        self.assertEqual(first, self.reader.read(self.image.copy()))
        self.assertEqual(self.reader.session.batches, [1])
        changed = self.image.copy(); changed[0, 0, 0] += 1
        self.reader.read(changed)
        self.reader.begin_frame()
        self.reader.read(self.image)
        self.reader.end_frame()
        self.reader.read(self.image); self.reader.read(self.image)
        self.assertEqual(self.reader.session.batches, [1]*5)

    def test_independent_views_are_batched_and_duplicates_keep_their_votes(self):
        other = self.image.copy(); other[0, 0, 0] += 1
        result = self.reader.read_many([self.image, other, self.image.copy()])
        self.assertEqual(len(result), 3)
        self.assertTrue(all(reading.value == 1 for reading in result))
        self.assertEqual(self.reader.session.batches, [2])

    def test_large_batch_is_bounded(self):
        images = [np.full((20,20,3), i, np.uint8) for i in range(17)]
        self.assertEqual(len(self.reader.read_many(images)), 17)
        self.assertEqual(self.reader.session.batches, [8,8,1])

    def test_thread_scopes_do_not_share_previous_frames(self):
        def recognize(_):
            self.reader.begin_frame()
            self.reader.read(self.image); self.reader.read(self.image)
            self.reader.end_frame()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(recognize, range(2)))
        self.assertEqual(self.reader.session.batches, [1,1])

    def test_empty_letters_and_nonfinite_predictions_are_never_zero(self):
        invalid = self.reader.read(np.zeros((1,1,3),np.uint8))
        self.assertIsNone(invalid.value)
        probabilities = np.zeros((3,4),np.float32); probabilities[:,0] = 1.
        self.assertIsNone(self.reader._decode(probabilities).value)
        probabilities[1,0] = 0.; probabilities[1,2] = .99
        self.assertEqual(self.reader._decode(probabilities).text, 'S')
        self.assertIsNone(self.reader._decode(probabilities).value)
        probabilities[1,2] = np.nan
        self.assertIsNone(self.reader._decode(probabilities).value)


class RealNumericBatchTests(unittest.TestCase):
    def test_batch_keeps_real_strokes_text_confidence_and_rejection(self):
        reader = DigitReader(ROOT)
        views = []
        for name in ('quest-sleep-rank3.png','bag-rank5-stopped-map.png','quest-success-rank4.png'):
            frame = np.asarray(Image.open(ROOT/'tests/fixtures/parallel'/name).convert('RGB'))
            for x,y,X,Y in ((80,12,123,58),(71,9,136,59)):
                views.append(frame[y:Y,x:X])
        single = [reader.read(view) for view in views]
        batch = reader.read_many(views)
        self.assertEqual([(r.text,r.value) for r in single], [(r.text,r.value) for r in batch])
        for before, after in zip(single,batch):
            self.assertAlmostEqual(before.confidence,after.confidence,places=4)
            self.assertEqual(before.confidence >= .95, after.confidence >= .95)


if __name__ == '__main__': unittest.main()
