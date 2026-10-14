from __future__ import annotations

from collections import deque


class CausalOnsetPeaks:
    def __init__(self, threshold=.30):
        self.threshold = threshold
        self.samples = deque(maxlen=7)

    def observe(self, timestamp, probability, valid):
        self.samples.append((timestamp, probability, valid))
        if len(self.samples) != 7:
            return None
        samples = list(self.samples)
        center = samples[3]
        if (all(sample[2] > 0 for sample in samples) and center[1] >= self.threshold
                and all(right[0] - left[0] == 3000 for left, right in zip(samples, samples[1:]))
                and all(center[1] > sample[1] for sample in samples[:3])
                and all(center[1] >= sample[1] for sample in samples[4:])):
            return center[0], center[1], timestamp
        return None
