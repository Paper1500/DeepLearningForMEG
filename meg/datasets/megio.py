from abc import ABC, abstractmethod, abstractproperty

import mne
import numpy as np
import torch
from torchaudio.transforms import Resample


class MegFile(ABC):

    def __init__(self, filename, subject,
                 filter_type: str = 'mne',
                 window_size=1000,
                 sample_generation: str = 'numpy',
                 ):
        self.filename = filename
        self.subject = subject
        self.filter_type = filter_type
        self.window_size = window_size
        self.sample_generation = sample_generation
        self.resamplers = dict()

    @abstractmethod
    def generate_samples(self):
        pass

    def _create_resampler(self, factor):
        return Resample(self.sfreq, self.sfreq // factor)

    def resample(self, data, factor):
        if factor not in self.resamplers:
            self.resamplers[factor] = self._create_resampler(factor)
        
        rs = self.resamplers[factor]
        data = torch.from_numpy(data.astype(np.float32))
        return rs(data).numpy()


class MNEFile(MegFile):

    def filter(self, raw):
        return raw.filter(l_freq=1, h_freq=45)

    def downsample(self, data, factor):
        return data[:, ::factor]

    @abstractproperty
    def channels(self):
        pass

    @abstractproperty
    def events(self):
        pass

    @abstractmethod
    def parse_event(self, event):
        pass

    def load_mne_info(self):
        self.sfreq = self.raw.info['sfreq']

    def _generate_mne_samples(self):
        channels = self.channels
        tmin = -0.3
        tmax = self.window_size / 1000

        epochs = mne.epochs.Epochs(
            self.raw,
            self.events,
            tmin=tmin,
            tmax=tmax,
            detrend=1,
            picks=channels
        )
        for epoch, event in zip(epochs.get_data(), epochs.events[:, -1]):
            prestim_window = int(0.3 * self.sfreq)
            prestim = epoch[:, :prestim_window]
            poststim = epoch[:, prestim_window:]
            yield prestim, poststim, event

    def _generate_numpy_samples(self):
        channels = self.channels

        for onset, _, event in self.events:
            onset -= self.raw.first_samp

            start = onset - 300
            end = onset + self.window_size
            data, _ = self.raw[channels, start: end]
            baseline = data[:, : 300]
            window = data[:, 300: end]

            yield baseline, window, event

    def generate_samples(self):
        self.load()
        info = self.info

        if self.sample_generation == 'numpy':
            samples = self._generate_numpy_samples()
        elif self.sample_generation == 'mne':
            samples = self._generate_mne_samples()
        else:
            raise NotImplementedError(
                f'Variant {self.variant} is not implemented on {self.__name__}')

        for i, (baseline, window, event) in enumerate(samples):
            sample_info = {
                'filename': f'{self.subject}-{i}.npy',
                **info,
                **self.parse_event(event)
            }
            yield (baseline, window), sample_info


class CamCanFile(MNEFile):

    @property
    def info(self):
        return {
            'subject': self.subject
        }

    def load(self):
        raw = mne.io.RawFIF(self.filename, preload=True, verbose='CRITICAL')
        if self.filter_type == 'mne':
            raw = self.filter(raw)
        self.raw = raw
        self.load_mne_info()

    @property
    def events(self):
        events = mne.find_events(
            self.raw,
            stim_channel='STI101',
            min_duration=0.003
        )
        # events[:, 0] -= self.raw.first_samp
        return events

    @property
    def channels(self):
        return mne.pick_types(self.raw.info, meg='grad')

    def parse_event(self, event):
        is_audio = event != 9
        if is_audio:
            tone = event
        else:
            tone = None
        return {
            'event': event,
            'is_audio': is_audio,
            'tone': tone
        }


class MousFile(MNEFile):

    @property
    def info(self):
        return {
            'subject': self.subject,
            'is_audio': 'A' in self.subject
        }

    @property
    def excluded_channels(self) -> str:
        return [
            'BP2', 'EEG061', 'EEG062', 'EEG063',
            'EEG064', 'MLC11', 'MLF62', 'MLT37',
            'MRF66', 'MRO52'
        ]

    @property
    def channels(self):
        return mne.pick_types(self.raw.info, meg=True, exclude=self.excluded_channels)

    def load(self):
        raw = mne.io.read_raw_ctf(
            str(self.filename),
            preload=True,
            clean_names=True,
            verbose='CRITICAL'
        )
        if self.filter_type == 'mne':
            raw = self.filter(raw)
        self.raw = raw
        self.load_mne_info()

    @property
    def events(self):
        return [
            (t, v, e)
            for t, v, e in mne.find_events(self.raw, shortest_event=0)
            if e in {2, 4, 6, 8}
        ]

    def parse_event(self, event):
        target_word = event in {2, 4, 6, 8}
        sentence = event in {1, 2, 5, 6}
        relative_clause = event in {1, 2, 3, 4}
        return {
            'event': event,
            'target_word': target_word,
            'sentence': sentence,
            'relative_clause': relative_clause
        }

