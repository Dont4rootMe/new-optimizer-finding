"""Data pipeline for Speech Commands tiny transformer experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from experiments.optimization_survey._runtime.errors import OptionalDependencyError


class _SpeechFeatureDataset(Dataset):
    """Wrap Speech Commands samples into fixed log-mel features."""

    def __init__(self, base_dataset: Dataset, cfg: DictConfig, label_to_idx: dict[str, int], train: bool) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.label_to_idx = label_to_idx
        self.train = train

        import torchaudio

        self.target_sr = int(cfg.data.sample_rate)
        self.target_samples = int(float(cfg.data.clip_seconds) * self.target_sr)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_mels=int(cfg.data.n_mels),
            win_length=int(cfg.data.win_length),
            hop_length=int(cfg.data.hop_length),
        )

        self.use_specaugment = bool(cfg.data.specaugment) and train
        if self.use_specaugment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(int(cfg.data.freq_mask_param))
            self.time_mask = torchaudio.transforms.TimeMasking(int(cfg.data.time_mask_param))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        import torchaudio

        waveform, sample_rate, label, _, _ = self.base_dataset[idx]

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sr)

        if waveform.size(1) < self.target_samples:
            pad = self.target_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.size(1) > self.target_samples:
            waveform = waveform[:, : self.target_samples]

        mel = self.mel_transform(waveform)
        if self.use_specaugment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        mel = torch.log(mel.clamp_min(1e-6))
        features = mel.squeeze(0).transpose(0, 1).contiguous()
        label_idx = self.label_to_idx[label]
        return features, label_idx


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    """Build Speech Commands dataloaders and metadata."""

    try:
        import torchaudio
    except ImportError as exc:
        raise OptionalDependencyError("audio", "Install extra dependencies: pip install -e .[audio]") from exc

    data_root = Path(str(cfg.data.root)).expanduser()

    train_base = torchaudio.datasets.SPEECHCOMMANDS(root=str(data_root), subset="training", download=True)
    val_base = torchaudio.datasets.SPEECHCOMMANDS(root=str(data_root), subset="validation", download=True)
    test_base = torchaudio.datasets.SPEECHCOMMANDS(root=str(data_root), subset="testing", download=True)

    labels = sorted({sample[2] for sample in train_base})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_set = _SpeechFeatureDataset(train_base, cfg, label_to_idx, train=True)
    val_set = _SpeechFeatureDataset(val_base, cfg, label_to_idx, train=False)
    test_set = _SpeechFeatureDataset(test_base, cfg, label_to_idx, train=False)

    batch_size = int(cfg.compute.batch_size)
    num_workers = int(cfg.compute.num_workers)
    pin_memory = str(cfg.compute.device) == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": len(labels),
    }
