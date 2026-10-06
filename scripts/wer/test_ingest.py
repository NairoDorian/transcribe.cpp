#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=5.1.0", "huggingface-hub>=2.1.1", "numpy", "soundfile>=0.14.0"]
# ///
"""Regression checks for the Datasets 5 Audio and Parquet ingestion path."""
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Audio, Dataset
import numpy as np
import soundfile as sf

import ingest


class IngestTests(unittest.TestCase):
    def test_current_audio_records_and_resampling(self):
        raw = io.BytesIO()
        sf.write(raw, np.full((800, 2), 0.25, dtype=np.float32), 8000, format="WAV")
        ds = Dataset.from_list([{"audio": {"bytes": raw.getvalue(), "path": "clip.wav"}}])
        ds = ds.cast_column("audio", Audio(decode=False))
        data, sr = ingest.read_hf_audio(ds[0]["audio"])
        self.assertEqual(data.shape, (800,))
        self.assertEqual(sr, 8000)
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "out.wav"
            ingest.write_wav_16k_mono(data, sr, wav)
            audio, rate = sf.read(wav)
            self.assertEqual(rate, 16000)
            self.assertEqual(audio.shape, (1600,))
            np.testing.assert_allclose(audio, 0.25, atol=1e-4)

    def test_fleurs_parquet_preserves_recording_ids_and_unicode(self):
        raw = io.BytesIO()
        sf.write(raw, np.zeros(160, dtype=np.float32), 16000, format="WAV")
        ds = Dataset.from_list([
            {"id": 7, "path": "002.wav", "audio": {"bytes": raw.getvalue(), "path": "002.wav"}, "transcription": "été"},
            {"id": 7, "path": "001.wav", "audio": {"bytes": raw.getvalue(), "path": "001.wav"}, "transcription": "bonjour"},
        ])
        with tempfile.TemporaryDirectory() as directory, \
                patch("huggingface_hub.HfApi.list_repo_files", return_value=["fr_fr/test/0000.parquet", "fr_fr/train/0000.parquet"]), \
                patch("datasets.load_dataset", return_value=ds) as loader:
            repo = Path(directory)
            result = ingest.ingest_fleurs(repo, argparse.Namespace(lang="fr", split="test", force=True))
            self.assertEqual(result, 0)
            args, kwargs = loader.call_args
            self.assertEqual(args, ("parquet",))
            self.assertEqual(kwargs["split"], "test")
            self.assertNotIn("trust_remote_code", kwargs)
            self.assertEqual(len(kwargs["data_files"]["test"]), 1)
            rows = [json.loads(line) for line in (repo / "samples/wer/fleurs-fr.manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in rows], ["fleurs-fr-001", "fleurs-fr-002"])
            self.assertEqual(rows[1]["ref_text"], "été")
            self.assertTrue(all(Path(row["audio"]).is_file() for row in rows))


if __name__ == "__main__":
    unittest.main()
