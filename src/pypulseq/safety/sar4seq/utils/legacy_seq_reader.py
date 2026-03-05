from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LegacyRF:
    signal: complex
    num_samples: int
    duration_s: float


@dataclass
class LegacyBlock:
    rf: Optional[LegacyRF]
    block_duration: float


def parse_legacy_seq(path: str, rf_raster_s: float = 1e-6) -> List[LegacyBlock]:
    """Very lightweight parser for Pulseq v1.2.1 to extract RF blocks and durations.

    Assumptions:
    - [BLOCKS] rows map to RF ids; non-zero RF column indicates RF event id
    - [RF] lines: id amplitude mag_id phase_id delay freq phase
    - [SHAPES] blocks: 'shape_id X' then 'num_samples N' — we use N to estimate RF duration
    - [DELAYS] blocks: id delay_us - handle delay blocks for correct TR calculation
    - RF envelope shape magnitude is not reconstructed; we approximate by constant amplitude
    - Gradient/ADC timing not modeled; block duration approximated as RF duration where RF present, else delay duration
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Section indices
    sec = None
    blocks = []
    rf_events = {}
    shapes_samples = {}
    delays = {}  # Add delay storage

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        if line.startswith('[') and line.endswith(']'):
            sec = line[1:-1]
            i += 1
            continue

        if sec == 'BLOCKS':
            # columns: # D RF GX GY GZ ADC
            parts = line.split()
            if len(parts) >= 7:
                _, D, RF, _GX, _GY, _GZ, _ADC = parts[:7]
                blocks.append({'D': int(D), 'RF': int(RF)})
        elif sec == 'RF':
            parts = line.split()
            if len(parts) >= 7:
                rid = int(parts[0])
                amp = float(parts[1])
                mag_id = int(parts[2])
                phase_id = int(parts[3])
                delay_us = float(parts[4])
                rf_events[rid] = {
                    'amp': amp,
                    'mag_id': mag_id,
                    'phase_id': phase_id,
                    'delay_s': delay_us * 1e-6,
                }
        elif sec == 'DELAYS':
            # Parse delay blocks: id delay_us
            parts = line.split()
            if len(parts) >= 2:
                delay_id = int(parts[0])
                delay_us = float(parts[1])
                delays[delay_id] = delay_us * 1e-6  # Convert to seconds
        elif sec == 'SHAPES':
            if line.startswith('shape_id'):
                shape_id = int(line.split()[1])
                # next line should be num_samples
                i += 1
                ns_line = lines[i].strip()
                if ns_line.startswith('num_samples'):
                    num_samples = int(ns_line.split()[1])
                    shapes_samples[shape_id] = num_samples
            # skip rest; we only need num_samples
        i += 1

    legacy_blocks: List[LegacyBlock] = []
    for b in blocks:
        delay_id = b['D']
        rf_id = b['RF']

        # Calculate block duration
        block_duration = 0.0
        rf_block = None

        if rf_id != 0 and rf_id in rf_events:
            # RF block
            evt = rf_events[rf_id]
            ns = shapes_samples.get(evt['mag_id'], 0)
            rf_duration = ns * rf_raster_s + evt['delay_s']
            rf_block = LegacyRF(signal=complex(evt['amp'], 0.0), num_samples=ns, duration_s=rf_duration)
            block_duration = rf_duration

        # Add delay if present (delays can be combined with RF)
        if delay_id != 0 and delay_id in delays:
            block_duration += delays[delay_id]

        # If no RF and no delay, estimate minimal block duration
        if block_duration == 0.0:
            block_duration = 10e-6  # 10 microseconds minimal block duration

        legacy_blocks.append(LegacyBlock(rf=rf_block, block_duration=block_duration))

    return legacy_blocks


