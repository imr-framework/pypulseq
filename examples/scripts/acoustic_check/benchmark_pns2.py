
import os
import psutil
import threading
import time

from pathlib import Path

import pypulseq as pp
import pulserver.core.Sequence as pseq

# Set system limits
SYSTEM = pp.Opts(
    max_grad=32,
    grad_unit='mT/m',
    max_slew=130,
    slew_unit='T/m/s',
    rf_ringdown_time=0.0,
    rf_dead_time=0.0,
    adc_dead_time=0.0,
    grad_raster_time=4e-6,
    rf_raster_time=2e-6,
    adc_raster_time=2e-6,
    block_duration_raster=4e-6,
)


class MemorySampler(threading.Thread):
    def __init__(self, process, interval=0.01):
        super().__init__()
        self.process = process
        self.interval = interval
        self.peak_rss = 0
        self._running = True

    def run(self):
        while self._running:
            try:
                rss = self.process.memory_info().rss
                self.peak_rss = max(self.peak_rss, rss)
            except psutil.Error:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._running = False

def profile_pns(
    name,
    seqfile,
    report_file="pns_profile_report.txt",
    sample_interval=0.01,
):
    process = psutil.Process(os.getpid())

    # Prepare sequence
    seq0 = pp.Sequence(SYSTEM)
    seq0.read(seqfile)
    seq = pseq.PulserverSequence(seq0)

    # Baseline memory
    mem_before = process.memory_info().rss

    # Start memory sampler
    sampler = MemorySampler(process, interval=sample_interval)
    sampler.start()

    # Runtime
    t0 = time.perf_counter()
    pns = pseq.get_pns(seq, chronaxie_us=360.0, rheobase=20.0, alpha=0.333, do_plot=False)
    runtime = time.perf_counter() - t0

    # Stop sampler
    sampler.stop()
    sampler.join()

    mem_after = process.memory_info().rss
    mem_peak = sampler.peak_rss

    # Convert to MB
    mb = 1024**2
    report = (
        f"Sequence: {name}\n"
        f"File: {seqfile}\n"
        f"Runtime: {runtime:.3f} s\n"
        f"RSS before: {mem_before/mb:.2f} MB\n"
        f"RSS peak:   {mem_peak/mb:.2f} MB\n"
        f"RSS after:  {mem_after/mb:.2f} MB\n"
        f"Peak delta: {(mem_peak - mem_before)/mb:+.2f} MB\n"
        f"{'-'*40}\n"
    )

    # Print
    print(report)

    # Append to file
    path = Path(report_file)
    if path.exists():
        path.write_text(path.read_text() + report)
    else:
        path.write_text(report)

    return pns

if __name__ == '__main__':
    profile_pns("bSSFP (TR 7 ms)", "./bssfp.seq")
    profile_pns("GRE (TR 12 ms)", "./gre.seq")
    profile_pns("HASTE (ESP 16 ms)", "./haste.seq")
    profile_pns("EPI (ESP 640 µs)", "./epi.seq")
    profile_pns("MPRAGE (TR 10 s)", "./mprage.seq")