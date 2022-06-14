"""
Demo low-performance EPI sequence without ramp-sampling.
"""

import numpy as np

import pypulseq as pp


def main(plot: bool, write_seq: bool, seq_filename: str = "epi_pypulseq.seq"):
    # ======
    # SETUP
    # ======
    seq = pp.Sequence()  # Create a new sequence object
    # Define FOV and resolution
    fov = 220e-3
    Nx = 64
    Ny = 64
    slice_thickness = 3e-3  # Slice thickness
    n_slices = 3

    # Set system limits
    system = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )

    # ======
    # CREATE EVENTS
    # ======
    # Create 90 degree slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        system=system,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
    )

    # Define other gradients and ADC events
    delta_k = 1 / fov
    k_width = Nx * delta_k
    dwell_time = 4e-6
    readout_time = Nx * dwell_time
    flat_time = np.ceil(readout_time * 1e5) * 1e-5  # round-up to the gradient raster
    gx = pp.make_trapezoid(
        channel="x",
        system=system,
        amplitude=k_width / readout_time,
        flat_time=flat_time,
    )
    adc = pp.make_adc(
        num_samples=Nx,
        duration=readout_time,
        delay=gx.rise_time + flat_time / 2 - (readout_time - dwell_time) / 2,
    )

    # Pre-phasing gradients
    pre_time = 8e-4
    gx_pre = pp.make_trapezoid(
        channel="x", system=system, area=-gx.area / 2, duration=pre_time
    )
    gz_reph = pp.make_trapezoid(
        channel="z", system=system, area=-gz.area / 2, duration=pre_time
    )
    gy_pre = pp.make_trapezoid(
        channel="y", system=system, area=-Ny / 2 * delta_k, duration=pre_time
    )

    # Phase blip in the shortest possible time
    dur = np.ceil(2 * np.sqrt(delta_k / system.max_slew) / 10e-6) * 10e-6
    gy = pp.make_trapezoid(channel="y", system=system, area=delta_k, duration=dur)

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    # Define sequence blocks
    for s in range(n_slices):
        rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        for i in range(Ny):
            seq.add_block(gx, adc)  # Read one line of k-space
            seq.add_block(gy)  # Phase blip
            gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed! Error listing follows:")
        print(error_report)

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot()  # Plot sequence waveforms

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        seq.write(seq_filename)


if __name__ == "__main__":
    main(plot=True, write_seq=True)
