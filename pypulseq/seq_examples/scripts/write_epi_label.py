"""
Demo low-performance EPI sequence without ramp-sampling.
In addition, it demonstrates how the LABEL extension can be used to set data header values, which can be used either in
combination with integrated image reconstruction or to guide the off-line reconstruction tools.
"""

import numpy as np

import pypulseq as pp
from pypulseq import calc_rf_center


def main(plot: bool, write_seq: bool, seq_filename: str = "epi_lable_pypulseq.seq"):
    # ======
    # SETUP
    # ======
    seq = pp.Sequence()  # Create a new sequence object
    fov = 220e-3  # Define FOV and resolution
    Nx = 96
    Ny = 96
    slice_thickness = 3e-3  # Slice thickness
    n_slices = 7
    n_reps = 4
    navigator = 3

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

    # Define trigger
    trig = pp.make_trigger(channel="physio1", duration=2000e-6)

    # Define other gradients and ADC events
    delta_k = 1 / fov
    k_width = Nx * delta_k
    dwell_time = 4e-6
    readout_time = Nx * dwell_time
    flat_time = np.ceil(readout_time * 1e5) * 1e-5  # Round-up to the gradient raster
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
        channel="y", system=system, area=Ny / 2 * delta_k, duration=pre_time
    )

    # Phase blip in the shortest possible time
    dur = np.ceil(2 * np.sqrt(delta_k / system.max_slew) / 10e-6) * 10e-6
    gy = pp.make_trapezoid(channel="y", system=system, area=-delta_k, duration=dur)

    gz_spoil = pp.make_trapezoid(channel="z", system=system, area=delta_k * Nx * 4)

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    # Define sequence blocks
    for r in range(n_reps):
        seq.add_block(trig, pp.make_label(type="SET", label="SLC", value=0))
        for s in range(n_slices):
            rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            # Compensate for the slide-offset induced phase
            rf.phase_offset = -rf.freq_offset * calc_rf_center(rf)[0]
            seq.add_block(rf, gz)
            seq.add_block(
                gx_pre,
                gz_reph,
                pp.make_label(type="SET", label="NAV", value=1),
                pp.make_label(type="SET", label="LIN", value=np.round(Ny / 2)),
            )
            for n in range(navigator):
                seq.add_block(
                    gx,
                    adc,
                    pp.make_label(type="SET", label="REV", value=gx.amplitude < 0),
                    pp.make_label(type="SET", label="SEG", value=gx.amplitude < 0),
                    pp.make_label(type="SET", label="AVG", value=n + 1 == 3),
                )
                if n + 1 != navigator:
                    # Dummy blip pulse to maintain identical RO gradient timing and the corresponding eddy currents
                    seq.add_block(pp.make_delay(pp.calc_duration(gy)))

                gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

            # Reset lin/nav/avg
            seq.add_block(
                gy_pre,
                pp.make_label(type="SET", label="LIN", value=0),
                pp.make_label(type="SET", label="NAV", value=0),
                pp.make_label(type="SET", label="AVG", value=0),
            )

            for i in range(Ny):
                seq.add_block(
                    pp.make_label(type="SET", label="REV", value=gx.amplitude < 0),
                    pp.make_label(type="SET", label="SEG", value=gx.amplitude < 0),
                )
                seq.add_block(gx, adc)  # Read one line of k-space
                # Phase blip
                seq.add_block(gy, pp.make_label(type="INC", label="LIN", value=1))
                gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

            seq.add_block(
                gz_spoil,
                pp.make_delay(0.1),
                pp.make_label(type="INC", label="SLC", value=1),
            )
            if np.remainder(navigator + Ny, 2) != 0:
                gx.amplitude = -gx.amplitude

        seq.add_block(pp.make_label(type="INC", label="REP", value=1))

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
        seq.plot(
            time_range=(0, 0.1), time_disp="ms", label="SEG, LIN, SLC"
        )  # Plot sequence waveforms

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        # Prepare sequence report
        seq.set_definition(key="FOV", value=[fov, fov, slice_thickness * n_slices])
        seq.set_definition(key="Name", value="epi_lbl")
        seq.write(seq_filename)


if __name__ == "__main__":
    main(plot=True, write_seq=True)
