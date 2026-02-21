import argparse
import matplotlib.pyplot as plt

from pypulseq import Sequence

def main():
    """
    Test and visualize Pulseq sequence files.

    This command-line tool reads a .seq file and provides options to:
    - Display a test report of the sequence
    - Plot sequence waveforms (RF, gradients, ADC)
    - Visualize k-space trajectory

    Command-line arguments:
        seq_file: Path to the .seq file to analyze
        --report, -r: Print a detailed test report
        --plot, -p: Display sequence waveforms
        --k-space, -k: Plot k-space trajectory with ADC sampling points

    At least one action flag (--report, --plot, or --k-space) must be specified.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('seq_file', type=str, help='Path to .seq file')
    argparser.add_argument('--k-space', '-k', action='store_true', help='Plot k-space trajectory')
    argparser.add_argument('--plot', '-p', action='store_true', help='Plot sequence waveforms')
    argparser.add_argument('--report', '-r', action='store_true', help='Print test report')
    args = argparser.parse_args()

    if not (args.k_space or args.plot or args.report):
        print("No action specified. Use --k-space, --plot, or --report.")
        return

    seq = Sequence()
    seq.read(args.seq_file)

    if args.report:
        print(seq.test_report())

    if args.plot:
        seq.plot(plot_now=False)

    if args.k_space:
        k_traj_adc, k_traj, *_ = seq.calculate_kspace()
        plt.figure()
        plt.plot(k_traj[1], k_traj[2])
        plt.plot(k_traj_adc[1], k_traj_adc[2], '.', alpha=0.5)
        ax = plt.gca()
        ax.set_aspect('equal')

    if args.k_space or args.plot:
        plt.show()

if __name__ == "__main__":
    main()