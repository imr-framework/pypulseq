import subprocess
from pathlib import Path
from sys import platform
from typing import Any, List, Tuple, Union

from pypulseq import Sequence

# Dictionary containing scanner definitions, target (groups), and the scanner
# detection cache.
scanner_definitions = {}
scanner_targets = {}
scanner_cache = {}


class ScannerDefinition:
    """
    Abstract base class for scanner install implementations.
    """

    def can_install(self) -> bool:
        """
        Check whether the sequence can be installed to this scanner. E.g.
        check whether a network connection can be established, whether
        transfers work, etc.

        Returns
        -------
        bool
            True when the scanner is available, False otherwise.
        """
        return True

    # Install a sequence to this scanner
    def install(self, seq: Sequence, **kwargs: Any) -> bool:
        """
        Install a given sequence to this scanner. Ideally `can_install` should
        be checked beforehand.

        Parameters
        ----------
        seq : Sequence
            Sequence object to install.
        **kwargs : Any
            The implementation for any scanner can accept additional keyword
            arguments (e.g. sequence name).

        Returns
        -------
        bool
            True if the install was successful.
        """
        raise NotImplementedError()


def register_scanner(name: str, definition: ScannerDefinition, groups: Union[List[str], None] = None) -> None:
    """
    Adds a `ScannerDefinition` to the list of known scanner targets.

    Parameters
    ----------
    name : str
        Name of the scanner target.
    definition : ScannerDefinition
        ScannerDefinition implementing the scanner's `install` functionality.
    groups : List[str], optional
        List of target groups the scanner belongs to. The default is None.
    """
    if name in scanner_definitions:
        raise ValueError(f'A target with name `{name}` already exists.')
    if name in scanner_targets:
        raise ValueError(f'A target group with name `{name}` already exists.')

    # Register scanner
    scanner_definitions[name] = definition
    scanner_targets[name] = [name]

    # Add scanner to target groups
    if groups is not None:
        for g in groups:
            if g not in scanner_targets:
                scanner_targets[g] = [name]
            else:
                scanner_targets[g].append(name)


def detect_scanner(target: Union[str, None] = None, clear_cache: bool = False) -> Tuple[str, ScannerDefinition]:
    """
    Detects whether any known scanner is available for installing pulseq
    sequences.

    `target` can specify a specific scanner, or a group of scanners as
    defined by `register_scanner`. If None, all scanners types are tested.

    Parameters
    ----------
    target : str, optional
        Type of scanner to detect. The default is None.
    clear_cache : bool, optional
        Clear the detection cache. The default is False.

    Raises
    ------
    ValueError
        If the `target` is not known.

    Returns
    -------
    Tuple[str, ScannerDefinition]
        The name of the scanner and an instance of the corresponding
        ScannerDefinition object.
    """
    if clear_cache:
        scanner_cache.clear()

    if target in scanner_cache:
        return scanner_cache[target]

    if target is not None:
        if target not in scanner_targets:
            raise ValueError('Unknown scanner target')
        sd = {k: scanner_definitions[k] for k in scanner_targets[target]}
    else:
        sd = scanner_definitions

    for scanner, definition in sd.items():
        if definition.can_install():
            scanner_cache[scanner] = (scanner, definition)
            scanner_cache[target] = (scanner, definition)
            return scanner, definition

    return None, None


def silent_call(command: str) -> int:
    """
    Calls a system command while suppressing all output.

    Parameters
    ----------
    command : str
        Command to execute.

    Returns
    -------
    int
        Status code for the command.
    """
    return subprocess.call(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603


# Implementation for Siemens scanners
class SiemensDefinition(ScannerDefinition):
    """
    Scanner implementation for Siemens Numaris scanners.

    Checks whether the scanner responds to ping at the specified `ice_ip`, and
    if so, checks whether the specified `pulseq_seq_path` exists.

    When installing, saves the sequence to file, transfers it to the ICE
    machine, and renames and moves it into the `pulseq_seq_path`.
    """

    def __init__(self, ice_ip: str, pulseq_seq_path: str):
        self.ice_ip = ice_ip
        self.pulseq_seq_path = pulseq_seq_path

    def execute(self, command: str) -> bool:
        """
        Executes a command on the ICE machine.

        Parameters
        ----------
        command : str
            Command to execute.

        Returns
        -------
        bool
            Status whether the command was successfully executed.
        """
        status = silent_call(
            f'ssh -oBatchMode=yes -oStrictHostKeyChecking=no -oHostKeyAlgorithms=+ssh-rsa root@{self.ice_ip} "{command}"'
        )
        return status == 0

    def can_install(self) -> bool:
        """
        Check whether the sequence can be installed to this scanner.

        Checks:
            1. Does the scanner respond to pings?
            2. Can remote commands be executed, and does the pulseq directory
               exist on the ICE machine?

        Returns
        -------
        bool
            True if all checks pass.
        """
        if platform == 'win32':
            # windows
            ping_command = 'ping -w 1000 -n 1'
        elif platform == 'linux' or platform == 'darwin':
            # unix-like
            ping_command = 'ping -q -n -W1 -c1'

        # Does the scanner respond to pings?
        if silent_call(f'{ping_command} {self.ice_ip}') != 0:
            return False

        # Does the pulseq seq path exist on the scanner? (i.e. is this the right scanner?)
        return self.execute(f'test -d {self.pulseq_seq_path}')

    def install(
        self,
        seq: Sequence,
        name: str = 'external',
        local_filename: str = 'external.seq',
        remove_local_file: bool = True,
    ) -> bool:
        """
        Install the sequence to this Siemens scanner.

        Steps:
            1. Save the sequence to a local file
            2. Transfer the sequence file to the ICE machine with a temporary
               filename
            3. Change permissions on the sequence file
            4. Remove any existing sequence with the specified name
            5. Rename the temporary file to the specified name
            6. Remove the local file if requested

        Parameters
        ----------
        seq : Sequence
            Sequence object to install.
        name : str, optional
            Name for the sequence (.seq is added automatically if not specified).
            The default is 'external'.
        local_filename : str, optional
            Local filename to save the sequence to.
            The default is 'external.seq'.
        remove_local_file : bool, optional
            Whether or not to remove the local sequence file that was saved to
            `local_filename` after installing.
            The default is True.

        Returns
        -------
        bool
            True if the install was successful.
        """
        # Save the sequence to a local file
        seq.write(local_filename)

        # Transfer the sequence file to the ICE machine with a temporary filename
        status = silent_call(
            f'scp -oBatchMode=yes -oStrictHostKeyChecking=no -oHostKeyAlgorithms=+ssh-rsa {local_filename} root@{self.ice_ip}:{self.pulseq_seq_path}/external_tmp.seq'
        )
        if status != 0:
            return False

        # Change permissions on the sequence file
        if not self.execute(f'chmod a+rw {self.pulseq_seq_path}/external_tmp.seq'):
            return False
        # Remove any existing sequence with the specified name
        if not self.execute(f'rm -f {self.pulseq_seq_path}/{name}.seq'):
            return False
        # Rename the temporary file to the specified name
        # TODO: Create directories in pulseq path if `name` contains subdirectories (The Pulseq 1.4.3 interpreter will support subdirectories)
        if not self.execute(f'mv {self.pulseq_seq_path}/external_tmp.seq {self.pulseq_seq_path}/{name}.seq'):
            return False

        # Remove the local file if requested
        if remove_local_file:
            Path(local_filename).unlink()

        return True


# Built-in scanner definitions

# Siemens Numaris 4, two different IPs
register_scanner(
    'siemens_n4_2', SiemensDefinition('192.168.2.2', '/opt/medcom/MriCustomer/seq/pulseq'), ['siemens', 'siemens_n4']
)
register_scanner(
    'siemens_n4_3', SiemensDefinition('192.168.2.3', '/opt/medcom/MriCustomer/seq/pulseq'), ['siemens', 'siemens_n4']
)
# Siemens Numaris X
register_scanner(
    'siemens_nx', SiemensDefinition('192.168.2.2', '/opt/medcom/MriCustomer/CustomerSeq/pulseq'), ['siemens']
)
