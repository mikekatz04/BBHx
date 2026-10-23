"""Definition of BBHX package common exceptions.

:class:`BBHXException` is the root class. The module is star-imported by
:mod:`bbhx.cutils`, where :class:`CudaException`, :class:`CuPyException` and
:class:`MissingDependency` shadow the ``gpubackendtools.exceptions`` classes of the same
name. The input-file, configuration, file-manager and trajectory exceptions were copied
from FastEMRIWaveforms' ``few.utils.exceptions``; nothing in BBHx raises them.
``ExceptionGroup`` is re-exported (the ``exceptiongroup`` backport when installed, the
builtin otherwise).
"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    ExceptionGroup = ExceptionGroup


class BBHXException(Exception):
    """Base class for BBHX package exceptions."""

    pass


class CudaException(BBHXException):
    """Base class for CUDA-related exceptions."""

    pass


class CuPyException(BBHXException):
    """Base class for CuPy-related exceptions."""

    pass


class MissingDependency(BBHXException):
    """Exception raised when a required dependency is missing."""

    pass


# TODO check: keep/remove/document -- copied from few.utils.exceptions; nothing in BBHx raises it
class InvalidInputFile(BBHXException):
    """Exception raised when the content of an input file does not match expectations."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no config layer
class ConfigurationError(BBHXException):
    """Exception raised when configuration setup fails."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no config layer
class ConfigurationMissing(ConfigurationError):
    """Exception raised when an expected configuration entry is missing."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no config layer
class ConfigurationValidationError(ConfigurationError):
    """Exception raised when a configuration entry is invalid."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileManagerException(BBHXException):
    """Exception raised by the FileManager."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileNotInRegistry(FileManagerException):
    """Exception raised when a requested file is not in file registry."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileNotFoundLocally(FileManagerException):
    """Exception raised when file not found locally but expected to be."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileInvalidChecksum(FileManagerException):
    """Exception raised when file has invalid checksum."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileDownloadException(FileManagerException):
    """Exception raised if file download failed."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileDownloadNotFound(FileDownloadException):
    """Exception raised if file is not found at expected URL."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileDownloadConnectionError(FileDownloadException):
    """Exception raised in case of connection error during download."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileDownloadIntegrityError(FileDownloadException):
    """Exception raised in case of integrity error after download."""


# TODO check: keep/remove/document -- copied from few.utils.exceptions; BBHx has no FileManager
class FileManagerDisabledAccess(FileManagerException):
    """Exception raised when trying to access a file whose tags are disabled"""

    disabled_tag: str
    file_name: str

    def __init__(self, /, *args, disabled_tag: str, file_name: str, **kwargs):
        self.disabled_tag = disabled_tag
        self.file_name = file_name
        super().__init__(*args, **kwargs)


### Trajectory-related exceptions
# TODO check: keep/remove/document -- copied from few.utils.exceptions; unused in BBHx
class TrajectoryOffGridException(Exception):
    """Exception raised when a trajectory goes off-grid (except for the lower boundary in p)."""
