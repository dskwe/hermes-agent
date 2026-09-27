"""PEP 706 extraction-filter backport for CPython 3.11.0-3.11.3 tarfiles.

Hermes supports every CPython >= 3.11, but the tarfile extraction-filter API
(``filter=`` on TarFile.extract()/extractall(), ``tarfile.data_filter``,
``TarInfo.replace()``, the ``FilterError`` hierarchy) only shipped in
CPython 3.11.4 / 3.12. On the earlier 3.11 micro releases pm.store's one
containment policy dies with ``TypeError: extractall() got an unexpected
keyword argument 'filter'`` before any archive can bootstrap a runtime.

``patch_tarfile(module)`` is a no-op on interpreters that already have the
API (probe: ``hasattr(module, "data_filter")`` -- also true for distro
tarfiles that backported it, e.g. Ubuntu's 3.10). On legacy modules it
layers the modern semantics on top, mirroring the CPython reference
implementation of the filter pipeline:

- ``TarInfo.replace()`` and the ``FilterError`` exception hierarchy appear.
- ``TarFile.extract()/extractall()`` gain the ``filter=`` keyword; filtered
  members may be altered (the returned TarInfo is extracted) or skipped.
- ``chmod``/``chown``/``utime``/``makedir`` tolerate the ``None`` attrs a
  data-filtered TarInfo carries (ownership dropped, dir/symlink mode
  ignored), which the pre-3.11.4 setters would crash on.
- the hardlink fallback-to-copy re-runs the filter on the target member and
  raises ``LinkFallbackError`` when that target is rejected;
- directory fixup re-applies the filter against the *current* filesystem
  state, as 3.11.4+ does.

pm.store calls this before any filtered extraction; behavior on modern
interpreters is bit-for-bit unchanged (nothing is even assigned there).
"""

from __future__ import annotations

import copy as _copy
import os as _os
import stat as _stat


def patch_tarfile(module=None) -> None:
    """Install the PEP 706 extraction-filter API on a legacy tarfile module.

    No-op when the module already ships it. Idempotent.
    """
    if module is None:
        import tarfile as module  # noqa: PLW0127 shadow-by-assignment in function
    tf = module

    if hasattr(tf, "data_filter"):  # CPython >= 3.11.4 / distro backports
        return

    TarFile = tf.TarFile
    TarInfo = tf.TarInfo
    ExtractError = tf.ExtractError

    symlink_exception = getattr(
        tf, "symlink_exception", (AttributeError, NotImplementedError, OSError)
    )

    # --- the FilterError hierarchy (verbatim from modern tarfile) ---------
    class FilterError(tf.TarError):
        pass

    class AbsolutePathError(FilterError):
        def __init__(self, tarinfo):
            self.tarinfo = tarinfo
            super().__init__(f"member {tarinfo.name!r} has an absolute path")

    class OutsideDestinationError(FilterError):
        def __init__(self, tarinfo, path):
            self.tarinfo = tarinfo
            self._path = path
            super().__init__(
                f"{tarinfo.name!r} would be extracted to {path!r}, "
                + "which is outside the destination"
            )

    class SpecialFileError(FilterError):
        def __init__(self, tarinfo):
            self.tarinfo = tarinfo
            super().__init__(f"{tarinfo.name!r} is a special file")

    class AbsoluteLinkError(FilterError):
        def __init__(self, tarinfo):
            self.tarinfo = tarinfo
            super().__init__(f"{tarinfo.name!r} is a link to an absolute path")

    class LinkOutsideDestinationError(FilterError):
        def __init__(self, tarinfo, path):
            self.tarinfo = tarinfo
            self._path = path
            super().__init__(
                f"{tarinfo.name!r} would link to {path!r}, "
                + "which is outside the destination"
            )

    class LinkFallbackError(FilterError):
        def __init__(self, tarinfo, path):
            self.tarinfo = tarinfo
            self._path = path
            super().__init__(
                f"link {tarinfo.name!r} would be extracted as a "
                + f"copy of {path!r}, which was rejected"
            )

    # Issues with the argument rather than bugs in the filter function.
    _FILTER_ERRORS = (FilterError, OSError, ExtractError)

    # --- TarInfo.replace() -------------------------------------------------
    _KEEP = object()

    def _tarinfo_replace(self, *, name=_KEEP, mtime=_KEEP, mode=_KEEP,
                         linkname=_KEEP, uid=_KEEP, gid=_KEEP, uname=_KEEP,
                         gname=_KEEP, deep=True, _KEEP=_KEEP):
        """Return a copy of self with the given attributes replaced."""
        if deep:
            result = _copy.deepcopy(self)
        else:
            result = _copy.copy(self)
        if name is not _KEEP:
            result.name = name
        if mtime is not _KEEP:
            result.mtime = mtime
        if mode is not _KEEP:
            result.mode = mode
        if linkname is not _KEEP:
            result.linkname = linkname
        if uid is not _KEEP:
            result.uid = uid
        if gid is not _KEEP:
            result.gid = gid
        if uname is not _KEEP:
            result.uname = uname
        if gname is not _KEEP:
            result.gname = gname
        return result

    # --- the filters --------------------------------------------------------
    def _get_filtered_attrs(member, dest_path, for_data):
        new_attrs = {}
        name = member.name
        dest_path = _os.path.realpath(dest_path)
        # Strip leading / (tar's directory separator) from filenames.
        # Include os.sep (target OS directory separator) as well.
        if name.startswith(("/", _os.sep)):
            name = new_attrs["name"] = member.name.lstrip("/" + _os.sep)
        if _os.path.isabs(name):
            # Path is absolute even after stripping.
            # For example, 'C:/foo' on Windows.
            raise AbsolutePathError(member)
        # Ensure we stay in the destination
        target_path = _os.path.realpath(_os.path.join(dest_path, name))
        if _os.path.commonpath([target_path, dest_path]) != dest_path:
            raise OutsideDestinationError(member, target_path)
        # Limit permissions (no high bits, and go-w)
        mode = member.mode
        if mode is not None:
            # Strip high bits & group/other write bits
            mode = mode & 0o755
            if for_data:
                # For data, handle permissions & file types
                if member.isreg() or member.islnk():
                    if not mode & 0o100:
                        # Clear executable bits if not executable by user
                        mode &= ~0o111
                    # Ensure owner can read & write
                    mode |= 0o600
                elif member.isdir() or member.issym():
                    # Ignore mode for directories & symlinks
                    mode = None
                else:
                    # Reject special files
                    raise SpecialFileError(member)
            if mode != member.mode:
                new_attrs["mode"] = mode
        if for_data:
            # Ignore ownership for 'data'
            if member.uid is not None:
                new_attrs["uid"] = None
            if member.gid is not None:
                new_attrs["gid"] = None
            if member.uname is not None:
                new_attrs["uname"] = None
            if member.gname is not None:
                new_attrs["gname"] = None
            # Check link destination for 'data'
            if member.islnk() or member.issym():
                if _os.path.isabs(member.linkname):
                    raise AbsoluteLinkError(member)
                normalized = _os.path.normpath(member.linkname)
                if normalized != member.linkname:
                    new_attrs["linkname"] = normalized
                if member.issym():
                    target_path = _os.path.join(
                        dest_path, _os.path.dirname(name), member.linkname
                    )
                else:
                    target_path = _os.path.join(dest_path, member.linkname)
                target_path = _os.path.realpath(target_path)
                if _os.path.commonpath([target_path, dest_path]) != dest_path:
                    raise LinkOutsideDestinationError(member, target_path)
        return new_attrs

    def fully_trusted_filter(member, dest_path):
        return member

    def tar_filter(member, dest_path):
        new_attrs = _get_filtered_attrs(member, dest_path, False)
        if new_attrs:
            return member.replace(**new_attrs, deep=False)
        return member

    def data_filter(member, dest_path):
        new_attrs = _get_filtered_attrs(member, dest_path, True)
        if new_attrs:
            return member.replace(**new_attrs, deep=False)
        return member

    # --- filter-aware TarFile extraction pipeline --------------------------
    def _get_filter_function(self, filter):
        if filter is None:
            filter = getattr(self, "extraction_filter", None)
            if filter is None:
                return data_filter
            if isinstance(filter, str):
                raise TypeError(
                    "String names are not supported for "
                    + "TarFile.extraction_filter. Use a function such as "
                    + "tarfile.data_filter directly."
                )
            return filter
        if callable(filter):
            return filter
        try:
            return tf._NAMED_FILTERS[filter]
        except KeyError:
            raise ValueError(f"filter {filter!r} not found") from None

    def _handle_fatal_error(self, e):
        """Handle "fatal" error according to self.errorlevel"""
        if self.errorlevel > 0:
            raise
        elif isinstance(e, OSError):
            if e.filename is None:
                self._dbg(1, "tarfile: %s" % e.strerror)
            else:
                self._dbg(1, "tarfile: %s %r" % (e.strerror, e.filename))
        else:
            self._dbg(1, "tarfile: %s %s" % (type(e).__name__, e))

    def _handle_nonfatal_error(self, e):
        """Handle non-fatal error (ExtractError) according to errorlevel"""
        if self.errorlevel > 1:
            raise
        else:
            self._dbg(1, "tarfile: %s" % e)

    def _log_no_directory_fixup(self, member, reason):
        self._dbg(2, "tarfile: Not fixing up directory %r (%s)" %
                  (member.name, reason))

    def _get_extract_tarinfo(self, member, filter_function, path):
        """Get (filtered, unfiltered) TarInfos from *member*

        *member* might be a string.

        Return (None, None) if not found.
        """
        if isinstance(member, str):
            unfiltered = self.getmember(member)
        else:
            unfiltered = member

        filtered = None
        try:
            filtered = filter_function(unfiltered, path)
        except (OSError, UnicodeEncodeError, FilterError) as e:
            self._handle_fatal_error(e)
        except ExtractError as e:
            self._handle_nonfatal_error(e)
        if filtered is None:
            self._dbg(2, "tarfile: Excluded %r" % unfiltered.name)
            return None, None

        # Prepare the link target for makelink().
        if filtered.islnk():
            filtered = _copy.copy(filtered)
            filtered._link_target = _os.path.join(path, filtered.linkname)
        return filtered, unfiltered

    def _extract_one(self, tarinfo, path, set_attrs, numeric_owner,
                     filter_function=None):
        """Extract from filtered tarinfo to disk.

        filter_function is only used when extracting a *different*
        member (e.g. as fallback to creating a symlink)
        """
        self._check("r")

        try:
            self._extract_member(tarinfo, _os.path.join(path, tarinfo.name),
                                 set_attrs=set_attrs,
                                 numeric_owner=numeric_owner,
                                 filter_function=filter_function,
                                 extraction_root=path)
        except (OSError, UnicodeEncodeError) as e:
            self._handle_fatal_error(e)
        except ExtractError as e:
            self._handle_nonfatal_error(e)

    def extractall(self, path=".", members=None, *, numeric_owner=False,
                   filter=None):
        """Extract all members from the archive to the current working
           directory and set owner, modification time and permissions on
           directories afterwards. 'path' specifies a different directory
           to extract to. 'members' is optional and must be a subset of the
           list returned by getmembers(). If 'numeric_owner' is True, only
           the numbers for user/group names are used and not the names.

           The 'filter' function will be called on each member just
           before extraction.
           It can return a changed TarInfo or None to skip the member.
           String names of common filters are accepted.
        """
        directories = []

        filter_function = self._get_filter_function(filter)
        if members is None:
            members = self

        for member in members:
            tarinfo, unfiltered = self._get_extract_tarinfo(
                member, filter_function, path)
            if tarinfo is None:
                continue
            if tarinfo.isdir():
                # For directories, delay setting attributes until later,
                # since permissions can interfere with extraction and
                # extracting contents can reset mtime.
                directories.append(unfiltered)
            self._extract_one(tarinfo, path, set_attrs=not tarinfo.isdir(),
                              numeric_owner=numeric_owner,
                              filter_function=filter_function)

        # Reverse sort directories.
        directories.sort(key=lambda a: a.name, reverse=True)

        # Set correct owner, mtime and filemode on directories.
        for unfiltered in directories:
            try:
                # Need to re-apply any filter, to take the *current*
                # filesystem state into account.
                try:
                    tarinfo = filter_function(unfiltered, path)
                except _FILTER_ERRORS as exc:
                    self._log_no_directory_fixup(unfiltered, repr(exc))
                    continue
                if tarinfo is None:
                    self._log_no_directory_fixup(unfiltered,
                                                 "excluded by filter")
                    continue
                dirpath = _os.path.join(path, tarinfo.name)
                try:
                    lstat = _os.lstat(dirpath)
                except FileNotFoundError:
                    self._log_no_directory_fixup(tarinfo, "missing")
                    continue
                if not _stat.S_ISDIR(lstat.st_mode):
                    # This is no longer a directory; presumably a later
                    # member overwrote the entry.
                    self._log_no_directory_fixup(tarinfo, "not a directory")
                    continue
                self.chown(tarinfo, dirpath, numeric_owner=numeric_owner)
                self.utime(tarinfo, dirpath)
                self.chmod(tarinfo, dirpath)
            except ExtractError as e:
                self._handle_nonfatal_error(e)
            except OSError as e:
                self._handle_fatal_error(e)

    def extract(self, member, path="", set_attrs=True, *, numeric_owner=False,
                filter=None):
        """Extract a member from the archive to the current working directory,
           using its full name. Its file information is extracted as
           accurately as possible. 'member' may be a filename or a TarInfo
           object. You can specify a different directory using 'path'. File
           attributes (owner, mtime, mode) are set unless 'set_attrs' is
           False. If 'numeric_owner' is True, only the numbers for user/group
           names are used and not the names.

           The 'filter' function will be called before extraction.
           It can return a changed TarInfo or None to skip the member.
           String names of common filters are accepted.
        """
        filter_function = self._get_filter_function(filter)
        tarinfo, unfiltered = self._get_extract_tarinfo(
            member, filter_function, path)
        if tarinfo is not None:
            self._extract_one(tarinfo, path, set_attrs, numeric_owner)

    def _extract_member(self, tarinfo, targetpath, set_attrs=True,
                        numeric_owner=False, *, filter_function=None,
                        extraction_root=None):
        """Extract the filtered TarInfo object tarinfo to a physical
           file called targetpath.

           filter_function is only used when extracting a *different*
           member (e.g. as fallback to creating a symlink)
        """
        # Fetch the TarInfo object for the given name
        # and build the destination pathname, replacing
        # forward slashes to platform specific separators.
        targetpath = targetpath.rstrip("/")
        targetpath = targetpath.replace("/", _os.sep)

        # Create all upper directories.
        upperdirs = _os.path.dirname(targetpath)
        if upperdirs and not _os.path.exists(upperdirs):
            # Create directories that are not part of the archive with
            # default permissions.
            _os.makedirs(upperdirs, exist_ok=True)

        if tarinfo.islnk() or tarinfo.issym():
            self._dbg(1, "%s -> %s" % (tarinfo.name, tarinfo.linkname))
        else:
            self._dbg(1, tarinfo.name)

        if tarinfo.isreg():
            self.makefile(tarinfo, targetpath)
        elif tarinfo.isdir():
            self.makedir(tarinfo, targetpath)
        elif tarinfo.isfifo():
            self.makefifo(tarinfo, targetpath)
        elif tarinfo.ischr() or tarinfo.isblk():
            self.makedev(tarinfo, targetpath)
        elif tarinfo.islnk() or tarinfo.issym():
            self.makelink_with_filter(
                tarinfo, targetpath,
                filter_function=filter_function,
                extraction_root=extraction_root)
        elif tarinfo.type not in tf.SUPPORTED_TYPES:
            self.makeunknown(tarinfo, targetpath)
        else:
            self.makefile(tarinfo, targetpath)

        if set_attrs:
            self.chown(tarinfo, targetpath, numeric_owner)
            if not tarinfo.issym():
                self.chmod(tarinfo, targetpath)
                self.utime(tarinfo, targetpath)

    def makelink_with_filter(self, tarinfo, targetpath,
                             filter_function, extraction_root):
        """Make a (symbolic) link called targetpath. If it cannot be created
          (platform limitation), we try to make a copy of the referenced file
          instead of a link.

          filter_function is only used when extracting a *different*
          member (e.g. as fallback to creating a symlink)
        """
        keyerror_to_extracterror = False
        try:
            # For systems that support symbolic and hard links.
            if tarinfo.issym():
                if _os.path.lexists(targetpath):
                    # Avoid FileExistsError on following os.symlink.
                    _os.unlink(targetpath)
                _os.symlink(tarinfo.linkname, targetpath)
                return
            else:
                if _os.path.exists(tarinfo._link_target):
                    if _os.path.lexists(targetpath):
                        # Avoid FileExistsError on following os.link.
                        _os.unlink(targetpath)
                    _os.link(tarinfo._link_target, targetpath)
                    return
        except symlink_exception:
            keyerror_to_extracterror = True

        try:
            unfiltered = self._find_link_target(tarinfo)
        except KeyError:
            if keyerror_to_extracterror:
                raise ExtractError(
                    "unable to resolve link inside archive") from None
            else:
                raise

        if filter_function is None:
            filtered = unfiltered
        else:
            if extraction_root is None:
                raise ExtractError(
                    "makelink_with_filter: if filter_function is not None, "
                    + "extraction_root must also not be None")
            try:
                filtered = filter_function(unfiltered, extraction_root)
            except _FILTER_ERRORS as cause:
                raise LinkFallbackError(tarinfo, unfiltered.name) from cause
        if filtered is not None:
            self._extract_member(filtered, targetpath,
                                 filter_function=filter_function,
                                 extraction_root=extraction_root)

    # --- attribute setters that tolerate the filter's None attrs ----------
    def makedir(self, tarinfo, targetpath):
        """Make a directory called targetpath.
        """
        try:
            if tarinfo.mode is None:
                # Use the system's default mode
                _os.mkdir(targetpath)
            else:
                # Use a safe mode for the directory, the real mode is set
                # later in _extract_member().
                _os.mkdir(targetpath, 0o700)
        except FileExistsError:
            if not _os.path.isdir(targetpath):
                raise

    def chown(self, tarinfo, targetpath, numeric_owner):
        """Set owner of targetpath according to tarinfo. If 'numeric_owner'
           is True, use .gid/.uid instead of .gname/.uname. If
           'numeric_owner' is False, fall back to .gid/.uid when the search
           based on name fails.
        """
        if hasattr(_os, "geteuid") and _os.geteuid() == 0:
            # We have to be root to do so.
            grp = getattr(tf, "grp", None)
            pwd = getattr(tf, "pwd", None)
            g = tarinfo.gid
            u = tarinfo.uid
            if not numeric_owner:
                try:
                    if grp and tarinfo.gname:
                        g = grp.getgrnam(tarinfo.gname)[2]
                except KeyError:
                    pass
                try:
                    if pwd and tarinfo.uname:
                        u = pwd.getpwnam(tarinfo.uname)[2]
                except KeyError:
                    pass
            if g is None:
                g = -1
            if u is None:
                u = -1
            try:
                if tarinfo.issym() and hasattr(_os, "lchown"):
                    _os.lchown(targetpath, u, g)
                else:
                    _os.chown(targetpath, u, g)
            except (OSError, OverflowError) as e:
                # OverflowError can be raised if an ID doesn't fit in 'id_t'
                raise ExtractError("could not change owner") from e

    def chmod(self, tarinfo, targetpath):
        """Set file permissions of targetpath according to tarinfo.
        """
        if tarinfo.mode is None:
            return
        try:
            _os.chmod(targetpath, tarinfo.mode)
        except OSError as e:
            raise ExtractError("could not change mode") from e

    def utime(self, tarinfo, targetpath):
        """Set modification time of targetpath according to tarinfo.
        """
        mtime = tarinfo.mtime
        if mtime is None:
            return
        if not hasattr(_os, "utime"):
            return
        try:
            _os.utime(targetpath, (mtime, mtime))
        except OSError as e:
            raise ExtractError("could not change modification time") from e

    # --- install everything -------------------------------------------------
    TarInfo.replace = _tarinfo_replace

    TarFile._get_filter_function = _get_filter_function
    TarFile._handle_fatal_error = _handle_fatal_error
    TarFile._handle_nonfatal_error = _handle_nonfatal_error
    TarFile._log_no_directory_fixup = _log_no_directory_fixup
    TarFile._get_extract_tarinfo = _get_extract_tarinfo
    TarFile._extract_one = _extract_one
    TarFile.extractall = extractall
    TarFile.extract = extract
    TarFile._extract_member = _extract_member
    TarFile.makelink_with_filter = makelink_with_filter
    TarFile.makedir = makedir
    TarFile.chown = chown
    TarFile.chmod = chmod
    TarFile.utime = utime

    tf.FilterError = FilterError
    tf.AbsolutePathError = AbsolutePathError
    tf.OutsideDestinationError = OutsideDestinationError
    tf.SpecialFileError = SpecialFileError
    tf.AbsoluteLinkError = AbsoluteLinkError
    tf.LinkOutsideDestinationError = LinkOutsideDestinationError
    tf.LinkFallbackError = LinkFallbackError
    tf.fully_trusted_filter = fully_trusted_filter
    tf.tar_filter = tar_filter
    tf.data_filter = data_filter
    tf._get_filtered_attrs = _get_filtered_attrs
    tf._NAMED_FILTERS = {
        "fully_trusted": fully_trusted_filter,
        "tar": tar_filter,
        "data": data_filter,
    }
