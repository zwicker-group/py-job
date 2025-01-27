"""Defines a class storing data in various storages.

Requires the optional :mod:`zarr` module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Collection, Sequence, Union

import numpy as np
import zarr
from numpy.typing import ArrayLike, DTypeLike

try:
    # try import path from zarr version 3
    from zarr.abc.store import Store

    is_zarr2 = False

except ImportError:
    # try import path from zarr version 2
    from zarr._storage.store import Store

    is_zarr2 = True

from ..access_modes import ModeType
from ..attributes import AttrsLike
from ..base import StorageBase

zarrElement = Union[zarr.Group, zarr.Array]


class ZarrStorage(StorageBase):
    """Storage that stores data in an zarr file or database."""

    extensions = ["zarr", "zip", "sqldb", "lmdb"]

    def __init__(self, store_or_path: str | Path | Store, *, mode: ModeType = "read"):
        """
        Args:
            store_or_path (str or :class:`~pathlib.Path` or :class:`~zarr._storage.store.Store`):
                File path to the file/folder or a :mod:`zarr` Store
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations.
        """
        super().__init__(mode=mode)

        if isinstance(store_or_path, (str, Path)):
            # open zarr storage on file system
            self._close = True
            path = Path(store_or_path)

            if self.mode.file_mode == "x" and path.exists():
                raise FileExistsError(f"File `{path}` already exists")

            if path.suffix in {"", ".zarr"}:
                # path seems to be a directory or a zarr direction => DirectoryStore
                if path.is_dir() and self.mode.file_mode == "w":
                    self._logger.info("Delete directory `{%s}`", path)
                    shutil.rmtree(path)  # remove the directory to reinstate it
                if self.mode.file_mode == "r":
                    self._logger.info("DirectoryStore is always opened writable")

                if is_zarr2:
                    self._store = zarr.DirectoryStore(path)
                else:
                    self._store = zarr.LocalStore(path)

            elif path.suffix == ".zip":
                # create a ZipStore
                file_mode = self.mode.file_mode
                if path.exists() and file_mode == "w":
                    self._logger.info("Delete file `%s`", path)
                    path.unlink()
                self._store = zarr.storage.ZipStore(path, mode=file_mode)

            elif is_zarr2 and path.suffix == ".sqldb":
                # create a SQLiteStore
                if self.mode.file_mode == "w" and path.exists():
                    self._logger.info("Delete file `%s`", path)
                    path.unlink()
                self._store = zarr.SQLiteStore(path)

        elif isinstance(store_or_path, Store):
            # use already opened zarr storage
            self._close = False
            self._store = store_or_path

        else:
            raise TypeError(f"Unknown store `{store_or_path}`")

        self._root = zarr.group(store=self._store, overwrite=False)

    @property
    def can_update(self) -> bool:
        """bool: indicates whether the storage supports updating items"""
        return not isinstance(self._store, zarr.storage.ZipStore)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._root.store}, mode="{self.mode.name}")'

    def close(self) -> None:
        if self._close:
            self._store.close()
        self._root = None
        super().close()

    def _get_parent(self, loc: Sequence[str]) -> tuple[zarr.Group, str]:
        """Get the parent group for a particular location.

        Args:
            loc (list of str):
                The location in the storage where the group will be created

        Returns:
            (group, str):
                A tuple consisting of the parent group and the name of the current item
        """
        try:
            path, name = loc[:-1], loc[-1]
        except IndexError as err:
            raise KeyError(f"Location `/{'/'.join(loc)}` has no parent") from err

        parent = self._root
        for part in path:
            try:
                parent = parent[part]
            except KeyError:
                parent = parent.create_group(part, overwrite=False)

        return parent, name

    def __getitem__(self, loc: Sequence[str]) -> Any:
        if len(loc) == 0:
            return self._root
        else:
            parent, name = self._get_parent(loc)
            return parent[name]

    def keys(self, loc: Sequence[str] | None = None) -> Collection[str]:
        if loc:
            return self[loc].keys()  # type: ignore
        else:
            return self._root.keys()  # type: ignore

    def is_group(self, loc: Sequence[str], *, ignore_cls: bool = False) -> bool:
        return isinstance(self[loc], zarr.Group)

    def _create_group(self, loc: Sequence[str]) -> None:
        parent, name = self._get_parent(loc)
        parent.create_group(name)

    def _read_attrs(self, loc: Sequence[str]) -> AttrsLike:
        return self[loc].attrs  # type: ignore

    def _write_attr(self, loc: Sequence[str], name: str, value: str) -> None:
        self[loc].attrs[name] = value

    def _read_array(
        self, loc: Sequence[str], *, copy: bool, index: int | None = None
    ) -> np.ndarray:
        arr_like = self[loc]

        if not isinstance(arr_like, zarr.Array):
            raise RuntimeError(
                f"Found {arr_like.__class__} at location `/{'/'.join(loc)}`"
            )

        is_recarray = arr_like.attrs.get("__recarray__", False)
        if index is not None:
            arr_like = arr_like[index]

        # convert it into the right type
        arr = np.array(arr_like, copy=copy)
        if is_recarray:
            arr = arr.view(np.recarray)

        return arr

    def _write_array(self, loc: Sequence[str], arr: np.ndarray) -> None:
        parent, name = self._get_parent(loc)

        if name in parent:
            # update an existing array assuming it has the same shape. The credentials
            # for this operation need to be checked by the caller!
            parent[name][...] = arr

        else:
            # create a new array element
            if arr.dtype == object:
                el = parent.array(name, arr, object_codec=self.codec, overwrite=True)
            else:
                el = parent.array(name, arr, overwrite=True)

            if isinstance(arr, np.recarray):
                el.attrs["__recarray__"] = True

    def _create_dynamic_array(
        self,
        loc: Sequence[str],
        shape: tuple[int, ...],
        dtype: DTypeLike,
        record_array: bool = False,
    ) -> None:
        parent, name = self._get_parent(loc)
        try:
            if dtype == object:
                element = parent.zeros(
                    name,
                    shape=(0,) + shape,
                    chunks=(1,) + shape,
                    dtype=dtype,
                    object_codec=self.codec,
                    overwrite=True,
                )

            else:
                element = parent.zeros(
                    name, shape=(0,) + shape, chunks=(1,) + shape, dtype=dtype
                )
        except zarr.errors.ContainsArrayError as err:
            raise RuntimeError(f"Array `/{'/'.join(loc)}` already exists") from err
        else:
            if record_array:
                element.attrs["__recarray__"] = True

    def _extend_dynamic_array(self, loc: Sequence[str], data: ArrayLike) -> None:
        arr_obj = self[loc]
        arr_obj.append([data])

    def _read_object(self, loc: Sequence[str]) -> Any:
        return self[loc][0]

    def _write_object(self, loc: Sequence[str], obj: Any) -> None:
        arr: np.ndarray = np.empty(1, dtype=object)  # encode object in an array
        arr[0] = obj
        parent, name = self._get_parent(loc)
        parent.array(name, arr, object_codec=self.codec, overwrite=True)
