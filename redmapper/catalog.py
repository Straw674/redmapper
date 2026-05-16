"""Generic object and catalog classes for redmapper.

This module provides base classes for catalogs and entries, transitioning
to a functional style using astropy.table.Table.
"""

import fitsio
import numpy as np
from astropy.table import Table, Row


class Entry(Row):
    """
    An Entry is a single row of a Catalog, based on astropy Row.
    """

    def __init__(self, *args, **kwargs):
        """
        Instantiate an Entry.
        Supports standard Row(table, index) and legacy Entry(array).
        """
        if len(args) > 0 and isinstance(args[0], Table):
            # When called via Table.__getitem__, kwargs are usually empty
            super().__init__(*args, **kwargs)
        else:
            # Legacy initialization: Entry(array)
            array = args[0] if len(args) > 0 else kwargs.get("array")
            if array is not None:
                if array.size != 1:
                    raise ValueError(
                        "Input array must have length one for Entry. Use Catalog for multiple rows."
                    )
                t = Catalog(array)
                super().__init__(t, 0)
            else:
                raise TypeError(
                    "Entry requires either a Table and index, or a single-row array."
                )

    @property
    def _ndarray(self):
        """Return the entry as a numpy record (compatibility property)."""
        return self.table.as_array()[self.index]

    def as_array(self):
        """Return the entry as a 1-row numpy structured array (compatibility method)."""
        return self.table.as_array()[self.index : self.index + 1]

    def __getattr__(self, attr):
        """Allow attribute access to columns for backward compatibility."""
        if attr.startswith("_") or attr in ("table", "_index", "colnames"):
            return super().__getattribute__(attr)

        # Case-insensitive column access
        try:
            if attr in self.colnames:
                return self[attr]
            colnames_lower = [c.lower() for c in self.colnames]
            if attr.lower() in colnames_lower:
                return self[self.colnames[colnames_lower.index(attr.lower())]]
        except (AttributeError, TypeError):
            pass

        return super().__getattribute__(attr)

    def __setattr__(self, attr, val):
        """Allow setting column values via attribute access for backward compatibility."""
        if attr.startswith("_") or attr in ("table", "_index"):
            super().__setattr__(attr, val)
        else:
            try:
                colnames_lower = [c.lower() for c in self.colnames]
                if attr.lower() in colnames_lower:
                    self[self.colnames[colnames_lower.index(attr.lower())]] = val
                    return
            except (AttributeError, TypeError):
                pass
            super().__setattr__(attr, val)

    @property
    def dtype(self):
        """Return the numpy dtype associated with the Entry."""
        return self.table.dtype

    def add_fields(self, newdtype):
        """Add new fields to the parent table."""
        self.table.add_fields(newdtype)

    @classmethod
    def from_fits_file(cls, filename, ext=1, rows=None):
        """Construct an Entry from a fits file."""
        array = fitsio.read(filename, ext=ext, rows=rows, lower=True, trim_strings=True)
        if array.size != 1:
            raise ValueError(
                "Input array must have length one for Entry. Use Catalog for multiple rows."
            )
        t = Catalog(array)
        return t[0]

    @classmethod
    def from_fits_ext(cls, fits_ext):
        """Construct an Entry from a fitsio fits extension."""
        array = fits_ext.read(upper=True)
        if array.size != 1:
            raise ValueError("Input array must have length one for Entry.")
        t = Catalog(array)
        return t[0]

    def to_fits_file(self, filename, clobber=False, header=None, extname=None):
        """Save Entry to a fits file."""
        self.table[self.index : self.index + 1].to_fits_file(
            filename, clobber=clobber, header=header, extname=extname
        )


class Catalog(Table):
    """
    A Catalog is a collection of Entry objects, based on astropy Table.
    """

    _RowClass = Entry
    RowClass = Entry

    def __init__(self, *args, **kwargs):
        """Instantiate a Catalog."""
        # Extract Table-specific kwargs if any
        table_kwargs = {}
        for key in [
            "data",
            "masked",
            "names",
            "dtype",
            "meta",
            "copy",
            "rows",
            "copy_indices",
            "units",
            "descriptions",
        ]:
            if key in kwargs:
                table_kwargs[key] = kwargs.pop(key)

        super().__init__(*args, **table_kwargs)
        self._RowClass = Entry
        self.RowClass = Entry
        # Ensure all columns are lowercase for consistent access
        for col in self.colnames:
            if col != col.lower():
                self.rename_column(col, col.lower())

    @property
    def _ndarray(self):
        """Return the catalog as a numpy structured array (compatibility property)."""
        return self.as_array()

    @property
    def size(self):
        """Return the number of entries in the catalog."""
        return len(self)

    def __getitem__(self, item):
        """Return an Entry for an integer index, or a new Catalog for a slice."""
        if isinstance(item, (int, np.integer)):
            return self.RowClass(self, item)
        res = super().__getitem__(item)
        if isinstance(res, Table):
            new_cat = type(self)(res)
            # Copy over extra attributes that are not standard Table attributes
            # We skip internal attributes (starting with _)
            # This is important for ClusterCatalog and GalaxyCatalog state propagation.
            table_attrs = set(
                [
                    "columns",
                    "formatter",
                    "primary_key",
                    "data",
                    "masked",
                    "names",
                    "dtype",
                    "meta",
                    "copy",
                    "rows",
                    "copy_indices",
                    "units",
                    "descriptions",
                    "RowClass",
                    "_RowClass",
                    "_column_class",
                    "pprint_exclude_names",
                    "pprint_include_names",
                ]
            )
            for key, val in self.__dict__.items():
                if key not in table_attrs and not key.startswith("_"):
                    setattr(new_cat, key, val)
            return new_cat
        return res

    def __iter__(self):
        """Iterate over the catalog, returning Entry objects."""
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, attr):
        """Allow attribute access to columns for backward compatibility."""
        if attr.startswith("_") or attr in (
            "columns",
            "colnames",
            "meta",
            "primary_key",
            "indices",
            "masked",
            "dtype",
            "shape",
            "size",
        ):
            return super().__getattribute__(attr)

        try:
            if attr in self.colnames:
                return self[attr]
            # Case-insensitive column access
            colnames_lower = [c.lower() for c in self.colnames]
            if attr.lower() in colnames_lower:
                return self[self.colnames[colnames_lower.index(attr.lower())]]
        except (AttributeError, TypeError):
            pass

        return super().__getattribute__(attr)

    def __setattr__(self, attr, val):
        """Allow setting column values via attribute access for backward compatibility."""
        if attr.startswith("_") or attr in (
            "RowClass",
            "_RowClass",
            "_column_class",
            "columns",
            "colnames",
            "meta",
            "primary_key",
            "indices",
            "masked",
        ):
            super().__setattr__(attr, val)
            return

        try:
            if attr in self.colnames:
                self[attr] = val
                return
            # Case-insensitive
            colnames_lower = [c.lower() for c in self.colnames]
            if attr.lower() in colnames_lower:
                self[self.colnames[colnames_lower.index(attr.lower())]] = val
                return
        except (AttributeError, TypeError):
            pass

        super().__setattr__(attr, val)

    def append(self, other):
        """Append another catalog or array (compatibility method)."""
        if isinstance(other, Table):
            other_tab = other
        else:
            other_tab = Table(other)

        for row in other_tab:
            self.add_row(row)

    def extend(self, n_new):
        """Extend catalog with zero-filled rows (compatibility method)."""
        temp = Table(np.zeros(n_new, dtype=self.dtype))
        for row in temp:
            self.add_row(row)

    def add_fields(self, newdtype):
        """Add new fields (compatibility method)."""
        for item in newdtype:
            name = item[0].lower()
            dtype = item[1]
            if name not in self.colnames:
                if len(item) == 3:
                    # Array field
                    self[name] = np.zeros((len(self), item[2]), dtype=dtype)
                else:
                    self[name] = np.zeros(len(self), dtype=dtype)

    @classmethod
    def from_fits_file(cls, filename, ext=1, rows=None):
        """Construct a Catalog from a fits file."""
        array = fitsio.read(filename, ext=ext, rows=rows, lower=True, trim_strings=True)
        return cls(array)

    @classmethod
    def from_fits_ext(cls, fits_ext):
        """Construct a Catalog from a fitsio fits extension."""
        array = fits_ext.read(upper=True)
        return cls(array)

    @classmethod
    def zeros(cls, size, dtype):
        """Construct a Catalog filled with all 0s."""
        return cls(np.zeros(size, dtype=dtype))

    def to_fits_file(
        self, filename, clobber=False, header=None, extname=None, indices=None
    ):
        """Save Catalog to a fits file."""
        data = self.as_array()
        if indices is not None:
            data = data[indices]

        fitsio.write(filename, data, clobber=clobber, header=header, extname=extname)
