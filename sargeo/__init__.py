"""sargeo: Sentinel-1 terrain correction from scratch."""
from __future__ import annotations

import warnings


def _silence_upstream_rasterio_warning() -> None:
    """Mute a DeprecationWarning that comes from rasterio, not from this code.

    rasterio <= 1.5.1 assigns ``ndarray.shape`` directly, which NumPy 2.5
    deprecated ("Setting the shape on a NumPy array has been deprecated").
    The warning is raised with stacklevel pointing at *our* ``ds.read(...)``
    lines, so it looks like our bug while the cause is inside rasterio.
    It is fixed in rasterio 1.5.2.

    Silenced only for that exact message, only on the version pair that
    produces it, so the filter retires itself once rasterio is upgraded.
    """
    try:
        import numpy as np
        import rasterio
    except ImportError:  # pragma: no cover - rasterio is a hard dependency at runtime
        return

    def ver(s: str) -> tuple[int, ...]:
        out = []
        for part in s.split(".")[:3]:
            digits = "".join(ch for ch in part if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return tuple(out)

    if ver(np.__version__) >= (2, 5) and ver(rasterio.__version__) < (1, 5, 2):
        warnings.filterwarnings(
            "ignore",
            message=r"Setting the shape on a NumPy array",
            category=DeprecationWarning,
        )


_silence_upstream_rasterio_warning()