from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class LinkPolicy(IntEnum):
    DISCARD = -1
    LINK_ONE = 1
    LINK_ANY = 2


@dataclass
class FrontStitchSample:
    """A stitch sample carried along the advancing knitting front."""

    chain_t: float
    link_policy: LinkPolicy
