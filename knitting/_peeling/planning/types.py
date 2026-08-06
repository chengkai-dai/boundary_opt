from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, TypeAlias

from knitting._peeling.surface import SurfacePoint
from knitting._peeling.planning.stitch_state import FrontStitchSample

ChainIndex = NewType("ChainIndex", int)
StitchIndex = NewType("StitchIndex", int)

ModelChain: TypeAlias = list[SurfacePoint]
ModelChains: TypeAlias = list[ModelChain]
SliceChains: TypeAlias = list[list[int]]
SliceToModelMap: TypeAlias = list[SurfacePoint]
FrontStitches: TypeAlias = list[list[FrontStitchSample]]
FrontGraphVertices: TypeAlias = list[list[int]]


@dataclass(frozen=True)
class ActiveFront:
    model_chains: ModelChains
    stitches: FrontStitches
    graph_vertices: FrontGraphVertices

    def __post_init__(self) -> None:
        if len(self.model_chains) != len(self.stitches):
            raise ValueError("front model chains and stitches must align")
        if len(self.stitches) != len(self.graph_vertices):
            raise ValueError("front stitches and graph vertices must align")
        for stitches, refs in zip(self.stitches, self.graph_vertices, strict=True):
            if len(stitches) != len(refs):
                raise ValueError("front stitch samples and graph vertices must align")


@dataclass(frozen=True)
class StitchRef:
    chain_index: ChainIndex
    stitch_index: StitchIndex

    @classmethod
    def of(cls, chain: int, stitch: int) -> StitchRef:
        return cls(ChainIndex(chain), StitchIndex(stitch))


@dataclass(frozen=True)
class StitchLink:
    front_ref: StitchRef
    next_ref: StitchRef


@dataclass(frozen=True)
class LinkChainsResult:
    next_stitches: FrontStitches
    links: list[StitchLink]


@dataclass(frozen=True)
class BuildNextFrontResult:
    front: ActiveFront

    @property
    def front_chains(self) -> ModelChains:
        return self.front.model_chains

    @property
    def front_stitches(self) -> FrontStitches:
        return self.front.stitches
