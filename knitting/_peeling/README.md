# Internal front peeling core

This package contains the standalone pipeline's private mutable
harmonic-guided trim-and-peel implementation and graph tracer.

Open-surface adaptations:

- an explicitly selected minimum boundary chain is the initial active front;
- raw open offset contours stay open instead of being closed along the physical boundary;
- cyclic stitch-link rolling is enabled only when both matched courses are closed;
- peeling stops before a candidate front whose harmonic time no longer advances.

Public callers should use `knitting.peel`; this package is an implementation
detail and is not part of the stable API.
