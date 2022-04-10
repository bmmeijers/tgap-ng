from enum import IntEnum


class Operations(IntEnum):
    Keep = 0 # do nothing to this segment; PositionRelevant Point not important
    Ignore = 1 # this will be skipped when creating a new geometry
    Remove = 2 # completely remove it; PositionRelevant Point not important
    Extend = 3 # change the relevant point by moving it back/forth on the original line
    KeepRefPtOnly = 4 # Here the relevant point will become anchor location to determine the perpendicular to another segment

class PositionRelevantPt(IntEnum):
    Start = 0
    End = 1
    NA = 2 # Not-applicable

class Direction(IntEnum):
    Left = 0
    Right = 0