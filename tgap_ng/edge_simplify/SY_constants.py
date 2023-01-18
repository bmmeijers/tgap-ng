from enum import IntEnum

class ObjectNotCreatedException(Exception):
    pass

class PreClassificationException(Exception):
    #print("Could not simplify due to the initial segment classifictation being faulty")
    pass

class TopologyIssuesException(Exception):
    #print("Simplification has resulted in Topological Issues")
    pass
    
class VerticalIntersectionNotImplemented(Exception):
    pass

class IntersectionPointSameAsOtherPoint(Exception):
    pass

class Operations(IntEnum):
    Keep = 0 # do nothing to this segment; PositionRelevant Point not important
    Ignore = 1 # this will be skipped when creating a new geometry
    Remove = 2 # completely remove it; PositionRelevant Point not important
    Extend = 3 # change the relevant point by moving it back/forth on the original line
    KeepRefPtOnly = 4 # Here the relevant point will become anchor location to determine the perpendicular to another segment
    KeepWithAnchorPt = 5 # Used to replaced one of the Extends in the old Median simplification case

    ShortInterior = 6 # Used for a short polyline, where an interior segment needs to be simplified

class PositionRelevantPt(IntEnum):
    Start = 0
    End = 1
    NA = 2 # Not-applicable

class Direction(IntEnum):
    Left = 0
    Right = 1

class Transaction(IntEnum):
    Remove = 0
    Add = 1
    UseOnlyForCheck = 2 # will not be removed or added, we just want to save it for checking