from collections import defaultdict
class CrossLinks:

    def __init__(self):
        # defaultdict (e.g. with set)
        self.blocks = defaultdict(set)
        self.blocked_by = defaultdict(set)

    def add_blocker(self, pt, edge_id):
        self.blocks[pt].add(edge_id)
        self.blocked_by[edge_id].add(pt)
        # if pt in self.blocks:
        #     self.blocks[pt].append(edge_id)
        # else:
        #     self.blocks[pt] = [edge_id]
        # if edge_id in self.blocked_by:
        #     self.blocked_by[edge_id].append(pt)
        # else:
        #     self.blocked_by[edge_id] = [pt]

    def remove_blocker(self, pt):
        unblocked_edge_ids = []
        if pt in self.blocks:
            for edge_id in self.blocks[pt]:
                try:
                    self.blocked_by[edge_id].remove(pt) # hmmm, why (because unblocked not simplified again)?
                except KeyError:
                    print(f"edge id: {edge_id}, pt: {pt}")
                if len(self.blocked_by[edge_id]) == 0:
                    del self.blocked_by[edge_id]
                    unblocked_edge_ids.append(edge_id)
            del self.blocks[pt]
        # else:
        #     raise ValueError( f"pt {pt} -- [as blocker] not there" )
        return unblocked_edge_ids


def test():
    cross = CrossLinks()
    cross.add_blocker((5,5), 1)                 # vertex from other line blocks edge being simplified

    print(cross.blocks)
    print(cross.blocked_by)

    unblocked = cross.remove_blocker((5,5))     # remove the vertex, and see if edges have become free

    print(unblocked)

    unblocked = cross.remove_blocker((6, 6))    # remove a vertex, and see if edges have become free

    print(unblocked)


if __name__ == "__main__":
    test()
