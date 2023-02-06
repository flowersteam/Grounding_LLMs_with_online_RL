"""
Levels described in BabyAI's ICLR 2019 submission.

Note: ELLA Custom Levels are included in this file as well.
"""

import gym
from .verifier import *
from .levelgen import *


class Level_GoToRedBallGrey(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    The distractors are all grey to reduce perceptual complexity.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, 'ball', 'red')
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        for dist in dists:
            dist.color = 'grey'

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBall(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, 'ball', 'red')
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBallNoDists(Level_GoToRedBall):
    """
    Go to the red ball. No distractors present.
    """

    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=0, seed=seed)


class Level_GoToObj(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=1)
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToObjS4(Level_GoToObj):
    def __init__(self, seed=None):
        super().__init__(room_size=4, seed=seed)


class Level_GoToObjS6(Level_GoToObj):
    def __init__(self, seed=None):
        super().__init__(room_size=6, seed=seed)


class Level_GoToLocal(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, num_dists=8, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToLocalS5N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_dists=2, seed=seed)


class Level_GoToLocalS6N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=2, seed=seed)


class Level_GoToLocalS6N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=3, seed=seed)


class Level_GoToLocalS6N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=4, seed=seed)


class Level_GoToLocalS7N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=4, seed=seed)


class Level_GoToLocalS7N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=5, seed=seed)


class Level_GoToLocalS8N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=2, seed=seed)


class Level_GoToLocalS8N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=3, seed=seed)


class Level_GoToLocalS8N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=4, seed=seed)


class Level_GoToLocalS8N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=5, seed=seed)


class Level_GoToLocalS8N6(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=6, seed=seed)


class Level_GoToLocalS8N7(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=7, seed=seed)


class Level_GoToLocalS8N16(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=16, seed=seed)


class Level_GoToLocal2(LevelGen):
    """
    GoToLocal twice    
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            force_colors=True,
            seed=seed,
            instr_kinds=['and'],
            action_kinds=['goto'],
            locked_room_prob=0,
            locations=False,
            unblocking=False
        )


class Level_PutNextLocal(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, no distractors
    """

    def __init__(self, room_size=8, num_objs=8, seed=None):
        self.num_objs = num_objs
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PutNextLocalS5N3(Level_PutNextLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_objs=3, seed=seed)


class Level_PutNextLocalS6N4(Level_PutNextLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_objs=4, seed=seed)


class Level_GoTo(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            room_size=8,
            num_rows=3,
            num_cols=3,
            num_dists=18,
            doors_open=False,
            seed=None
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()


class Level_GoToOpen(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(doors_open=True, seed=seed)


class Level_GoToFrench(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            doors_open=True,
            seed=None,
            language='french'
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed,
            language=language
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()


class Level_GoToObjMaze(Level_GoTo):
    """
    Go to an object, the object may be in another room. No distractors.
    """

    def __init__(self, seed=None):
        super().__init__(num_dists=1, doors_open=False, seed=seed)


class Level_GoToObjMazeOpen(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, doors_open=True, seed=seed)


class Level_GoToObjMazeS4R2(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, num_rows=2, num_cols=2, seed=seed)


class Level_GoToObjMazeS4(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, seed=seed)


class Level_GoToObjMazeS5(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=5, seed=seed)


class Level_GoToObjMazeS6(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=6, seed=seed)


class Level_GoToObjMazeS7(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=7, seed=seed)


class Level_GoToImpUnlock(RoomGridLevel):
    """
    Go to an object, which may be in a locked room.
    Competencies: Maze, GoTo, ImpUnlock
    No unblocking.
    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_rows)
        jd = self._rand_int(0, self.num_cols)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i is not id or j is not jd:
                    self.add_distractors(
                        i,
                        j,
                        num_distractors=2,
                        all_unique=False
                    )

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        # Add a single object to the locked room
        # The instruction requires going to an object matching that description
        obj, = self.add_distractors(id, jd, num_distractors=1, all_unique=False)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_Pickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_UnblockPickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room. The path may
    be blocked by one or more obstructors.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=20, all_unique=False)

        # Ensure that at least one object is not reachable without unblocking
        # Note: the selected object will still be reachable most of the time
        if self.check_objs_reachable(raise_exc=False):
            raise RejectSampling('all objects reachable')

        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_Open(RoomGridLevel):
    """
    Open a door, which may be in another room
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()

        # Collect a list of all the doors in the environment
        doors = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)

        door = self._rand_elem(doors)
        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_Unlock(RoomGridLevel):
    """
    Unlock a door.

    Competencies: Maze, Open, Unlock. No unblocking.
    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_rows)
        jd = self._rand_int(0, self.num_cols)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        # With 50% probability, ensure that the locked door is the only
        # door of that color
        if self._rand_bool():
            colors = list(filter(lambda c: c is not door.color, COLOR_NAMES))
            self.connect_all(door_colors=colors)
        else:
            self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i is not id or j is not jd:
                    self.add_distractors(
                        i,
                        j,
                        num_distractors=3,
                        all_unique=False
                    )

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_PutNext(RoomGridLevel):
    """
    Put an object next to another object. Either of these may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)
        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_GoToMedium(LevelGen):
    """
    GoTo, 2 rooms
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['goto'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=2,
            num_dists=8,
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            force_colors=True
        )


class Level_GoToMediumOpen(LevelGen):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            doors_open=True,
            seed=None
    ):
        self.doors_open = doors_open
        super().__init__(
            seed=seed,
            action_kinds=['goto'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=2,
            num_dists=8,
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            force_colors=True
        )

    def gen_mission(self):
        if self._rand_float(0, 1) < self.locked_room_prob:
            self.add_locked_room()

        self.connect_all()

        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is self.locked_room:
                continue
            break

        # If no unblocking required, make sure all objects are
        # reachable without unblocking
        if not self.unblocking:
            self.check_objs_reachable()

        # Generate random instructions
        self.instrs = self.rand_instr(
            action_kinds=self.action_kinds,
            instr_kinds=self.instr_kinds
        )
        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()


class Level_GoToLarge(LevelGen):
    """
    GoTo, 2x2
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['goto'],
            instr_kinds=['action'],
            num_rows=2,
            num_cols=2,
            num_dists=8,
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            force_colors=True
        )


class Level_GoToLargeOpen(LevelGen):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            doors_open=True,
            seed=None
    ):
        self.doors_open = doors_open
        super().__init__(
            seed=seed,
            action_kinds=['goto'],
            instr_kinds=['action'],
            num_rows=2,
            num_cols=2,
            num_dists=8,
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            force_colors=True
        )

    def gen_mission(self):
        if self._rand_float(0, 1) < self.locked_room_prob:
            self.add_locked_room()

        self.connect_all()

        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is self.locked_room:
                continue
            break

        # If no unblocking required, make sure all objects are
        # reachable without unblocking
        if not self.unblocking:
            self.check_objs_reachable()

        # Generate random instructions
        self.instrs = self.rand_instr(
            action_kinds=self.action_kinds,
            instr_kinds=self.instr_kinds
        )
        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()


class Level_PutNextMedium(LevelGen):
    """
    Put an object next to another object. Either of these may be in another room.
    2 rooms
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['putnext'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=2,
            num_dists=8,
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            force_colors=True
        )


class Level_PutNextLarge(LevelGen):
    """
    Put an object next to another object. Either of these may be in another room.
    2x2
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['putnext'],
            instr_kinds=['action'],
            num_rows=2,
            num_cols=2,
            num_dists=8,
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            force_colors=True
        )


class Level_PickupLoc(LevelGen):
    """
    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False
        )


class Level_GoToSeq(LevelGen):
    """
    Sequencing of go-to-object commands.

    Competencies: Maze, GoTo, Seq
    No locked room.
    No locations.
    No unblocking.
    """

    def __init__(
            self,
            room_size=8,
            num_rows=3,
            num_cols=3,
            num_dists=18,
            seed=None
    ):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['goto'],
            locked_room_prob=0,
            locations=False,
            unblocking=False
        )


class Level_GoToSeqS5R2(Level_GoToSeq):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_rows=2, num_cols=2, num_dists=4, seed=seed)


class Level_Synth(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
            self,
            room_size=8,
            num_rows=3,
            num_cols=3,
            num_dists=18,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            instr_kinds=['action'],
            locations=False,
            unblocking=True,
            implicit_unlock=False
        )


class Level_SynthMedium(Level_Synth):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_rows=1,
            num_cols=2,
            num_dists=8
        )


class Level_PickupLocal(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            force_colors=True,
            implicit_unlock=False
        )


class Level_PickupMedium(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=2,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            force_colors=True,
            implicit_unlock=False
        )


class Level_PickupLarge(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
            self,
            room_size=8,
            num_rows=2,
            num_cols=2,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            force_colors=True,
            implicit_unlock=False
        )


class Level_PickupOpenMedium(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=2,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['pickup', 'open'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            force_colors=True,
            implicit_unlock=False
        )


class Level_PickupOpenLarge(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
            self,
            room_size=8,
            num_rows=2,
            num_cols=2,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['pickup', 'open'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            force_colors=True,
            implicit_unlock=False
        )


class Level_PickupOpenPutNextMedium(LevelGen):
    """
    Union of all instructions from PutNext, Open, and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=2,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['putnext', 'pickup', 'open'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            force_colors=True,
            implicit_unlock=False
        )


class Level_PickupOpenPutNextLarge(LevelGen):
    """
    Union of all instructions from PutNext, Open, and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
            self,
            room_size=8,
            num_rows=2,
            num_cols=2,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['putnext', 'pickup', 'open'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            force_colors=True,
            implicit_unlock=False
        )


class Level_UnlockMedium(LevelGen):

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=2,
            num_dists=8,
            seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['open'],
            instr_kinds=['action'],
            locations=False,
            unblocking=False,
            locked_room_prob=1,
            force_colors=True,
            implicit_unlock=True
        )


class Level_SynthS5R2(Level_Synth):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=2,
            num_cols=2,
            num_dists=7,
            seed=seed
        )


class Level_SynthLoc(LevelGen):
    """
    Like Synth, but a significant share of object descriptions involves
    location language like in PickUpLoc. No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            instr_kinds=['action'],
            locations=True,
            unblocking=True,
            implicit_unlock=False
        )


class Level_SynthSeq(LevelGen):
    """
    Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.
    No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            locations=True,
            unblocking=True,
            implicit_unlock=False
        )


class Level_OpenAndPickupMedium(LevelGen):
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            num_rows=1,
            num_cols=2,
            num_dists=5,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            locked_room_prob=0.0,
            instr_kinds=['x_and_y'],
            action_kinds=['open', 'pickup'],
            force_colors=True,
            assert_first=True
        )


class Level_OpenAndPickupLarge(LevelGen):
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            num_rows=2,
            num_cols=2,
            num_dists=5,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            locked_room_prob=0.0,
            instr_kinds=['x_and_y'],
            action_kinds=['open', 'pickup'],
            force_colors=True,
            assert_first=True
        )


class Level_OpenGoToMedium(LevelGen):
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            num_rows=1,
            num_cols=2,
            num_dists=5,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            locked_room_prob=0.0,
            instr_kinds=['action'],
            action_kinds=['goto', 'open'],
            force_colors=True,
            assert_first=True
        )


class Level_SynthThenSynthMedium(LevelGen):
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            num_rows=1,
            num_cols=2,
            num_dists=8,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            locked_room_prob=0.0,
            instr_kinds=['seq1'],
            action_kinds=['open', 'pickup', 'putnext'],
            force_colors=True,
            assert_first=True
        )


class Level_SynthThenSynthLarge(LevelGen):
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            num_rows=2,
            num_cols=2,
            num_dists=8,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            locked_room_prob=0.0,
            instr_kinds=['seq1'],
            action_kinds=['open', 'pickup', 'putnext'],
            force_colors=True,
            assert_first=True
        )


class Level_OpenGoToLarge(LevelGen):
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            num_rows=2,
            num_cols=2,
            num_dists=5,
            locations=False,
            unblocking=False,
            implicit_unlock=False,
            locked_room_prob=0.0,
            instr_kinds=['action'],
            action_kinds=['goto', 'open'],
            force_colors=True,
            assert_first=True
        )


class Level_MiniBossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25
        )


class Level_BossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )


class Level_BossLevelNoUnlock(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            locked_room_prob=0,
            implicit_unlock=False
        )

class Level_PickUpSeqPickUpLocal(LevelGen):
    """
    In order to test generalisation we only give to the agent the instruction:
    seq restricted to pick up A then/before pick up B  with A and B without the following adj-noun pairs:
    - yellow box
    - red door/key
    - green ball
    - grey door
    (for memory issue our agent only used the past 3 observations)

    Competencies: Seq never seen in MixedTrainLocal
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):

        action = 'pick up seq pick up '

        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=[action],
            instr_kinds=['seq1'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):

        mission_accepted = False
        all_objects_reachable = False

        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 2, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_a = self._rand_elem(objs)
            while str(obj_a.type) == 'door':
                obj_a = self._rand_elem(objs)
            instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
            obj_b = self._rand_elem(objs)
            if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                desc = ObjDesc(obj_a.type, obj_a.color)
                objas, poss = desc.find_matching_objs(self)
                if len(objas) < 2:
                    # if obj_a is the only object with this description obj_b has to be different
                    while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                        obj_b = self._rand_elem(objs)
            instr_b = PickupInstr(ObjDesc(obj_b.type, obj_b.color))

            type_instr = self._rand_elem(['Before', 'After'])

            if type_instr == 'Before':
                self.instrs = BeforeInstr(instr_a, instr_b)
            else:
                self.instrs = AfterInstr(instr_b, instr_a)

            mission_accepted = not (self.exclude_substrings())


    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0

# Register the levels in this file
register_levels(__name__, globals())


class Level_PickUpSeqGoToLocal(LevelGen):
    """
    In order to test generalisation we only give to the agent the instruction:
    seq restricted to pick up A then/before go to B  with A and B without the following adj-noun pairs:
    - yellow box
    - red door/key
    - green ball
    - grey door
    (for memory issue our agent only used the past 3 observations)

    Competencies: Seq never seen in MixedTrainLocal
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):

        action = 'pick up seq pick up '

        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=[action],
            instr_kinds=['seq1'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):

        mission_accepted = False
        all_objects_reachable = False

        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 2, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_a = self._rand_elem(objs)
            while str(obj_a.type) == 'door':
                obj_a = self._rand_elem(objs)
            instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
            obj_b = self._rand_elem(objs)
            if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                desc = ObjDesc(obj_a.type, obj_a.color)
                objas, poss = desc.find_matching_objs(self)
                if len(objas) < 2:
                    # if obj_a is the only object with this description obj_b has to be different
                    while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                        obj_b = self._rand_elem(objs)
            instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

            type_instr = self._rand_elem(['Before', 'After'])

            if type_instr == 'Before':
                self.instrs = BeforeInstr(instr_a, instr_b)
            else:
                self.instrs = AfterInstr(instr_b, instr_a)

            mission_accepted = not (self.exclude_substrings())


    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0

class Level_PickUpThenGoToLocal(LevelGen):
    """
    In order to test generalisation we only give to the agent the instruction:
    seq restricted to pick up A then go to B  with A and B without the following adj-noun pairs:
    - yellow box
    - red door/key
    - green ball
    - grey door
    (for memory issue our agent only used the past 3 observations)

    Competencies: Seq never seen in MixedTrainLocal
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):

        action = 'pick up seq pick up '

        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=[action],
            instr_kinds=['seq1'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):

        mission_accepted = False
        all_objects_reachable = False

        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 2, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_a = self._rand_elem(objs)
            while str(obj_a.type) == 'door':
                obj_a = self._rand_elem(objs)
            instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
            obj_b = self._rand_elem(objs)
            if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                desc = ObjDesc(obj_a.type, obj_a.color)
                objas, poss = desc.find_matching_objs(self)
                if len(objas) < 2:
                    # if obj_a is the only object with this description obj_b has to be different
                    while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                        obj_b = self._rand_elem(objs)
            instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

            self.instrs = BeforeInstr(instr_a, instr_b)

            mission_accepted = not (self.exclude_substrings())


    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0

class Level_GoToAfterPickUpLocal(LevelGen):
    """
    In order to test generalisation we only give to the agent the instruction:
    seq restricted to go to B after pickup A  with A and B without the following adj-noun pairs:
    - yellow box
    - red door/key
    - green ball
    - grey door
    (for memory issue our agent only used the past 3 observations)

    Competencies: Seq never seen in MixedTrainLocal
    """

    def __init__(
            self,
            room_size=8,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            seed=None
    ):

        action = 'pick up seq pick up '

        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=[action],
            instr_kinds=['seq1'],
            locations=False,
            unblocking=False,
            implicit_unlock=False
        )

    def gen_mission(self):

        mission_accepted = False
        all_objects_reachable = False

        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists + 2, all_unique=False)
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_a = self._rand_elem(objs)
            while str(obj_a.type) == 'door':
                obj_a = self._rand_elem(objs)
            instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
            obj_b = self._rand_elem(objs)
            if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                desc = ObjDesc(obj_a.type, obj_a.color)
                objas, poss = desc.find_matching_objs(self)
                if len(objas) < 2:
                    # if obj_a is the only object with this description obj_b has to be different
                    while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                        obj_b = self._rand_elem(objs)
            instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

            self.instrs = AfterInstr(instr_b, instr_a)

            mission_accepted = not (self.exclude_substrings())


    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = ["yellow box", "red key", "red door", "green ball", "grey door"]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.agent_dir = 0

# Register the levels in this file
register_levels(__name__, globals())
