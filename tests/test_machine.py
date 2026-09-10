from typing import Self, Type
from datetime import datetime
import time
from enum import Enum, auto

import pytest
from stateflow import StateMachine, StateMachineTransitionArg, DecisionArg


class SubTrigger(Enum):
    GoToA = auto()
    goToC = auto()


class TestStatemachine:
    class State(Enum):
        A = auto()
        B = auto()
        C = auto()

    class Trigger(Enum):
        AtoCviaB = auto()
        AtoB = auto()
        BtoC = auto()
        CtoA = auto()

    @pytest.fixture()
    def state_machine(self):
        sm = StateMachine[self.State, self.Trigger](initial_state=self.State.A)
        sm.permit(self.State.A, self.State.B, self.Trigger.AtoB)
        sm.permit(self.State.B, self.State.C, self.Trigger.BtoC)
        sm.permit(self.State.C, self.State.A, self.Trigger.CtoA)
        return sm

    def test_legal_transitions(self, state_machine: StateMachine[State, Trigger]):
        assert state_machine.state == self.State.A
        assert state_machine.fire(self.Trigger.AtoB)
        assert state_machine.state == self.State.B
        assert state_machine.fire(self.Trigger.BtoC)
        assert self.State.C == state_machine.state
        assert state_machine.fire(self.Trigger.CtoA)
        assert self.State.A == state_machine.state

    def test_illegal_transitions(self, state_machine: StateMachine):
        assert self.State.A == state_machine.state
        assert not state_machine.fire(self.Trigger.CtoA)

    def test_subscription(self, state_machine: StateMachine[State, Trigger]):
        times = {}

        def on_exit(arg: StateMachineTransitionArg[self.State, self.Trigger]):
            assert arg.old_state == self.State.A
            assert arg.new_state == self.State.B
            assert arg.trigger == self.Trigger.AtoB
            assert not arg.passing_by
            times["exit"] = datetime.now()
            time.sleep(1)

        def on_enter(arg: StateMachineTransitionArg[self.State, self.Trigger]):
            assert arg.old_state == self.State.A
            assert arg.new_state == self.State.B
            assert arg.trigger == self.Trigger.AtoB
            assert not arg.passing_by
            times["enter"] = datetime.now()

        state_machine.subscribe_on_exit(self.State.A, on_exit)

        state_machine.subscribe_on_enter(self.State.B, on_enter)
        state_machine.fire(self.Trigger.AtoB)
        assert len(times) == 2
        assert times["exit"] < times["enter"]

    def test_intermediate_transition(self, state_machine):
        state_machine.permit(
            self.State.A, self.State.C, self.Trigger.AtoCviaB, [self.State.B]
        )
        trigger_dict = {self.State.A: [], self.State.B: [], self.State.C: []}

        def on_exit_A(arg: StateMachineTransitionArg[self.State, self.Trigger]):
            trigger_dict[arg.old_state] += ["exit"]
            assert not arg.passing_by
            assert arg.old_state == self.State.A
            assert arg.new_state == self.State.C

        def on_enter_B(arg: StateMachineTransitionArg[self.State, self.Trigger]):
            assert arg.child_transition is not None
            trigger_dict[arg.child_transition.new_state] += ["enter"]
            assert arg.passing_by
            assert arg.old_state == self.State.A
            assert arg.new_state == self.State.C

            assert arg.child_transition.old_state == self.State.A
            assert arg.child_transition.new_state == self.State.B
            assert arg.child_transition.trigger is None

        def on_exit_B(arg: StateMachineTransitionArg[self.State, self.Trigger]):
            assert arg.child_transition is not None
            trigger_dict[arg.child_transition.old_state] += ["exit"]
            assert arg.passing_by
            assert arg.old_state == self.State.A
            assert arg.new_state == self.State.C
            assert arg.child_transition.old_state == self.State.B
            assert arg.child_transition.new_state == self.State.C
            assert arg.child_transition.trigger is None

        def on_enter_C(arg: StateMachineTransitionArg[self.State, self.Trigger]):
            trigger_dict[arg.new_state] += ["enter"]
            assert not arg.passing_by
            assert arg.old_state == self.State.A
            assert arg.new_state == self.State.C

        state_machine.subscribe_on_exit(self.State.A, on_exit_A)
        state_machine.subscribe_on_enter(self.State.B, on_enter_B)
        state_machine.subscribe_on_exit(self.State.B, on_exit_B)
        state_machine.subscribe_on_enter(self.State.C, on_enter_C)

        assert state_machine.fire(self.Trigger.AtoCviaB)
        assert state_machine.state == self.State.C

        assert len(trigger_dict[self.State.A]) == 1
        assert trigger_dict[self.State.A][0] == "exit"

        assert len(trigger_dict[self.State.B]) == 2
        assert trigger_dict[self.State.B][0] == "enter"
        assert trigger_dict[self.State.B][1] == "exit"

        assert len(trigger_dict[self.State.C]) == 1
        assert trigger_dict[self.State.C][0] == "enter"

    def test_ignore_transition(self, state_machine):
        state_machine.ignore(self.State.A, trigger=self.Trigger.AtoB)

        assert self.State.A == state_machine.state
        assert not state_machine.fire(self.Trigger.AtoB)
        assert self.State.A == state_machine.state


class TestComplexStatemachine:
    class State(Enum):
        A = auto()
        B = auto()
        C = auto()
        D = auto()
        E = auto()

    class Trigger(Enum):
        AtoC = auto()
        CtoE = auto()
        EtoB = auto()
        DtoA = auto()

    class SubTrigger(Enum):
        to_B = auto()
        to_E = auto()
        to_None = auto()

    def set_return_trigger(self, trig: SubTrigger):
        self._trig = trig

    def get_trig(self):
        self._used += 1
        return self._trig

    @pytest.fixture()
    def state_machine(self):
        state, trigger, subtrigger = self.State, self.Trigger, self.SubTrigger
        sm = StateMachine[state, trigger](initial_state=state.A)
        sm.permit(
            state.A,
            state.C,
            trigger.AtoC,
            detour_list=[
                DecisionArg(
                    decision_dict={
                        subtrigger.to_B: [state.B],
                        subtrigger.to_E: [state.E],
                        subtrigger.to_None: [None],
                    },
                    decision_handler=self.get_trig,
                )
            ],
        )
        self._used = 0
        return sm

    @pytest.mark.parametrize(
        "sub_trig, expected_state",
        [(SubTrigger.to_B, State.B), (SubTrigger.to_E, State.E)],
    )
    def test_detour_decision(self, state_machine, sub_trig, expected_state):
        check_list = []
        state, trigger, subtrigger = self.State, self.Trigger, self.SubTrigger
        self.set_return_trigger(sub_trig)

        def _on_entry(arg: StateMachineTransitionArg[self.State, self.Trigger]):
            assert arg.child_transition is not None
            check_list.append(arg.child_transition.new_state)

        state_machine.subscribe_on_enter(expected_state, _on_entry)

        assert self._used == 0
        assert state_machine.state == state.A
        assert state_machine.fire(trigger.AtoC)
        assert state_machine.state == state.C
        assert len(check_list) == 1
        assert check_list[0] == expected_state
        assert self._used == 1

    def test_abort_option(self, state_machine):
        state, trigger, subtrigger = self.State, self.Trigger, self.SubTrigger

        self.set_return_trigger(subtrigger.to_None)
        assert self._used == 0
        assert state_machine.state == state.A
        assert state_machine.fire(trigger.AtoC)
        assert state_machine.state == state.C
        assert self._used == 1
