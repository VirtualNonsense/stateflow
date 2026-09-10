"""
Module containing the statemachine
"""

import dataclasses
import logging
from enum import Enum
from typing import (
    TypeVar,
    Generic,
    Dict,
    Callable,
    Tuple,
    List,
    Iterable,
    Union,
    Optional,
    Iterator,
)


TRIGGER = TypeVar("TRIGGER", bound=Enum)
STATE = TypeVar("STATE", bound=Enum)

LOGGER = logging.getLogger(__name__)


class StateMachineConfigurationError(Exception):
    """
    This error will be called when the statemachine has been misconfigured
    """


class StateMachineTransitionError(Exception):
    """
    This exception will be called when there is no transition available and the statemachine is in strict mode
    """


class StateMachineBusyError(Exception):
    """
    This error will be risen when a transition has been called while the machine is already busy
    """


@dataclasses.dataclass
class StateMachineTransitionArg(Generic[STATE, TRIGGER]):
    """
    Dataclass containing all information about the current transaction
    """

    old_state: STATE
    """
    Origin state
    """

    new_state: STATE
    """
    Destination state
    """

    trigger: Optional[TRIGGER]
    """
    Trigger used for transition
    """

    child_transition: Optional["StateMachineTransitionArg[STATE, TRIGGER]"] = None

    def __str__(self):
        conti = ""
        if self.child_transition is not None:
            conti = f" Subtransition: {self.child_transition._short_info()}"
        assert self.trigger is not None
        return f"Moving from {self._short_info()} via {self.trigger.name}" + conti

    def _short_info(self) -> str:
        return f"{self.old_state.name} to {self.new_state.name}"

    @property
    def passing_by(self):
        return self.child_transition is not None


TRANSITION_CALLBACK = Callable[[StateMachineTransitionArg[STATE, TRIGGER]], None]

SUBTRIGGER = TypeVar("SUBTRIGGER", bound=Enum)


@dataclasses.dataclass
class DecisionArg(Generic[STATE, SUBTRIGGER]):
    """
    Argument used in complex sub transitions.
    """

    decision_dict: Dict[SUBTRIGGER, List[Union[STATE, "DecisionArg[STATE,SUBTRIGGER]", None]]]
    decision_handler: Callable[[], SUBTRIGGER]


class SubStateDecision(Generic[SUBTRIGGER]):
    """
    Decides the path of a sub transition.
    In hindsight, this class and its implications are a little silly
    """

    def __init__(
        self,
        decision_dict: Dict[
            SUBTRIGGER, List[Union["StateMachineNode", "SubStateDecision"]]
        ],
        decision_handler: Callable[[], SUBTRIGGER],
    ):
        self._decision_dict = decision_dict
        self._handler = decision_handler

    def __call__(self) -> List[Union["StateMachineNode", "SubStateDecision"]]:
        substate = self._handler()
        if substate not in self._decision_dict:
            raise StateMachineConfigurationError(
                f"{substate} is not a valid subtransition."
            )
        return self._decision_dict[substate]


class NodeIterator(Iterator["StateMachineNode"]):
    """Iterator class for Nodes and SubStatesDecisions"""

    def __init__(self, state_list: List[Union["StateMachineNode", SubStateDecision]]):
        self._iter = state_list
        self._iter_counter: int
        self._reset_counter()

    def _reset_counter(self):
        self._iter_counter = -1

    def _overwrite(self, other: List[Union["StateMachineNode", SubStateDecision]]):
        """
        Switches the iteration object and resets the counter.
        Because why not.
        :param other:
        :return:
        """
        self._iter = other
        self._reset_counter()

    def __next__(self):
        self._iter_counter += 1
        if self._iter_counter >= len(self._iter):
            raise StopIteration()
        _next = self._iter[self._iter_counter]
        if isinstance(_next, SubStateDecision):
            self._overwrite(_next())
            return self.__next__()
        return _next


class StateMachineTransition(Generic[STATE, TRIGGER], Iterable["StateMachineNode[STATE, TRIGGER]"]):
    """
    This class represents the way from :class:`StateMachineNode` to another,
    While the origin and target have to be defined, it is possible to add intermediate nodes in between.
    """

    def __init__(
        self,
        trigger: TRIGGER,
        origin_node: "StateMachineNode[STATE, TRIGGER]",
        target_node: "StateMachineNode[STATE, TRIGGER]",
        intermediate_nodes: List[
            Union["StateMachineNode", SubStateDecision]
        ] = [],
    ):
        self.trigger = trigger
        self.origin_node = origin_node
        self.target_node = target_node
        self.intermediate_nodes = intermediate_nodes

    def __len__(self):
        raise NotImplementedError(
            "The length is not deterministic since a decision may be dependent on the state"
        )

    def __iter__(self):
        return NodeIterator(self.intermediate_nodes)

    def __str__(self):
        return f"Transition from {self.origin_node.state.name} to {self.target_node.state.name} via {self.trigger.name}"


class StateMachineNode(Generic[STATE, TRIGGER]):
    """
    Class Representing a state or node within the statemachine graph
    """

    def __init__(
        self,
        state: STATE,
        on_entry: Callable[[StateMachineTransitionArg], None],
        on_exit: Callable[[StateMachineTransitionArg], None],
    ):
        self.state = state
        self._on_entry = on_entry
        self._on_exit = on_exit
        self._ignored_trigger: List[TRIGGER] = []
        self._transitions: Dict[TRIGGER, StateMachineTransition[STATE, TRIGGER]] = {}

    def on_entry(self, arg):
        """triggers the on entry callback"""
        self._on_entry(arg)

    def on_exit(self, arg):
        """triggers the on exit callback"""
        self._on_exit(arg)

    def is_final(self):
        """returns true if the node does not have any way to change"""
        return len(self._transitions) == 0

    def perform_transition(self, trigger: TRIGGER, strict: bool) -> Tuple[bool, STATE]:
        """
        tries to perform all necessary to transition between two states
        :param trigger:
        :param strict:
        :return:
            0: whether the transition was successful
            1: the new state
        """
        if trigger in self._ignored_trigger:
            return False, self.state
        if trigger not in self._transitions:
            if strict:
                raise StateMachineTransitionError(
                    f"No valid transition in {self.state} for {trigger}"
                )
            return False, self.state

        transition = self._transitions[trigger]
        target_node = transition.target_node
        arg = StateMachineTransitionArg(
            old_state=self.state, new_state=target_node.state, trigger=trigger
        )
        self.on_exit(arg)
        last_state = self.state
        transition_iter = transition.__iter__()
        try:
            intermediate_node = transition_iter.__next__()
        except StopIteration:
            intermediate_node = None

        if intermediate_node is not None:
            break_at_the_end = False
            while True:
                try:
                    next_node = transition_iter.__next__()
                except StopIteration:
                    next_node = None

                if next_node is None:
                    next_node = target_node
                    break_at_the_end = True

                arg.child_transition = StateMachineTransitionArg(
                    old_state=last_state,
                    new_state=intermediate_node.state,
                    trigger=None,
                )
                intermediate_node.on_entry(arg)
                next_state = next_node.state
                arg.child_transition = StateMachineTransitionArg(
                    old_state=intermediate_node.state,
                    new_state=next_state,
                    trigger=None,
                )
                intermediate_node.on_exit(arg)
                if break_at_the_end:
                    break
                last_state = intermediate_node.state
                intermediate_node = next_node
        arg.child_transition = None
        target_node.on_entry(arg)
        return True, target_node.state

    def add_transition(self, transition: StateMachineTransition[STATE, TRIGGER]):
        """Adds a transition to the node"""
        trigger = transition.trigger
        if trigger in self._transitions:
            raise StateMachineConfigurationError("transition already defined")
        if trigger in self._ignored_trigger:
            raise StateMachineConfigurationError(
                "adding transition for ignored trigger"
            )
        self._transitions[trigger] = transition

    def ignore_trigger(self, trigger):
        """ignores a certain trigger completely"""
        if trigger in self._ignored_trigger:
            return
        if trigger in self._transitions:
            LOGGER.warning("Ignoring previously permitted transition")
        self._ignored_trigger.append(trigger)


class StateMachine(Generic[STATE, TRIGGER]):
    """
    Simple Statemachine class that uses Enums as state and trigger.
    """

    def __init__(self, initial_state: STATE, strict: bool = False):
        """

        :param initial_state: Active state after constructing the statemachine.
        :param strict: Statemachine will raise :class:`StatemachineTransitionError` when trying to perform permitted
                       transitions
        """
        self._state = initial_state
        self._nodes: Dict[STATE, StateMachineNode[STATE, TRIGGER]] = {}
        self._subjects_on_enter: Dict[STATE, TRANSITION_CALLBACK] = {}
        self._subjects_on_exit: Dict[STATE, TRANSITION_CALLBACK] = {}
        self.strict = strict
        self._in_transition = False

    @property
    def state(self) -> STATE:
        """
        Current state of statemachine
        :return:
        """
        # making sure that it will raise an error when in a state that does not have an associated node
        return self._node.state

    @property
    def _node(self) -> StateMachineNode:
        if not self._nodes:
            raise StateMachineConfigurationError(
                "The state machine has not been configured yet!"
            )
        return self._nodes[self._state]

    def fire(self, trigger: TRIGGER) -> bool:
        """
        Perform transition.
        :param trigger: Either a single or multiple trigger. The later will be fired in order
        :return: True when the transition was successful.
                 Note: When the machine is in strict mode, it will raise an Error instead of returning false.
        :raise: :class:`StateMachineBusyError` when firring while in a transition
        """
        if self._in_transition:
            raise StateMachineBusyError(
                "Tried to fire trigger while already in transition"
            )
        success, self._state = self._node.perform_transition(trigger, self.strict)
        return success

    def _on_entry(self, args: StateMachineTransitionArg):
        state = args.new_state
        if args.child_transition is not None:
            state = args.child_transition.new_state
        if state in self._subjects_on_enter:
            self._subjects_on_enter[state](args)

    def _on_exit(self, args: StateMachineTransitionArg):
        state = args.old_state
        if args.child_transition is not None:
            state = args.child_transition.old_state
        if state in self._subjects_on_exit:
            self._subjects_on_exit[state](args)

    def subscribe_on_enter(
        self,
        state: STATE,
        callback: Callable[[StateMachineTransitionArg[STATE, TRIGGER]], None],
    ) -> None:
        """
        Subscribe to on_enter event.
        It will be fired right after the new state has been set.
        Note: Be aware that it is not recommended to perform use :meth:`StateMachine.fire` within this method!
        :param state:
        :param callback: Method that should be called when the transition is done.
                         It should be able to handle a :class:`StateMachineTransitionArg`.
        :return: :class:`Disposable`
        """
        if state not in self._subjects_on_enter:
            self._subjects_on_enter[state] = callback

    def subscribe_on_exit(
        self,
        state: STATE,
        callback: Callable[[StateMachineTransitionArg[STATE, TRIGGER]], None],
    ) -> None:
        """
        Subscribe to on_exit event.
        It will be fired right before the new state has been set
        Note: Be aware that it is not recommended to perform use :meth:`StateMachine.fire` within this method!
        :param state:
        :param callback: Method that should be called before the transition is done.
                         It should be able to handle a Tuple containing the new state and the trigger that was used.
        :return: :class:`Disposable`
        """
        if state not in self._subjects_on_exit:
            self._subjects_on_exit[state] = callback

    def permit(
        self,
        start: STATE,
        destination: STATE,
        trigger: TRIGGER,
        detour_list: Optional[List[Union[STATE, DecisionArg, None]]] = None,
    ):
        """
        Add a transition from `start` to `destination` via `trigger`
        :raise StatemachineConfigurationError: When trying to permit the same combination of `start` and `trigger` for two
                                               different destinations.
                                               See :class:`StatemachineConfigurationError` for more details

        :return: None
        """

        detour_list = detour_list or []

        def _convert(
            intermediate: List[Union[STATE, DecisionArg, None]],
        ) -> List[Union[StateMachineNode, SubStateDecision]]:
            converted_list: List[Union[StateMachineNode, SubStateDecision]] = []
            for inter in intermediate:
                if inter is None:
                    break
                if isinstance(inter, DecisionArg):
                    converted_dict = {}
                    for _trigger, value in inter.decision_dict.items():
                        converted_dict[_trigger] = _convert(value)

                    converted_list.append(
                        SubStateDecision(
                            decision_dict=converted_dict,
                            decision_handler=inter.decision_handler,
                        )
                    )
                    # only one decision at the end of a path
                    break
                converted_list.append(self._get_node(inter))
            return converted_list

        start_node = self._get_node(start)
        destination_node = self._get_node(destination)
        intermediate_nodes = _convert(detour_list)
        start_node.add_transition(
            StateMachineTransition(
                trigger=trigger,
                origin_node=start_node,
                target_node=destination_node,
                intermediate_nodes=intermediate_nodes,
            )
        )

    def ignore(self, state: STATE, trigger: TRIGGER):
        """
        Ignore a trigger for the specified state.
        :param state:
        :param trigger:
        :return:
        """
        n = self._get_node(state)
        n.ignore_trigger(trigger)

    def is_busy(self) -> bool:
        """returns true if the state machine is within a transition"""
        return self._in_transition

    def _get_node(self, state):
        """
        this funktion will return the desired node.
        if no node is registered, it will auto register a new node and return it.
        :param state:
        :return:
        """
        if state in self._nodes:
            return self._nodes[state]
        node = self._nodes[state] = StateMachineNode(
            state=state, on_entry=self._on_entry, on_exit=self._on_exit
        )
        return node
