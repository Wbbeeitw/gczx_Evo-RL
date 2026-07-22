import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def make_queue(queue_blend_steps=0):
    return ActionQueue(
        RTCConfig(
            enabled=True,
            execution_horizon=5,
            queue_blend_steps=queue_blend_steps,
        )
    )


def test_action_execution_requires_acknowledgement():
    queue = make_queue()
    actions = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert queue.merge(actions, actions, real_delay=0)

    reserved = queue.get_for_execution()

    assert torch.equal(reserved, actions[0])
    assert queue.qsize() == 3
    assert queue.get_action_count() == 0

    queue.mark_action_sent(reserved)

    assert queue.qsize() == 2
    assert queue.get_action_count() == 1


def test_failed_action_does_not_advance_queue():
    queue = make_queue()
    actions = torch.ones(2, 4)
    assert queue.merge(actions, actions, real_delay=0)

    queue.get_for_execution()
    queue.mark_action_failed()

    assert queue.qsize() == 2
    assert queue.get_action_count() == 0


def test_merge_uses_executed_action_count_for_delay():
    queue = make_queue()
    first = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    assert queue.merge(first, first, real_delay=0)
    _, action_count, starvation_count = queue.get_rtc_snapshot()

    for _ in range(2):
        action = queue.get_for_execution()
        queue.mark_action_sent(action)

    replacement = torch.arange(40, 60, dtype=torch.float32).reshape(5, 4)
    assert queue.merge(
        replacement,
        replacement,
        real_delay=0,
        action_count_before_inference=action_count,
        starvation_count_before_inference=starvation_count,
    )

    assert queue.qsize() == 3
    assert torch.equal(queue.get_left_over()[0], replacement[2])


def test_queue_blend_starts_from_last_executed_action():
    queue = make_queue(queue_blend_steps=2)
    first = torch.tensor([[0.0], [1.0], [2.0]])
    assert queue.merge(first, first, real_delay=0)
    action = queue.get_for_execution()
    queue.mark_action_sent(action)

    replacement = torch.tensor([[10.0], [11.0], [12.0]])
    assert queue.merge(replacement, replacement, real_delay=0)

    blended = queue.get_for_execution()
    assert 0.0 < blended.item() < 10.0
