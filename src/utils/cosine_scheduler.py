import numpy as np


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    frozen_epochs=0,
    num_annealing_phases=1,
    lr_factor_by_phase=[1.]
):
    total_iters = epochs * niter_per_ep

    schedule_phases = []

    frozen_schedule = np.array([])
    frozen_iters = frozen_epochs * niter_per_ep
    if frozen_epochs > 0:
        frozen_schedule = np.zeros(frozen_iters)
    schedule_phases.append(frozen_schedule)

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    schedule_phases.append(warmup_schedule)

    total_iters_for_annealing = epochs * niter_per_ep - warmup_iters - frozen_iters

    for phase_idx in range(num_annealing_phases):
        factor_for_phase = lr_factor_by_phase[phase_idx]
        iters_for_phase = total_iters_for_annealing // num_annealing_phases
        iters = np.arange(iters_for_phase)
        schedule = final_value + 0.5 * (base_value * factor_for_phase - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        schedule_phases.append(schedule)

    tail = np.array([])
    total_scheduled_iters = sum([len(phase) for phase in schedule_phases])
    remaining_iters = total_iters - total_scheduled_iters
    if remaining_iters > 0: 
        tail = np.zeros(remaining_iters)
    schedule_phases.append(tail)

    schedule = np.concatenate(schedule_phases)
    assert len(schedule) == epochs * niter_per_ep
    return schedule