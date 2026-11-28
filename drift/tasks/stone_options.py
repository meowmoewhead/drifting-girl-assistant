"""Rank cutoff and journal identity for direct parallel-world restarts."""

DEFAULT_RESTART_RANK = 35


def restart_rank(value=DEFAULT_RESTART_RANK):
    if type(value) is not int or not 1 <= value <= 40:
        raise ValueError('重开 RANK 必须是 1 到 40 的整数')
    return value


def is_rank_restart(record):
    rank = record.get('rank')
    cutoff = record.get('restart_rank')
    source, target = record.get('from_stage'), record.get('to_stage')
    return (record.get('reason') == 'restart' and record.get('entry_rank') == 0
            and type(rank) is int and type(cutoff) is int and 1 <= cutoff <= rank <= 40
            and type(source) is int and type(target) is int and 1 <= source <= target <= 99
            and record.get('rank40_verified') is (rank == 40))
