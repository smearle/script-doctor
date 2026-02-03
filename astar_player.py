import time

import click
import jax
import jax.numpy as jnp
from puxle import Puzzle
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from puzzlejax.env import PuzzleJaxEnv

def main(
    puzzle: PuzzleJaxEnv,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    heuristic: Heuristic,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    config = {
        "puzzle_name": puzzle_name,
        "search_options": search_options.dict(),
        "heuristic": heuristic.__class__.__name__,
        "heuristic_metadata": getattr(heuristic, "metadata", {}),
        "visualize_options": visualize_options.dict(),
    }
    print_config("A* Search Configuration", config)
    astar_fn = astar_builder(
        puzzle,
        heuristic,
        search_options.batch_size,
        search_options.get_max_node_size(),
        pop_ratio=search_options.pop_ratio,
        cost_weight=search_options.cost_weight,
        show_compile_time=search_options.show_compile_time,
    )
    dist_fn = heuristic.distance
    total_search_times, states_per_second, single_search_time = search_samples(
        search_fn=astar_fn,
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        dist_fn=dist_fn,
        dist_fn_format=heuristic_dist_format,
        seeds=seeds,
        search_options=search_options,
        visualize_options=visualize_options,
    )

    if search_options.vmap_size == 1:
        return

    vmapped_search_samples(
        vmapped_search=vmapping_search(
            puzzle, astar_fn, search_options.vmap_size, search_options.show_compile_time
        ),
        puzzle=puzzle,
        seeds=seeds,
        search_options=search_options,
        total_search_times=total_search_times,
        states_per_second=states_per_second,
        single_search_time=single_search_time,
    )

def vmapped_search_samples(
    vmapped_search,
    puzzle: Puzzle,
    seeds: list[int],
    search_options: SearchOptions,
    total_search_times: jnp.ndarray,
    states_per_second: float,
    single_search_time: float,
):
    console = Console()
    has_target = puzzle.has_target
    vmap_size = search_options.vmap_size

    solve_configs, states = vmapping_init_target(puzzle, vmap_size, seeds)

    console.print(
        build_vmapped_setup_panel(has_target=has_target, solve_configs=solve_configs, states=states)
    )

    start = time.time()
    search_result = vmapped_search(solve_configs, states)
    solved = search_result.solved.block_until_ready()
    end = time.time()
    vmapped_search_time = end - start

    if not has_target and solved.any():
        solved_st = vmapping_get_state(search_result, search_result.solved_idx)
        grid = Table.grid(expand=False)
        grid.add_column()
        grid.add_row(Align.center("[bold green]Solution State[/bold green]"))
        grid.add_row(Text.from_ansi(str(solved_st)))
        console.print(Panel(grid, title="[bold green]Vmapped Solution[/bold green]", expand=False))

    search_states = jnp.sum(search_result.generated_size)
    vmapped_states_per_second = search_states / vmapped_search_time

    if len(seeds) > 1:
        sizes = search_result.generated_size
        table = Table(title="[bold]Vmapped Search Results[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row(
            "Search Time",
            f"{vmapped_search_time:6.2f}s "
            f"(x{vmapped_search_time/jnp.sum(total_search_times)*vmap_size:.1f}/{vmap_size})",
        )
        table.add_row(
            "Total Search States",
            f"{human_format(jnp.sum(sizes))} (avg: {human_format(jnp.mean(sizes))})",
        )
        table.add_row(
            "States per Second",
            f"{human_format(vmapped_states_per_second)} (x{vmapped_states_per_second/states_per_second:.1f} faster)",
        )
        table.add_row(
            "Solutions Found",
            f"{jnp.sum(solved)}/{len(solved)} ({jnp.mean(solved)*100:.2f}%)",
        )
        console.print(table)
    else:
        table = Table(title="[bold]Vmapped Search Result[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row(
            "Search Time",
            f"{vmapped_search_time:6.2f}s (x{vmapped_search_time/single_search_time:.1f}/{vmap_size})",
        )
        table.add_row(
            "Search States",
            f"{human_format(search_states)} ({human_format(vmapped_states_per_second)} states/s)",
        )
        table.add_row("Speedup", f"x{vmapped_states_per_second/states_per_second:.1f}")
        table.add_row("Solutions Found", f"{jnp.mean(solved)*100:.2f}%")
        console.print(table)


if __name__ == "__main__":
    main()