from __future__ import annotations

import pickle
from pathlib import Path

import pytest


def _write_neat_config(path: Path, *, num_inputs: int, num_outputs: int) -> None:
    path.write_text(
        (
            f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000000
pop_size              = 2
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_type          = gaussian
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# node response options
response_init_type      = gaussian
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01
enabled_rate_to_true_add  = 0.0
enabled_rate_to_false_add = 0.0

# connection weight options
weight_init_type        = gaussian
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate        = 0.8
weight_replace_rate     = 0.1

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = {int(num_inputs)}
num_outputs             = {int(num_outputs)}
feed_forward            = True
initial_connection      = full

# structural mutation options
single_structural_mutation = False
structural_mutation_surer   = default

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
""".lstrip()
            + "\n"
        ),
        encoding="utf-8",
    )


def _make_connection_gene(conn_gene_type, key: tuple[int, int], *, innovation: int):
    """
    neat-python compatibility helper.

    - neat-python <= 0.92: DefaultConnectionGene.__init__(key)
    - neat-python >= 1.0: DefaultConnectionGene.__init__(key, innovation) (required)
    """
    try:
        return conn_gene_type(key, innovation)
    except TypeError:
        return conn_gene_type(key)


def test_neat_viz_dot_skips_disabled_by_default(tmp_path: Path):
    neat = pytest.importorskip("neat")

    from snake7.neat_viz import generate_dot

    config_path = tmp_path / "neat_config.txt"
    _write_neat_config(config_path, num_inputs=2, num_outputs=2)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    genome = config.genome_type(0)
    genome.nodes = {}
    genome.connections = {}

    node_gene_type = config.genome_config.node_gene_type
    conn_gene_type = config.genome_config.connection_gene_type

    # Output nodes must exist in genome.nodes.
    for out_key in config.genome_config.output_keys:
        n = node_gene_type(out_key)
        genome.nodes[out_key] = n

    hidden_key = 10
    genome.nodes[hidden_key] = node_gene_type(hidden_key)

    in0, in1 = config.genome_config.input_keys[:2]
    out0, out1 = config.genome_config.output_keys[:2]

    c1 = _make_connection_gene(conn_gene_type, (in0, hidden_key), innovation=1)
    c1.weight = 0.5
    c1.enabled = True
    genome.connections[(in0, hidden_key)] = c1

    c2 = _make_connection_gene(conn_gene_type, (hidden_key, out0), innovation=2)
    c2.weight = -1.0
    c2.enabled = True
    genome.connections[(hidden_key, out0)] = c2

    c3 = _make_connection_gene(conn_gene_type, (in1, out1), innovation=3)
    c3.weight = 1.0
    c3.enabled = False
    genome.connections[(in1, out1)] = c3

    genome_path = tmp_path / "best_genome.pkl"
    with open(genome_path, "wb") as f:
        pickle.dump(genome, f)

    dot = generate_dot(genome_path, config_path)

    assert "digraph" in dot
    assert '"-1"' in dot
    assert '"-2"' in dot
    assert '"0"' in dot
    assert '"1"' in dot
    assert '"10"' in dot
    assert '"-1" -> "10"' in dot
    assert '"10" -> "0"' in dot

    # Disabled edges are skipped by default.
    assert '"-2" -> "1"' not in dot


def test_neat_viz_dot_include_disabled(tmp_path: Path):
    neat = pytest.importorskip("neat")

    from snake7.neat_viz import generate_dot

    config_path = tmp_path / "neat_config.txt"
    _write_neat_config(config_path, num_inputs=2, num_outputs=2)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    genome = config.genome_type(0)
    genome.nodes = {}
    genome.connections = {}

    node_gene_type = config.genome_config.node_gene_type
    conn_gene_type = config.genome_config.connection_gene_type

    for out_key in config.genome_config.output_keys:
        genome.nodes[out_key] = node_gene_type(out_key)

    in1 = config.genome_config.input_keys[1]
    out1 = config.genome_config.output_keys[1]

    c = _make_connection_gene(conn_gene_type, (in1, out1), innovation=1)
    c.weight = 1.0
    c.enabled = False
    genome.connections[(in1, out1)] = c

    genome_path = tmp_path / "best_genome.pkl"
    with open(genome_path, "wb") as f:
        pickle.dump(genome, f)

    dot = generate_dot(genome_path, config_path, include_disabled=True)

    assert '"-2" -> "1"' in dot
    assert "style=dotted" in dot or "style=\"dotted\"" in dot
