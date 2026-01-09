import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import snake7.train_neat
import snake7.watch

def test_neat_config_generation(tmp_path):
    """Test that train_neat runs and generates config/genome with defaults."""
    config_out = tmp_path / "neat_config.txt"
    genome_out = tmp_path / "genome.pkl"
    
    # Run for 1 generation
    args = [
        "snake7.train_neat",
        "--pop-size", "2",
        "--generations", "1",
        "--config-out", str(config_out),
        "--genome-out", str(genome_out),
        "--width", "6",
        "--height", "6",
        "--episodes-per-genome", "1",
        "--seed", "42"
    ]
    
    with patch.object(sys, "argv", args):
        snake7.train_neat.main()
        
    assert config_out.exists()
    assert genome_out.exists()
    
    content = config_out.read_text()
    # Check default env shapes
    assert "num_inputs              = 9" in content
    assert "num_outputs             = 3" in content
    
def test_watch_decoupling():
    """Test that watch.py has its own select_action and works."""
    assert hasattr(snake7.watch, "NeatAgent")
    assert hasattr(snake7.watch.NeatAgent, "_select_action")
    assert callable(snake7.watch.NeatAgent._select_action)
    
    # Verify logic (argmax)
    assert snake7.watch.NeatAgent._select_action([0.1, 0.9, 0.2]) == 1
    assert snake7.watch.NeatAgent._select_action([0.8, 0.1, 0.1]) == 0
    assert snake7.watch.NeatAgent._select_action([0.1, 0.1, 0.8]) == 2

def test_neat_config_dynamic_shape(tmp_path):
    """Test that config adapts to Env shapes."""
    config_out = tmp_path / "neat_config.txt"
    genome_out = tmp_path / "genome.pkl"
    
    # Mock SnakeEnv to return weird shapes
    with patch("snake7.env.SnakeEnv") as MockEnv:
        mock_instance = MockEnv.return_value
        
        # Setup spaces
        mock_obs = MagicMock()
        mock_obs.shape = (15,)
        mock_instance.observation_space = mock_obs
        
        mock_act = MagicMock()
        mock_act.n = 4
        mock_instance.action_space = mock_act
        
        # Prevent actual training run by mocking neat.Population
        with patch("neat.Population") as MockPop, \
             patch("neat.StatisticsReporter") as MockStats, \
             patch("pickle.dump"):
             
             # Mock best_genome to return a dummy
             mock_stats_instance = MockStats.return_value
             mock_stats_instance.best_genome.return_value = MagicMock(fitness=0.0)

             args = [
                "snake7.train_neat",
                "--pop-size", "2",
                "--generations", "1",
                "--config-out", str(config_out),
                "--genome-out", str(genome_out),
             ]
             with patch.object(sys, "argv", args):
                 snake7.train_neat.main()
                 
    assert config_out.exists()
    content = config_out.read_text()
    assert "num_inputs              = 15" in content
    assert "num_outputs             = 4" in content
