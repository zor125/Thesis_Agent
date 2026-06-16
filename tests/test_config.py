from paper_agent.config import load_config


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("app:\n  name: test-agent\n", encoding="utf-8")

    config = load_config(config_file)

    assert config["app"]["name"] == "test-agent"
