"""
Central configuration loader for the ZenML GitFlow template.

This module loads configuration from project_config.yaml and provides
easy access to project settings across all scripts and pipelines.
"""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Model configuration settings."""
    name: str = "MyModel"
    description: str = "My ML model"
    tags: List[str] = []


class PipelineConfig(BaseModel):
    """Pipeline configuration settings."""
    name: str = "my_pipeline"
    run_name_prefix: str = "run"
    tags: List[str] = []


class SnapshotConfig(BaseModel):
    """Snapshot configuration settings."""
    prefix: str = "snapshot"


class EnvironmentConfig(BaseModel):
    """Environment-specific configuration."""
    tags: List[str] = []


class EnvironmentsConfig(BaseModel):
    """All environment configurations."""
    local: EnvironmentConfig = EnvironmentConfig(tags=["local", "development"])
    staging: EnvironmentConfig = EnvironmentConfig(tags=["staging", "pre-release"])
    production: EnvironmentConfig = EnvironmentConfig(tags=["production", "release"])


class ProjectInfo(BaseModel):
    """Project information."""
    name: str = "my-project"
    description: str = "My ML project"


class ProjectConfig(BaseModel):
    """Complete project configuration."""
    project: ProjectInfo = ProjectInfo()
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()
    snapshot: SnapshotConfig = SnapshotConfig()
    environments: EnvironmentsConfig = EnvironmentsConfig()


def find_project_root() -> Path:
    """Find the project root directory by looking for project_config.yaml."""
    current = Path.cwd()
    
    # Walk up the directory tree looking for project_config.yaml
    for parent in [current] + list(current.parents):
        config_path = parent / "project_config.yaml"
        if config_path.exists():
            return parent
    
    # Fallback: check relative to this file's location
    utils_dir = Path(__file__).parent
    project_root = utils_dir.parent
    if (project_root / "project_config.yaml").exists():
        return project_root
    
    return current


def load_project_config(config_path: Optional[str] = None) -> ProjectConfig:
    """
    Load project configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. If not provided,
                    searches for project_config.yaml in project root.
    
    Returns:
        ProjectConfig object with all settings.
    """
    if config_path is None:
        project_root = find_project_root()
        config_path = project_root / "project_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return ProjectConfig()
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f) or {}
    
    return ProjectConfig(**raw_config)


# Global config instance (lazy loaded)
_config: Optional[ProjectConfig] = None


def get_config() -> ProjectConfig:
    """Get the global project configuration (lazy loaded)."""
    global _config
    if _config is None:
        _config = load_project_config()
    return _config


def get_model_name() -> str:
    """Get the configured model name."""
    return get_config().model.name


def get_model_description() -> str:
    """Get the configured model description."""
    return get_config().model.description


def get_model_tags() -> List[str]:
    """Get the configured model tags."""
    return get_config().model.tags


def get_pipeline_name() -> str:
    """Get the configured pipeline name."""
    return get_config().pipeline.name


def get_run_name_template(environment: Optional[str] = None) -> str:
    """
    Get the run name template with placeholders.
    
    Args:
        environment: Optional environment name to include in run name.
    
    Returns:
        Run name template like "training_run_{date}_{time}"
    """
    prefix = get_config().pipeline.run_name_prefix
    if environment:
        return f"{prefix}_{environment}_{{date}}_{{time}}"
    return f"{prefix}_{{date}}_{{time}}"


def get_pipeline_tags(environment: Optional[str] = None) -> List[str]:
    """
    Get pipeline tags, optionally including environment-specific tags.
    
    Args:
        environment: Optional environment name (local, staging, production)
    
    Returns:
        List of tags to apply to the pipeline run.
    """
    config = get_config()
    tags = list(config.pipeline.tags)  # Copy base tags
    
    if environment:
        env_config = getattr(config.environments, environment, None)
        if env_config:
            tags.extend(env_config.tags)
    
    return tags


def get_snapshot_name(environment: str, git_sha: Optional[str] = None) -> str:
    """
    Generate a snapshot name based on environment and git SHA.
    
    Args:
        environment: Environment name (staging, production)
        git_sha: Optional git commit SHA (uses short form if provided)
    
    Returns:
        Snapshot name like "STG_price_prediction_abc1234"
    """
    config = get_config()
    prefix_map = {
        "local": "LOCAL",
        "staging": "STG",
        "production": "PROD",
    }
    env_prefix = prefix_map.get(environment, environment.upper())
    
    if git_sha:
        # Use short SHA (first 7 characters)
        short_sha = git_sha[:7] if len(git_sha) > 7 else git_sha
        return f"{env_prefix}_{config.snapshot.prefix}_{short_sha}"
    else:
        return f"{env_prefix}_{config.snapshot.prefix}"


# Convenience exports
__all__ = [
    "ProjectConfig",
    "load_project_config",
    "get_config",
    "get_model_name",
    "get_model_description",
    "get_model_tags",
    "get_pipeline_name",
    "get_run_name_template",
    "get_pipeline_tags",
    "get_snapshot_name",
]

