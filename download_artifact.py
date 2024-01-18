from comet_ml import ExistingExperiment
import local_env
experiment = ExistingExperiment(
    api_key=local_env.comet_ml_api_key,
    project_name="jax-tcn-attn",
    previous_experiment="7c2a0a329fb744099d0855682c603d97"
)
logged_artifact = experiment.get_artifact(
    "checkpoint",
    "mikesol",
    version_or_alias="718.0.0"
)

# Download the artifact:
local_artifact = logged_artifact.download("/tmp/model")