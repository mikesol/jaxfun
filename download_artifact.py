from comet_ml import ExistingExperiment
import local_env

experiment = ExistingExperiment(
    api_key=local_env.comet_ml_api_key,
    project_name="jax-tcn-attn",
    previous_experiment="966f353bf95b43828662eb0c5c8f2b05",
)
logged_artifact = experiment.get_artifact(
    "checkpoint", "mikesol", version_or_alias="916.0.0"
)

# Download the artifact:
local_artifact = logged_artifact.download("/tmp/model")
