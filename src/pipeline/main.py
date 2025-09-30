from src.pipeline.training_pipeline import Pipeline
from src.config.constants import CONFIG

if __name__ == "__main__":
    pipeline = Pipeline(config=CONFIG)
    training_history = pipeline.run()
