import time

import dotenv
from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api
from r2_labs.sdk import logging as r2_logging
from r2_labs.sdk import sentry

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_name",
    "test_model",
    "Name of the model to train",
)

flags.DEFINE_string(
    "dataset_path",
    "/data/datasets/pick_up_can_all",
    "Path to the dataset to train on",
)

flags.DEFINE_integer(
    "training_steps",
    1000,
    "Number of training steps",
)


def main(_):
  dotenv.load_dotenv()
  r2_logging.configure(service="train-skill")
  sentry.init_sentry(service="train-skill")

  robot = r2client.Robot(
      f"tcp://localhost:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://localhost:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://localhost:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  # Start the skill training
  robot.skill_trainer.train_model(
      model_name=FLAGS.model_name,
      dataset_path=FLAGS.dataset_path,
      training_steps=FLAGS.training_steps,
  )

  # TODO: get the training progress and display it using tqdm or similar.
  while True:
    status = robot.skill_trainer.get_training_status()
    if status.is_finished:
      break
    print("training ...")
    time.sleep(10.0)

  print("Training finished")


if __name__ == "__main__":
  try:
    app.run(main)
  except SystemExit:
    raise
  except KeyboardInterrupt:
    pass
  except Exception:
    sentry.capture_exception()
    raise
