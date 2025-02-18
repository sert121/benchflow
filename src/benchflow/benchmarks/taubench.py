import os
import json
import logging
from typing import Any, Dict, List

from benchflow import BaseBench, BaseBenchConfig
from tau_bench.types import RunConfig
from tau_bench.envs import get_env  # Ensure this is correctly available


class TaubenchConfig(BaseBenchConfig):
    required_env = []
    optional_env = [
        "ENV", "MODEL", "MODEL_PROVIDER", "USER_MODEL", "USER_MODEL_PROVIDER",
        "AGENT_STRATEGY", "TEMPERATURE", "TASK_SPLIT", "START_INDEX", "END_INDEX",
        "TASK_IDS", "LOG_DIR", "MAX_CONCURRENCY", "SEED", "SHUFFLE", "USER_STRATEGY",
        "FEW_SHOT_DISPLAYS_PATH"
    ]

    defaults = {
        "ENV": "retail",
        "MODEL": "gpt-4o",
        "MODEL_PROVIDER": "openai",
        "USER_MODEL": "gpt-4o",
        "USER_MODEL_PROVIDER": "openai",
        "AGENT_STRATEGY": "tool-calling",
        "TEMPERATURE": 0.0,
        "TASK_SPLIT": "test",
        "START_INDEX": 0,
        "END_INDEX": -1,
        "LOG_DIR": "results",
        "MAX_CONCURRENCY": 1,
        "SEED": 10,
        "SHUFFLE": 0,
        "USER_STRATEGY": "llm",
        "FEW_SHOT_DISPLAYS_PATH": None,
    }

    def __init__(self, params: Dict[str, Any]):
        # Ensure all defaults are set
        for key, value in self.defaults.items():
            params.setdefault(key, value)

        # Handling TASK_IDS
        task_ids = None
        if isinstance(params.get("TASK_IDS"), str) and params["TASK_IDS"].strip():
            try:
                task_ids = [int(i.strip()) for i in params["TASK_IDS"].split(',')]
            except ValueError:
                logging.warning("Invalid TASK_IDS format. Should be comma-separated integers or a single integer.")
                task_ids = None
        elif isinstance(params.get("TASK_IDS"), int):
            task_ids = [params["TASK_IDS"]]

        params["TASK_IDS"] = task_ids  # Ensuring it's always a list or None

        # Call the parent constructor
        super().__init__(params)

    def validate(self) -> bool:
        """Validate the configuration parameters."""
        try:
            super().validate()  # Validate required environment variables

            if self.params["ENV"] not in ["retail", "airline"]:
                raise ValueError(f"Invalid ENV value: {self.params['ENV']}")

            if not isinstance(self.params["TEMPERATURE"], (int, float)) or not 0 <= self.params["TEMPERATURE"] <= 1:
                raise ValueError(f"Invalid TEMPERATURE value: {self.params['TEMPERATURE']}")

            if self.params["TASK_SPLIT"] not in ["train", "test", "dev"]:
                raise ValueError(f"Invalid TASK_SPLIT value: {self.params['TASK_SPLIT']}")

        except ValueError as e:
            logging.error(f"Validation error: {e}")
            return False

        return True

    def get_env(self) -> Dict[str, str]:
        """Get environment variables using tau_bench's get_env function."""
        task_index = int(self.params["task_index"]) if self.params.get("task_index") is not None else None

        env = get_env(
            env_name=self.params.get("ENV", "retail"),
            user_strategy=self.params.get("USER_STRATEGY", "llm"),
            user_model=self.params.get("USER_MODEL", "gpt-4o"),
            task_split=self.params.get("TASK_SPLIT", "test"),
            user_provider=self.params.get("USER_MODEL_PROVIDER", "openai"),
            task_index=task_index
        )

        # Add additional environment variables
        for key in self.required_env + self.optional_env:
            if key in self.params and self.params[key] is not None:
                env[key] = str(self.params[key])

        return env


class TauBench(BaseBench):
    def __init__(self, config: TaubenchConfig):
        super().__init__()
        self.config: TaubenchConfig = config
        self.params: Dict[str, Any] = config.params
        self.results_dir = self.get_results_dir_in_container()
        self.log_files_dir = self.get_log_files_dir_in_container()
        self.tasks = self.load_tasks()

    def load_tasks(self):
        """Load task definitions based on the configured environment."""
        if self.params["ENV"] == "retail":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
        elif self.params["ENV"] == "airline":
            from tau_bench.envs.airline.tasks_test import TASKS as tasks
        else:
            raise ValueError(f"Invalid ENV value: {self.params['ENV']}")
        return tasks

    def get_image_name(self) -> str:
        """Return the Docker image name dynamically."""
        return os.getenv("TAUBENCH_IMAGE", "yashdl/taubench")

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """Retrieve results from the result files."""
        results_file = os.path.join(self.results_dir, f"tau_bench_results.{task_id}.json")
        log_file = os.path.join(self.log_files_dir, f"tau_bench_log.{task_id}.txt")

        try:
            with open(results_file, 'r') as f:
                result_data = json.load(f)

            reward = result_data.get("reward", 0.0)

            with open(log_file, 'r') as f:
                log_content = f.read()

            is_resolved = reward > 0.99

            return {
                "is_resolved": is_resolved,
                "score": reward,
                "message": {"details": result_data},
                "log": log_content,
            }
        except FileNotFoundError:
            logging.warning(f"Result file not found for task {task_id}: {results_file}")
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": f"Result file not found: {results_file}"},
                "log": "",
            }
        except Exception as e:
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": str(e)},
                "log": str(e),
            }

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """Retrieve all task IDs."""
        if not self.tasks:
            return {"task_ids": [], "error_message": "Tasks not loaded."}

        try:
            task_ids = [str(tid) for tid in self.config.params["TASK_IDS"]] if self.config.params["TASK_IDS"] else \
                       [str(i) for i in range(len(self.tasks))]
        except Exception as e:
            return {"task_ids": [], "error_message": str(e)}

        return {"task_ids": task_ids, "error_message": None}

    def cleanup(self):
        pass

    def get_config(self, params: Dict[str, Any], task_id: str) -> TaubenchConfig:
        """Return a TaubenchConfig instance for the given task."""
        task_params = params.copy()
        task_params["task_index"] = int(task_id)
        return TaubenchConfig(task_params)

    def get_log_files_dir_in_container(self):
        return "/tmp/logs"

    def get_results_dir_in_container(self):
        return "/tmp/results"

    def run(self, task_ids, agents, requirements_dir, params, api=None):
        """Run benchmark tasks."""
        if isinstance(task_ids, (str, int)):
            task_ids = [str(task_ids)]

        results = []
        for task_id in task_ids:
            task_params = params.copy()
            task_params["task_index"] = int(task_id)

            result = self.run_bench(
                task_id=str(task_id),
                agent_url=agents.get_url() if hasattr(agents, 'get_url') else None,
                params=task_params
            )
            results.append(result)

        return results
