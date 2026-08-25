"""Asynchronous request-ID transport for a persistent remote SmolVLA worker."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import queue
import select
import shlex
import subprocess
import threading
import time
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image

from .observation import ObservationSnapshot


class RemoteInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteRequest:
    request_id: int
    t_obs: float
    image_rgb: np.ndarray
    state: np.ndarray

    def __post_init__(self) -> None:
        image = np.asarray(self.image_rgb, dtype=np.uint8).copy()
        state = np.asarray(self.state, dtype=np.float64).copy()
        if image.shape != (256, 256, 3) or state.shape != (6,):
            raise ValueError("remote request image/state has an invalid shape")
        if not np.isfinite(state).all():
            raise ValueError("remote request state must be finite")
        image.setflags(write=False)
        state.setflags(write=False)
        object.__setattr__(self, "image_rgb", image)
        object.__setattr__(self, "state", state)

    @classmethod
    def from_observation(cls, observation: ObservationSnapshot) -> "RemoteRequest":
        return cls(
            observation.request_id,
            observation.t_obs,
            observation.image_rgb.copy(),
            (observation.payload_state if observation.payload_state is not None else observation.s_obs).copy(),
        )


@dataclass(frozen=True)
class RemoteResponse:
    request_id: int
    t_arr: float
    action: Any | None
    error: str | None = None


class RequestTransport(Protocol):
    def start(self) -> None: ...
    def request(self, payload: Mapping[str, Any], timeout_s: float) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


def _payload(request: RemoteRequest) -> dict[str, Any]:
    image = np.asarray(request.image_rgb, dtype=np.uint8)
    state = np.asarray(request.state, dtype=np.float64)
    if image.shape != (256, 256, 3) or state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("remote request image/state has an invalid shape or value")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return {
        "id": request.request_id,
        "t_obs": request.t_obs,
        "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "state": state.tolist(),
    }


class SubprocessSSHTransport:
    """Explicit SSH-agent/key transport. ``start`` is the only network boundary."""

    def __init__(self, remote: Mapping[str, Any]):
        self.remote = dict(remote)
        self.process: subprocess.Popen[str] | None = None

    def _worker_source(self) -> str:
        checkpoint = repr(str(self.remote["checkpoint"]))
        base_model_path = os.environ.get("GOPRO_UMI_REMOTE_BASE_MODEL_PATH")
        base_model_override = repr(base_model_path) if base_model_path else "None"
        remote_hf_home = os.environ.get("GOPRO_UMI_REMOTE_HF_HOME")
        hf_home_override = repr(remote_hf_home) if remote_hf_home else "None"
        return f'''import base64, contextlib, io, json, os, sys, traceback
hf_home_override = {hf_home_override}
if hf_home_override is not None:
    os.environ["HF_HOME"] = hf_home_override
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home_override, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home_override, "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
protocol_stdout = sys.stdout
sys.stdout = sys.stderr
import numpy as np
import torch
from PIL import Image
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
checkpoint = {checkpoint}
base_model_override = {base_model_override}
policy_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
if base_model_override is not None:
    policy_config.vlm_model_name = base_model_override
    policy_config.load_vlm_weights = False
policy = SmolVLAPolicy.from_pretrained(
    checkpoint,
    config=policy_config,
    local_files_only=True,
    strict=base_model_override is not None,
).to("cuda")
pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
policy.eval()
sys.stdout = protocol_stdout
print(json.dumps({{"status": "ready"}}), file=protocol_stdout, flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        image = np.array(Image.open(io.BytesIO(base64.b64decode(request["image_b64"]))))
        state = np.asarray(request["state"], dtype=np.float32)
        observation = {{
            "observation.images.camera1": torch.from_numpy(image).permute(2,0,1).float().unsqueeze(0).cuda() / 255.0,
            "observation.state": torch.from_numpy(state).unsqueeze(0).cuda(),
            "task": ["pick and place the target object"],
            "robot_type": ["so_follower"],
        }}
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            actions = post(policy.predict_action_chunk(pre(observation)))
        print(json.dumps({{"id": request["id"], "action": actions.detach().cpu().numpy().tolist()}}), file=protocol_stdout, flush=True)
    except Exception:
        print(json.dumps({{"error": traceback.format_exc()}}), file=protocol_stdout, flush=True)
'''

    def start(self) -> None:
        if self.process is not None:
            raise RemoteInferenceError("SSH transport already started")
        encoded = base64.b64encode(self._worker_source().encode()).decode("ascii")
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(float(self.remote['connect_timeout_s']))}",
            "-p",
            str(self.remote["port"]),
            f"{self.remote['user']}@{self.remote['host']}",
            str(self.remote["python"]),
            "-u",
            "-c",
            shlex.quote(f"import base64;exec(base64.b64decode('{encoded}'))"),
        ]
        # An already-authenticated, process-external SSH control socket may be
        # supplied for a bounded live validation.  Authentication material is
        # never accepted through source/config or forwarded to the worker.
        control_path = os.environ.get("GOPRO_UMI_SSH_CONTROL_PATH")
        if control_path:
            command[1:1] = ["-S", control_path]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        ready = self._read_json(float(self.remote["ready_timeout_s"]))
        if ready.get("status") != "ready":
            raise RemoteInferenceError("remote worker did not report readiness")

    def _read_json(self, timeout_s: float) -> Mapping[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RemoteInferenceError("remote worker is not running")
        readable, _, _ = select.select([self.process.stdout], [], [], timeout_s)
        if not readable:
            raise TimeoutError("remote response timeout")
        line = self.process.stdout.readline()
        if not line:
            detail = self.process.stderr.read() if self.process.stderr else ""
            raise RemoteInferenceError(f"remote worker closed unexpectedly: {detail[-400:]}")
        response = json.loads(line)
        if "error" in response:
            raise RemoteInferenceError("remote worker returned an error")
        return response

    def request(self, payload: Mapping[str, Any], timeout_s: float) -> Mapping[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise RemoteInferenceError("remote worker is not running")
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        return self._read_json(timeout_s)

    def close(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=float(self.remote["shutdown_timeout_s"]))
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None


class RemoteSmolVLAClient:
    """One bounded request queue and bounded response queue with explicit lifecycle."""

    def __init__(
        self,
        transport: RequestTransport,
        request_timeout_s: float,
        request_queue_size: int = 1,
        response_queue_size: int = 4,
        clock=time.monotonic,
    ):
        self.transport = transport
        self.request_timeout_s = float(request_timeout_s)
        self.requests: queue.Queue[RemoteRequest | None] = queue.Queue(maxsize=request_queue_size)
        self.responses: queue.Queue[RemoteResponse] = queue.Queue(maxsize=response_queue_size)
        self.clock = clock
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.transport.start()
        self._thread = threading.Thread(target=self._run, name="smolvla-remote", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            request = self.requests.get()
            if request is None:
                return
            try:
                response = self.transport.request(_payload(request), self.request_timeout_s)
                response_id = int(response["id"])
                if response_id != request.request_id:
                    raise RemoteInferenceError("remote response ID does not match its request")
                item = RemoteResponse(response_id, float(self.clock()), response.get("action"))
            except Exception as exc:
                item = RemoteResponse(request.request_id, float(self.clock()), None, type(exc).__name__)
            self.responses.put(item)

    def submit(self, request: RemoteRequest) -> None:
        self.requests.put_nowait(request)

    def poll(self) -> RemoteResponse | None:
        try:
            return self.responses.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._thread is not None:
            self.requests.put(None)
            self._thread.join(timeout=2.0)
        self.transport.close()
        self._thread = None


__all__ = [
    "RemoteInferenceError",
    "RemoteRequest",
    "RemoteResponse",
    "RemoteSmolVLAClient",
    "RequestTransport",
    "SubprocessSSHTransport",
]
