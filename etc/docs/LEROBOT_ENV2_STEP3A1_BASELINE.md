# LeRobot env2 baseline before STEP 3A.1 dependency resolution

Captured: 2026-08-13 Asia/Seoul

- Interpreter: `/home/kimminje/miniconda3/envs/lerobot_env2/bin/python`
- Python: `3.12.13`
- Project LeRobot HEAD: `1bb9933215dcb7ffeeae6d3746cda3f73f5a59e2`
- Project LeRobot status: clean
- Existing editable metadata: `lerobot==0.6.1` points to removed path
  `4_deploy/Teleop/lerobot`; import fails.
- Missing before changes: `datasets`, `av`.
- Incompatible before changes: `pandas==3.0.5` while pinned project requires
  `pandas>=2.0.0,<3.0.0`.
- Source Zarr file-list/size/mtime manifest SHA256:
  `d6e0d3f29354980835b19919b2e9976951cafc1ad48c7ed7267007428c1c01d5`.
- Source Zarr: 2,978 files, 8,965,709,145 bytes.

## Post-resolution audit

Only the existing `lerobot_env2` environment was changed. No environment was
created. Project-local LeRobot source remained clean and was imported through
`PYTHONPATH` from `third_party/lerobot/src`.

- Added directly: `datasets==4.8.5`, `av==15.1.0`.
- Corrected to pinned range: `pandas==3.0.5` → `pandas==2.3.3`.
- Resolver-adjusted dependency: `fsspec==2026.6.0` → `fsspec==2026.2.0`.
- Added dataset transitive dependencies: `aiohappyeyeballs==2.7.1`,
  `aiohttp==3.14.3`, `aiosignal==1.4.0`, `dill==0.4.1`,
  `frozenlist==1.8.0`, `multidict==6.7.1`, `multiprocess==0.70.19`,
  `propcache==0.5.2`, `pytz==2026.3.post1`, `tzdata==2026.3`,
  `xxhash==4.0.0`, and `yarl==1.24.5`.
- `pip check`: no broken requirements.
- Writer import: PASS.
- Reader import: PASS.

## pip freeze

```text
accelerate==1.14.0
addict==2.4.0
annotated-doc==0.0.4
annotated-types==0.8.0
anyio==4.14.2
asttokens==3.0.2
attrs==26.1.0
blinker==1.9.0
certifi==2026.6.17
charset-normalizer==3.4.9
click==8.4.2
cloudpickle==3.1.2
cmake==4.1.3
cmeel==0.60.1
cmeel-assimp==5.4.3.1
cmeel-boost==1.87.0.1
cmeel-console-bridge==1.0.2.3
cmeel-octomap==1.10.0
cmeel-qhull==8.0.2.1
cmeel-tinyxml2==10.0.0
cmeel-urdfdom==4.0.1
cmeel-zlib==1.3.2
coal-library==3.0.1
comm==0.2.3
ConfigArgParse==1.7.5
contourpy==1.3.3
cuda-bindings==13.3.1
cuda-pathfinder==1.5.6
cuda-toolkit==13.0.2
cycler==0.12.1
dash==4.4.1
decorator==5.3.1
deepdiff==8.6.2
docopt==0.6.2
donfig==0.8.1.post1
draccus==0.10.0
eigenpy==3.10.3
einops==0.8.2
eiquadprog==1.2.9
executing==2.2.1
Farama-Notifications==0.0.6
fastapi==0.140.7
fastjsonschema==2.22.1
feetech-servo-sdk==1.0.0
filelock==3.31.2
Flask==3.1.3
fonttools==4.63.0
fsspec==2026.6.0
google-crc32c==1.8.0
gymnasium==1.3.0
h11==0.16.0
hebi-py==2.11.0
hf-xet==1.5.2
httpcore==1.0.9
httptools==0.8.0
httpx==0.28.1
huggingface_hub==1.24.0
idna==3.18
importlib_metadata==9.0.0
iniconfig==2.3.0
ipython==9.15.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.8
ischedule==1.2.7
itsdangerous==2.2.0
janus==2.0.0
jedi==0.20.0
Jinja2==3.1.6
joblib==1.5.3
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter_core==5.9.1
jupyterlab_widgets==3.0.16
kiwisolver==1.5.0
# Editable install with no version control (lerobot==0.6.1)
-e /home/kimminje/Desktop/project/gopro_umi/4_deploy/Teleop/lerobot
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.11.1
matplotlib-inline==0.2.2
mdurl==0.1.2
mergedeep==1.3.4
meshcat==0.3.2
mpmath==1.3.0
mypy_extensions==1.1.0
narwhals==2.24.0
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.6.1
num2words==0.5.14
numcodecs==0.16.5
numpy==2.2.6
nvidia-cublas==13.1.0.3
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti==13.0.85
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime==13.0.96
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==9.1.0.70
nvidia-cudnn-cu13==9.19.0.56
nvidia-cufft==12.0.0.61
nvidia-cufft-cu11==10.9.0.58
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver==12.0.4.66
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse==12.6.3.3
nvidia-cusparse-cu11==11.7.5.86
nvidia-cusparselt-cu13==0.8.0
nvidia-nccl-cu11==2.21.5
nvidia-nccl-cu13==2.28.9
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
nvidia-nvtx-cu11==11.8.86
open3d==0.19.0
opencv-python==5.0.0.93
opencv-python-headless==4.13.0.92
orderly-set==5.5.0
packaging==25.0
pandas==3.0.5
parso==0.8.7
peft==0.20.0
pexpect==4.9.0
pillow==12.3.0
pin==3.4.0
placo==0.9.15
platformdirs==4.11.0
plotly==6.9.0
pluggy==1.6.0
prompt_toolkit==3.0.52
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
pyarrow==25.0.0
pydantic==2.13.4
pydantic_core==2.46.4
Pygments==2.20.0
pyngrok==8.1.2
pyparsing==3.3.2
pyquaternion==0.9.9
pyrealsense2==2.58.3.10794
pyserial==3.5
pytest==9.1.1
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
PyYAML==6.0.3
pyyaml-include==1.4.1
pyzmq==27.1.0
referencing==0.37.0
regex==2026.7.19
requests==2.34.2
retrying==1.4.2
rhoban-cmeel-jsoncpp==1.9.4.9
rich==15.0.0
rpds-py==2026.6.3
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.18.0
setuptools==80.10.2
shellingham==1.5.4
six==1.17.0
stack-data==0.6.3
starlette==1.3.1
sympy==1.14.0
teleop==0.1.5
termcolor==3.3.0
threadpoolctl==3.6.0
tokenizers==0.22.2
toml==0.10.2
torch==2.7.1+cu118
torchaudio==2.7.1+cu118
torchvision==0.22.1+cu118
tornado==6.5.7
tqdm==4.69.0
traitlets==5.15.1
transformers==5.14.1
transforms3d==0.4.2
triton==3.3.1
typer==0.27.0
typing-inspect==0.9.0
typing-inspection==0.4.2
typing_extensions==4.16.0
u-msgpack-python==2.8.0
urllib3==2.7.0
uvicorn==0.51.0
uvloop==0.22.1
watchfiles==1.2.0
wcwidth==0.8.2
websocket-client==1.9.0
websockets==16.1.1
Werkzeug==3.1.8
wheel==0.47.0
widgetsnbextension==4.0.15
zarr==3.3.0
zipp==4.1.0
```

## conda list

```text
# packages in environment at /home/kimminje/miniconda3/envs/lerobot_env2:
#
# Name                       Version          Build            Channel
_libgcc_mutex                0.1              main
_openmp_mutex                5.1              52_gnu
accelerate                   1.14.0           pypi_0           pypi
addict                       2.4.0            pypi_0           pypi
annotated-doc                0.0.4            pypi_0           pypi
annotated-types              0.8.0            pypi_0           pypi
anyio                        4.14.2           pypi_0           pypi
asttokens                    3.0.2            pypi_0           pypi
attrs                        26.1.0           pypi_0           pypi
blinker                      1.9.0            pypi_0           pypi
bzip2                        1.0.8            h5eee18b_6
ca-certificates              2026.7.16        h06a4308_0
certifi                      2026.6.17        pypi_0           pypi
charset-normalizer           3.4.9            pypi_0           pypi
click                        8.4.2            pypi_0           pypi
cloudpickle                  3.1.2            pypi_0           pypi
cmake                        4.1.3            pypi_0           pypi
cmeel                        0.60.1           pypi_0           pypi
cmeel-assimp                 5.4.3.1          pypi_0           pypi
cmeel-boost                  1.87.0.1         pypi_0           pypi
cmeel-console-bridge         1.0.2.3          pypi_0           pypi
cmeel-octomap                1.10.0           pypi_0           pypi
cmeel-qhull                  8.0.2.1          pypi_0           pypi
cmeel-tinyxml2               10.0.0           pypi_0           pypi
cmeel-urdfdom                4.0.1            pypi_0           pypi
cmeel-zlib                   1.3.2            pypi_0           pypi
coal-library                 3.0.1            pypi_0           pypi
comm                         0.2.3            pypi_0           pypi
configargparse               1.7.5            pypi_0           pypi
contourpy                    1.3.3            pypi_0           pypi
cuda-bindings                13.3.1           pypi_0           pypi
cuda-pathfinder              1.5.6            pypi_0           pypi
cuda-toolkit                 13.0.2           pypi_0           pypi
cycler                       0.12.1           pypi_0           pypi
dash                         4.4.1            pypi_0           pypi
decorator                    5.3.1            pypi_0           pypi
deepdiff                     8.6.2            pypi_0           pypi
docopt                       0.6.2            pypi_0           pypi
donfig                       0.8.1.post1      pypi_0           pypi
draccus                      0.10.0           pypi_0           pypi
eigenpy                      3.10.3           pypi_0           pypi
einops                       0.8.2            pypi_0           pypi
eiquadprog                   1.2.9            pypi_0           pypi
executing                    2.2.1            pypi_0           pypi
farama-notifications         0.0.6            pypi_0           pypi
fastapi                      0.140.7          pypi_0           pypi
fastjsonschema               2.22.1           pypi_0           pypi
feetech-servo-sdk            1.0.0            pypi_0           pypi
filelock                     3.31.2           pypi_0           pypi
flask                        3.1.3            pypi_0           pypi
fonttools                    4.63.0           pypi_0           pypi
fsspec                       2026.6.0         pypi_0           pypi
google-crc32c                1.8.0            pypi_0           pypi
gymnasium                    1.3.0            pypi_0           pypi
h11                          0.16.0           pypi_0           pypi
hebi-py                      2.11.0           pypi_0           pypi
hf-xet                       1.5.2            pypi_0           pypi
httpcore                     1.0.9            pypi_0           pypi
httptools                    0.8.0            pypi_0           pypi
httpx                        0.28.1           pypi_0           pypi
huggingface-hub              1.24.0           pypi_0           pypi
idna                         3.18             pypi_0           pypi
importlib-metadata           9.0.0            pypi_0           pypi
iniconfig                    2.3.0            pypi_0           pypi
ipython                      9.15.0           pypi_0           pypi
ipython-pygments-lexers      1.1.1            pypi_0           pypi
ipywidgets                   8.1.8            pypi_0           pypi
ischedule                    1.2.7            pypi_0           pypi
itsdangerous                 2.2.0            pypi_0           pypi
janus                        2.0.0            pypi_0           pypi
jedi                         0.20.0           pypi_0           pypi
jinja2                       3.1.6            pypi_0           pypi
joblib                       1.5.3            pypi_0           pypi
jsonschema                   4.26.0           pypi_0           pypi
jsonschema-specifications    2025.9.1         pypi_0           pypi
jupyter-core                 5.9.1            pypi_0           pypi
jupyterlab-widgets           3.0.16           pypi_0           pypi
kiwisolver                   1.5.0            pypi_0           pypi
ld_impl_linux-64             2.44             h9e0c5a2_3
lerobot                      0.6.1            pypi_0           pypi
libexpat                     2.8.2            h7354ed3_1
libffi                       3.4.8            h06d3fd0_3
libgcc                       15.2.0           h69a1729_8
libgcc-ng                    15.2.0           h166f726_8
libstdcxx                    15.2.0           h39759b7_8
libuuid                      1.41.5           h5eee18b_0
libxcb                       1.17.0           h9b100fa_0
libzlib                      1.3.2            h47b2149_0
markdown-it-py               4.2.0            pypi_0           pypi
markupsafe                   3.0.3            pypi_0           pypi
matplotlib                   3.11.1           pypi_0           pypi
matplotlib-inline            0.2.2            pypi_0           pypi
mdurl                        0.1.2            pypi_0           pypi
mergedeep                    1.3.4            pypi_0           pypi
meshcat                      0.3.2            pypi_0           pypi
mpmath                       1.3.0            pypi_0           pypi
mypy-extensions              1.1.0            pypi_0           pypi
narwhals                     2.24.0           pypi_0           pypi
nbformat                     5.10.4           pypi_0           pypi
ncurses                      6.5              h7934f7d_0
nest-asyncio                 1.6.0            pypi_0           pypi
networkx                     3.6.1            pypi_0           pypi
num2words                    0.5.14           pypi_0           pypi
numcodecs                    0.16.5           pypi_0           pypi
numpy                        2.2.6            pypi_0           pypi
nvidia-cublas                13.1.0.3         pypi_0           pypi
nvidia-cublas-cu11           11.11.3.6        pypi_0           pypi
nvidia-cuda-cupti            13.0.85          pypi_0           pypi
nvidia-cuda-cupti-cu11       11.8.87          pypi_0           pypi
nvidia-cuda-nvrtc            13.0.88          pypi_0           pypi
nvidia-cuda-nvrtc-cu11       11.8.89          pypi_0           pypi
nvidia-cuda-runtime          13.0.96          pypi_0           pypi
nvidia-cuda-runtime-cu11     11.8.89          pypi_0           pypi
nvidia-cudnn-cu11            9.1.0.70         pypi_0           pypi
nvidia-cudnn-cu13            9.19.0.56        pypi_0           pypi
nvidia-cufft                 12.0.0.61        pypi_0           pypi
nvidia-cufft-cu11            10.9.0.58        pypi_0           pypi
nvidia-cufile                1.15.1.6         pypi_0           pypi
nvidia-curand                10.4.0.35        pypi_0           pypi
nvidia-curand-cu11           10.3.0.86        pypi_0           pypi
nvidia-cusolver              12.0.4.66        pypi_0           pypi
nvidia-cusolver-cu11         11.4.1.48        pypi_0           pypi
nvidia-cusparse              12.6.3.3         pypi_0           pypi
nvidia-cusparse-cu11         11.7.5.86        pypi_0           pypi
nvidia-cusparselt-cu13       0.8.0            pypi_0           pypi
nvidia-nccl-cu11             2.21.5           pypi_0           pypi
nvidia-nccl-cu13             2.28.9           pypi_0           pypi
nvidia-nvjitlink             13.0.88          pypi_0           pypi
nvidia-nvshmem-cu13          3.4.5            pypi_0           pypi
nvidia-nvtx                  13.0.85          pypi_0           pypi
nvidia-nvtx-cu11             11.8.86          pypi_0           pypi
open3d                       0.19.0           pypi_0           pypi
opencv-python                5.0.0.93         pypi_0           pypi
opencv-python-headless       4.13.0.92        pypi_0           pypi
openssl                      3.5.7            h1b28b03_0
orderly-set                  5.5.0            pypi_0           pypi
packaging                    25.0             pypi_0           pypi
pandas                       3.0.5            pypi_0           pypi
parso                        0.8.7            pypi_0           pypi
peft                         0.20.0           pypi_0           pypi
pexpect                      4.9.0            pypi_0           pypi
pillow                       12.3.0           pypi_0           pypi
pin                          3.4.0            pypi_0           pypi
pip                          26.1.2           pyhc872135_0
placo                        0.9.15           pypi_0           pypi
platformdirs                 4.11.0           pypi_0           pypi
plotly                       6.9.0            pypi_0           pypi
pluggy                       1.6.0            pypi_0           pypi
prompt-toolkit               3.0.52           pypi_0           pypi
psutil                       7.2.2            pypi_0           pypi
pthread-stubs                0.3              h0ce48e5_1
ptyprocess                   0.7.0            pypi_0           pypi
pure-eval                    0.2.3            pypi_0           pypi
pyarrow                      25.0.0           pypi_0           pypi
pydantic                     2.13.4           pypi_0           pypi
pydantic-core                2.46.4           pypi_0           pypi
pygments                     2.20.0           pypi_0           pypi
pyngrok                      8.1.2            pypi_0           pypi
pyparsing                    3.3.2            pypi_0           pypi
pyquaternion                 0.9.9            pypi_0           pypi
pyrealsense2                 2.58.3.10794     pypi_0           pypi
pyserial                     3.5              pypi_0           pypi
pytest                       9.1.1            pypi_0           pypi
python                       3.12.13          hc5f7cf0_3
python-dateutil              2.9.0.post0      pypi_0           pypi
python-dotenv                1.2.2            pypi_0           pypi
pyyaml                       6.0.3            pypi_0           pypi
pyyaml-include               1.4.1            pypi_0           pypi
pyzmq                        27.1.0           pypi_0           pypi
readline                     8.3              hc2a1206_0
referencing                  0.37.0           pypi_0           pypi
regex                        2026.7.19        pypi_0           pypi
requests                     2.34.2           pypi_0           pypi
retrying                     1.4.2            pypi_0           pypi
rhoban-cmeel-jsoncpp         1.9.4.9          pypi_0           pypi
rich                         15.0.0           pypi_0           pypi
rpds-py                      2026.6.3         pypi_0           pypi
safetensors                  0.8.0            pypi_0           pypi
scikit-learn                 1.9.0            pypi_0           pypi
scipy                        1.18.0           pypi_0           pypi
setuptools                   80.10.2          pypi_0           pypi
shellingham                  1.5.4            pypi_0           pypi
six                          1.17.0           pypi_0           pypi
sqlite                       3.53.2           h795bf6d_0
stack-data                   0.6.3            pypi_0           pypi
starlette                    1.3.1            pypi_0           pypi
sympy                        1.14.0           pypi_0           pypi
teleop                       0.1.5            pypi_0           pypi
termcolor                    3.3.0            pypi_0           pypi
threadpoolctl                3.6.0            pypi_0           pypi
tk                           8.6.15           h54e0aa7_0
tokenizers                   0.22.2           pypi_0           pypi
toml                         0.10.2           pypi_0           pypi
torch                        2.7.1+cu118      pypi_0           pypi
torchaudio                   2.7.1+cu118      pypi_0           pypi
torchvision                  0.22.1+cu118     pypi_0           pypi
tornado                      6.5.7            pypi_0           pypi
tqdm                         4.69.0           pypi_0           pypi
traitlets                    5.15.1           pypi_0           pypi
transformers                 5.14.1           pypi_0           pypi
transforms3d                 0.4.2            pypi_0           pypi
triton                       3.3.1            pypi_0           pypi
typer                        0.27.0           pypi_0           pypi
typing-extensions            4.16.0           pypi_0           pypi
typing-inspect               0.9.0            pypi_0           pypi
typing-inspection            0.4.2            pypi_0           pypi
tzdata                       2026c            he532380_0
u-msgpack-python             2.8.0            pypi_0           pypi
urllib3                      2.7.0            pypi_0           pypi
uvicorn                      0.51.0           pypi_0           pypi
uvloop                       0.22.1           pypi_0           pypi
watchfiles                   1.2.0            pypi_0           pypi
wcwidth                      0.8.2            pypi_0           pypi
websocket-client             1.9.0            pypi_0           pypi
websockets                   16.1.1           pypi_0           pypi
werkzeug                     3.1.8            pypi_0           pypi
wheel                        0.47.0           py312h06a4308_0
widgetsnbextension           4.0.15           pypi_0           pypi
xorg-libx11                  1.8.12           h9b100fa_1
xorg-libxau                  1.0.12           h9b100fa_0
xorg-libxdmcp                1.1.5            h9b100fa_0
xorg-xorgproto               2024.1           h47b2149_2
xz                           5.8.2            h448239c_0
zarr                         3.3.0            pypi_0           pypi
zipp                         4.1.0            pypi_0           pypi
zlib                         1.3.2            h47b2149_0
```
