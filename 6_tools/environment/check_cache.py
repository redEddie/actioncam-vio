import pexpect
import sys

def run_remote_command(cmd_str):
    cmd = ['ssh', '-p', '17970', 'kimminje@155.230.189.77', cmd_str]
    child = pexpect.spawn(cmd[0], cmd[1:], encoding='utf-8', timeout=10)
    try:
        idx = child.expect(['(?i)password:', pexpect.EOF, pexpect.TIMEOUT], timeout=5)
        if idx == 0:
            child.sendline(os.getenv('SSH_PASSWORD', ''))
        child.expect(pexpect.EOF)
        print(child.before)
    except Exception as e:
        print(f"Error: {e}")

print("=== STEP A: ENV VARS ===")
run_remote_command('echo "HF_HOME: $HF_HOME"; echo "HF_HUB_CACHE: $HF_HUB_CACHE"; echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"; echo "XDG_CACHE_HOME: $XDG_CACHE_HOME"')

print("=== STEP B: CACHE DIRECTORIES ===")
run_remote_command('ls -ld /home/kimminje/gopro_umi/smolvla_cache 2>/dev/null || echo "Not found A"')
run_remote_command('ls -ld /home/kimminje/.cache/huggingface 2>/dev/null || echo "Not found B"')

print("=== STEP C: CHECK SNAPSHOT ===")
run_remote_command('HASH=$(cat /mnt/hf_cache/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main); ls -la /mnt/hf_cache/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/$HASH')
