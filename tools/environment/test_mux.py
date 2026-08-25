import pexpect
import subprocess
import time
import os

os.system("rm -f ~/.ssh/ctrl_socket")

cmd = ['ssh', '-M', '-S', '/home/kimminje/.ssh/ctrl_socket', '-p', '17970', 'kimminje@155.230.189.77', '-N']
child = pexpect.spawn(cmd[0], cmd[1:], encoding='utf-8', timeout=5)
idx = child.expect(['(?i)password:', pexpect.EOF, pexpect.TIMEOUT])
if idx == 0:
    child.sendline(os.getenv('SSH_PASSWORD', ''))
    print("Sent password from environment to master")
else:
    print("No password prompt")

time.sleep(2)

# Now test a pipe connection
p = subprocess.Popen(
    ['ssh', '-S', '/home/kimminje/.ssh/ctrl_socket', '-p', '17970', 'kimminje@155.230.189.77', 'cat'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
)
p.stdin.write("Hello world!\n")
p.stdin.flush()
print("Got back:", p.stdout.readline().strip())
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait()

# Stop master
subprocess.run(['ssh', '-S', '/home/kimminje/.ssh/ctrl_socket', '-O', 'exit', 'kimminje@155.230.189.77'])
