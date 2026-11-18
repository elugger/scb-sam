"""Actual Linux descriptor/lease checks in a private synthetic directory.

Executes only the pinned launcher's three stdlib lease helpers. No CUDA, cases,
model, production lease, original save.lock, qualification or worker is used.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import tempfile


CHILD = r'''
import fcntl,json,os,sys
fd=int(sys.argv[1]); path=sys.argv[2]
s=os.fstat(fd); p=os.stat(path)
assert (s.st_dev,s.st_ino)==(p.st_dev,p.st_ino)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
print(json.dumps(dict(device=s.st_dev,inode=s.st_ino,inherited=True)),flush=True)
assert sys.stdin.readline().strip()=='release'
os.close(fd)
'''


def run(launcher, expected_sha, work):
    import fcntl
    if sys.platform != 'linux':
        raise ValueError('This check requires the real Linux flock/pass_fds APIs')
    raw=launcher.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Launcher SHA differs')
    selected=[node for node in ast.parse(raw).body
              if isinstance(node,ast.FunctionDef) and node.name in {'need','plain','lease_open'}]
    if {node.name for node in selected}!={'need','plain','lease_open'}:
        raise ValueError('Missing exact launcher helpers')
    checks=[]
    with tempfile.TemporaryDirectory(prefix='native_lease_synthetic_',dir=work) as name:
        private=Path(name).resolve()
        if private.parent!=work.resolve():
            raise ValueError('Synthetic directory escaped work root')
        target=private/'lease.lock'
        marker=b'NATIVE_MAIN_TABLE_R4_GPU1_EXCLUSIVE_OPERATION_V1\n'
        namespace=dict(Path=Path,os=os,stat=stat,LEASE=target,LEASE_BYTES=marker)
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(launcher),'exec'),namespace)
        opened=namespace['lease_open'](create=True)
        os.close(opened)
        before=(target.stat().st_dev,target.stat().st_ino,target.read_bytes())
        fd=namespace['lease_open']()
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checks.append('pinned_launcher_lease_created_and_locked_in_private_scratch')
        child=None
        try:
            child=subprocess.Popen([sys.executable,'-B','-c',CHILD,str(fd),str(target)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                text=True,pass_fds=(fd,),start_new_session=True)
            readable,_,_=select.select([child.stdout],[],[],10)
            if not readable:
                raise ValueError('Inherited child did not confirm within 10 seconds')
            reply=json.loads(child.stdout.readline())
            assert reply['inherited'] is True
            assert (reply['device'],reply['inode'])==before[:2]
            os.close(fd);fd=None
            competing=os.open(target,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(competing,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    checks.append('child_inherits_same_open_description_and_excludes_competitor_after_parent_close')
                else:
                    raise ValueError('Competing process acquired live child lease')
            finally:
                os.close(competing)
            output,error=child.communicate('release\n',timeout=10)
            if child.returncode!=0 or output or error:
                raise ValueError('Synthetic child did not exit cleanly: '+repr((output,error)))
            reopened=namespace['lease_open']()
            try:
                fcntl.flock(reopened,fcntl.LOCK_EX|fcntl.LOCK_NB)
                checks.append('lease_released_only_after_child_exits')
            finally:
                os.close(reopened)
            assert before==(target.stat().st_dev,target.stat().st_ino,target.read_bytes())
            checks.append('marker_inode_and_content_not_recreated_by_open_or_handoff')
        finally:
            if fd is not None:os.close(fd)
            if child is not None and child.poll() is None:
                # Only this program's explicitly created synthetic child.
                child.communicate('release\n',timeout=10)
        target.write_bytes(b'altered\n')
        try:namespace['lease_open']()
        except ValueError:checks.append('corrupt_lease_marker_rejected')
        else:raise ValueError('Corrupt marker was accepted')
        target.unlink()
        other=private/'regular';other.write_bytes(marker)
        target.symlink_to(other)
        try:namespace['lease_open']()
        except ValueError:checks.append('symlink_lease_rejected')
        else:raise ValueError('Symlink was accepted')
        target.unlink()
        os.link(other,target)
        try:namespace['lease_open']()
        except ValueError:checks.append('hardlinked_lease_rejected')
        else:raise ValueError('Hardlinked lease was accepted')
    return dict(status='LINUX_LEASE_MECHANISM_PASS',launcher_sha256=expected_sha,checks=checks,
        production_worker_executed=False,production_lease_opened=False,production_save_lock_opened=False,
        cases_read=0,gpu_calls=0,test_evaluations=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher',type=Path,required=True)
    parser.add_argument('--launcher-sha',required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing evidence')
    args.work_dir.mkdir(exist_ok=True)
    result=run(args.launcher,args.launcher_sha,args.work_dir)
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
