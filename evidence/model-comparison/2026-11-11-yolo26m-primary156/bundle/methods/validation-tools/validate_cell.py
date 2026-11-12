"""Read-only cell validation; writes only the requested owned validation report.

Call after native track_sequence exits successfully and before packet gzip.
The copied native validator decodes every packet canonically and checks all
frame identities/statuses, IDs, classes, scores and finite/nondegenerate boxes.
"""
from pathlib import Path
import argparse,datetime,gzip,hashlib,json,math,os,re,struct,subprocess,tempfile,time
from urllib.parse import unquote

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
VALIDATOR=HERE/'validate_packets'
VALIDATOR_SHA='b03679c4766f00dde9bb44945053352ad7e51d776d213eafe9c8e39ac21c3b0e'
BASELINE=REPO/'build/yolo26m-trt-comparison-20261111/quality/main/MOT17-10-SDP/resolved.config'
PACKET_BYTES=12368
SCMO_HEADER=struct.Struct('<4sI5iQQ')
SCMO_FRAME=struct.Struct('<qqIiiffff')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def config(path):
 lines=path.read_text().splitlines();d={}
 for line in lines:
  if '=' in line:
   k,v=line.split('=',1)
   assert k not in d,('duplicate config key',k)
   d[k]=v
 return d
def tracker_output(d):return {k:v for k,v in d.items() if k.startswith(('tracker.','output.'))}
def validate_motion(path,frames,width,height,epoch):
 opener=gzip.open if path.suffix=='.gz' else open
 statuses={str(i):0 for i in range(4)}
 with opener(path,'rb') as f:
  header=f.read(SCMO_HEADER.size);assert len(header)==44,'truncated SCMO header'
  magic,version,w,h,scale,mw,mh,e,n=SCMO_HEADER.unpack(header)
  expected_scale=max(1,(width+1919)//1920,(height+1079)//1080)
  assert (magic,version,w,h,scale,mw,mh,e,n)==(b'SCMO',1,width,height,expected_scale,width//expected_scale,height//expected_scale,epoch,frames),('SCMO header/key/count differs',(magic,version,w,h,scale,mw,mh,e,n))
  for i in range(frames):
   raw=f.read(44);assert len(raw)==44,('truncated SCMO frame',i)
   fid,ref,status,px,py,pcr,confidence,tx,ty=SCMO_FRAME.unpack(raw)
   assert fid==i and ref==i-1,('SCMO identity/reference frame',i,fid,ref)
   assert status in (1,2,3),'disabled/unknown status in live G1 stream'
   assert (i==0 and status==1) or (i>0 and status in (2,3)),('SCMO initialization/status sequence',i,status)
   assert all(math.isfinite(x) for x in (pcr,confidence,tx,ty)),('nonfinite SCMO values',i)
   assert 0<=confidence<=1,('SCMO confidence range',i)
   if status!=2:assert tx==0 and ty==0,('non-valid motion carries translation',i)
   statuses[str(status)]+=1
  assert not f.read(1),'trailing SCMO bytes'
 return {'frames':frames,'epoch':epoch,'key':[w,h,scale,mw,mh],'all_frame_references_contiguous':True,
         'all_floats_finite':True,'status_counts':statuses,'sha256_of_file':sha(path)}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['packets','motion','config','out']:p.add_argument('--'+k,type=Path,required=True)
 for k in ['frames','width','height']:p.add_argument('--'+k,type=int,required=True)
 p.add_argument('--epoch',type=int,default=1);p.add_argument('--arm');p.add_argument('--sequence')
 p.add_argument('--baseline-config',type=Path,default=BASELINE);p.add_argument('--engine-sha256')
 args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(HERE),'report must stay under validation-tools'
 out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists(),'refuse to overwrite validation receipt'
 started=time.monotonic();temp=None
 try:
  assert args.frames>0 and args.width>=32 and args.height>=32 and args.epoch==1
  assert sha(VALIDATOR)==VALIDATOR_SHA,'native validator changed'
  d=config(args.config);base=config(args.baseline_config)
  assert d['schema']=='saccade-tracking-config/1'
  assert tracker_output(d)==tracker_output(base),'tracker/output config differs from frozen baseline'
  assert (int(d['source.width']),int(d['source.height']),int(d['source.epoch']))==(args.width,args.height,args.epoch)
  assert (d['implementation.backend'],d['implementation.input'],d['implementation.model_profile'],d['implementation.motion_profile'])==('cuda','jpeg','hr-1280x768','v1-motion/2')
  assert (d['execution.mode'],d['execution.cuda_graph'],int(d['execution.candidate_slots']),int(d['execution.motion_slots']),int(d['execution.result_slots']),d['execution.perturb_seed'])==('pipelined','true',2,2,4,'0')
  assert int(d['preprocess.max_width'])>=args.width and int(d['preprocess.max_height'])>=args.height
  identity={}
  for line in unquote(d.get('build.identity','')).splitlines():
   if '=' in line:
    k,v=line.split('=',1);identity[k]=v
  if args.engine_sha256:assert identity.get('build.engine_sha256')==args.engine_sha256,'engine identity differs'
  path=args.packets;expected_bytes=args.frames*PACKET_BYTES
  if path.suffix=='.gz':
   tmpdir=HERE/'.tmp';tmpdir.mkdir(exist_ok=True)
   fd,name=tempfile.mkstemp(prefix='packets-',suffix='.bin',dir=tmpdir);os.close(fd);temp=Path(name)
   written=0
   with gzip.open(path,'rb') as src,temp.open('wb') as dst:
    while chunk:=src.read(8<<20):
     written+=len(chunk);assert written<=expected_bytes,'decompressed packet size exceeds declared frames';dst.write(chunk)
   assert written==expected_bytes,'decompressed packet size differs'
   raw=temp
  else:raw=path
  assert raw.stat().st_size==expected_bytes,'packet byte size/count differs'
  packet_sha=sha(raw)
  cmd=[str(VALIDATOR),str(raw),str(args.frames)]
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  assert r.returncode==0,('native packet validation failed',r.stdout)
  m=re.fullmatch(r'PASS frames=(\d+) observed_rows=(\d+) predicted_rows=(\d+) canonical=true all_status_ok=true\s*',r.stdout)
  assert m and int(m[1])==args.frames,'unexpected packet validator output'
  motion=validate_motion(args.motion,args.frames,args.width,args.height,args.epoch)
  stable={str(q):sha(q) for q in [args.motion,args.config]}
  result={'status':'PASS','scope':'CPU structural/config validation only; not GPU/session sanitizer or quality acceptance',
   'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arm':args.arm,'sequence':args.sequence,
   'frames':args.frames,'width':args.width,'height':args.height,'epoch':args.epoch,
   'packets':{'path':str(path.resolve()),'sha256_raw':packet_sha,'bytes_raw':expected_bytes,'observed_rows':int(m[2]),'predicted_rows':int(m[3]),'canonical_all_frames':True,'all_status_ok':True,
              'all_unique_UID_ID_class0_finite_boxes':True,'validator_sha256':VALIDATOR_SHA,'validator_command':cmd,'validator_stdout':r.stdout.strip()},
   'motion':motion,'config_sha256':sha(args.config),'frozen_tracker_output_sha256':hashlib.sha256(json.dumps(tracker_output(base),sort_keys=True).encode()).hexdigest(),
   'tracker_output_matches_frozen_baseline':True,'source_and_execution_config_correct':True,'build_identity':identity,
   'wall_seconds':time.monotonic()-started}
  assert stable=={q:sha(Path(q)) for q in stable},'input changed during validation'
  assert packet_sha==sha(raw),'packet changed during validation'
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'status':'PASS','arm':args.arm,'sequence':args.sequence,'frames':args.frames,'observed_rows':int(m[2]),'out':str(out)}),flush=True)
 except Exception as e:
  out.write_text(json.dumps({'status':'FAIL','error':str(e),'arm':args.arm,'sequence':args.sequence,'frames':args.frames,'wall_seconds':time.monotonic()-started},indent=2)+'\n')
  raise
 finally:
  if temp is not None:temp.unlink(missing_ok=True)

if __name__=='__main__':main()
