#!/usr/bin/env python3
"""Export the separately captured MOT17-02 observations and source video.

This exports replay assets only; it never executes inference or reads GT.
The one new inference capture is independently retained in evidence/report1/.
"""
import argparse
import configparser
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
DEMO = Path(__file__).resolve().parents[1]
ASSETS = DEMO/'assets'
EVIDENCE = ROOT/'evidence/report1/2026-11-11-mot17-02-demo'
spec = importlib.util.spec_from_file_location('packet_export', Path(__file__).with_name('build_data.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
sha = common.file_digest
dump = common.dump


def probe_video(path):
    command = ['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,color_range,color_space,color_transfer,color_primaries:format=duration,size',
        '-of','json',str(path)]
    result=json.loads(subprocess.check_output(command,text=True))
    stream,=result['streams']
    assert (stream['codec_name'],stream['width'],stream['height'],stream['pix_fmt'])==('h264',1920,1080,'yuv420p')
    assert stream['avg_frame_rate']==stream['r_frame_rate']=='30/1'
    assert int(stream['nb_frames'])==int(stream['nb_read_frames'])==600
    assert abs(float(result['format']['duration'])-20)<0.00001
    assert (stream['color_space'],stream['color_transfer'],stream['color_primaries'],stream['color_range'])==('bt709','iec61966-2-1','bt709','tv')
    boxes=[]
    with path.open('rb') as f:
        while header:=f.read(8):
            size,kind=struct.unpack('>I4s',header);consumed=8
            if size==1:size,=struct.unpack('>Q',f.read(8));consumed=16
            if size==0:size=path.stat().st_size-f.tell()+consumed
            assert size>=consumed
            boxes.append(kind.decode('ascii'));f.seek(size-consumed,1)
    assert boxes.index('moov')<boxes.index('mdat')
    result.update(command=command,mp4_box_order=boxes,faststart_verified=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-video',action='store_true')
    args=parser.parse_args()
    summary=json.loads((EVIDENCE/'summary.json').read_text())
    assert summary['status']=='PASS_COMPLETE_SOURCE_CAPTURE' and summary['source_inputs_binaries_unchanged']
    contract=summary['contract'];sequence=Path(contract['sequence'])
    info=configparser.ConfigParser();info.read(sequence/'seqinfo.ini');info=info['Sequence']
    assert (info['name'],info.getint('seqLength'),info.getint('frameRate'),info.getint('imWidth'),info.getint('imHeight'))==('MOT17-02-SDP',600,30,1920,1080)
    manifest_path=EVIDENCE/'inputs-before.json'
    manifest=json.loads(manifest_path.read_text())
    images=[sequence/info['imDir']/f'{i:06}{info["imExt"]}' for i in range(1,601)]
    for p in [sequence/'seqinfo.ini',*images]:
        expected=manifest[str(p)]
        assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],str(p)
    packet_path=EVIDENCE/'raw/run1/packets.bin.gz'
    motion_path=EVIDENCE/'raw/run1/motion.scmo.gz'
    raw_packets=gzip.decompress(packet_path.read_bytes());raw_motion=gzip.decompress(motion_path.read_bytes())
    assert hashlib.sha256(raw_packets).hexdigest()==summary['outputs']['packets.bin']['sha256']
    assert hashlib.sha256(raw_motion).hexdigest()==summary['outputs']['motion.scmo']['sha256']
    frames,packet_stats=common.packets(raw_packets,600)
    motion_stats=common.motions(raw_motion,frames)
    template=json.loads((ASSETS/'data.json').read_text())['meta'].copy()
    for key in ['singleSamples','queuedSamples','offlineSequenceFrames','offlineFreshProcesses']:
        template.pop(key,None)
    template.update(sequence=info['name'],width=1920,height=1080,fps=30,frameCount=600,durationSeconds=20,
        baseline=contract['runtime_base_commit'],engineHash=contract['engine_sha256'],
        sourceVideo='assets/source-02.mp4',poster='assets/poster-02.jpg',annotatedVideo='assets/tracking-demo-02.mp4',
        packetSha256=summary['outputs']['packets.bin']['sha256'],motionSha256=summary['outputs']['motion.scmo']['sha256'],
        driverVersion=summary['driver_version'],metricScope='No per-frame latency benchmark was performed for MOT17-02; metrics are null.',
        capturePurpose='one complete native source capture for prerecorded demonstration; no new GT or scoring')
    data={'meta':template,'frames':frames,'metrics':None}
    compact=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    (ASSETS/'data-02.json').write_text(compact+'\n')
    (ASSETS/'data-02.js').write_text('window.SACCADE_DEMOS=Object.assign({},window.SACCADE_DEMOS,{"MOT17-10-SDP":window.SACCADE_DEMO,"MOT17-02-SDP":'+compact+'});\n')
    poster_source=images[220]
    shutil.copyfile(poster_source,ASSETS/'poster-02.jpg')
    video=ASSETS/'source-02.mp4'
    command=['ffmpeg','-hide_banner','-nostdin','-n','-framerate','30','-start_number','1',
        '-i',str(sequence/info['imDir']/'%06d.jpg'),'-frames:v','600','-an',
        '-vf','scale=in_range=pc:out_range=tv:in_color_matrix=bt601:out_color_matrix=bt709,setsar=1,setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709',
        '-c:v','libx264','-preset','medium','-crf','21','-threads','4','-pix_fmt','yuv420p',
        '-fps_mode','passthrough','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','iec61966-2-1',
        '-movflags','+faststart',str(video)]
    if video.exists():
        assert args.reuse_video,'Existing02 video requires --reuse-video and provenance check'
        prior=json.loads((ASSETS/'provenance-02.json').read_text())
        assert prior['artifacts']['source-02.mp4']['sha256']==sha(video)
        elapsed=prior['encode']['wall_seconds']
    else:
        started=time.monotonic()
        print('START source-only02 encode, CPU4threads',flush=True)
        with (ASSETS/'source-02-encode.log').open('wb') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        elapsed=time.monotonic()-started
    probe=probe_video(video)
    for p in images:
        assert sha(p)==manifest[str(p)]['sha256']
    provenance={'schema':'saccade-report1-mot17-02-replay-assets/1','baseline':contract['runtime_base_commit'],
        'purpose':'600-frame MOT17-02 prerecorded packet observations and source-only video',
        'new_inference_capture':'evidence/report1/2026-11-11-mot17-02-demo/',
        'no_new_gt_quality_scoring_or_sanitizer':True,'runtime_modified':False,
        'source_inputs':{'sequence':str(sequence),'frame_count':600,'fps':30,
            'jpeg_bytes':sum(p.stat().st_size for p in images),'image_hashes_verified_before_after_encode':True,
            'archived_manifest':str(manifest_path.relative_to(ROOT)),'archived_manifest_sha256':sha(manifest_path),
            'packets_gzip':str(packet_path.relative_to(ROOT)),'packets_gzip_sha256':sha(packet_path),
            'packet_raw_sha256':hashlib.sha256(raw_packets).hexdigest(),
            'motion_gzip':str(motion_path.relative_to(ROOT)),'motion_gzip_sha256':sha(motion_path),
            'motion_raw_sha256':hashlib.sha256(raw_motion).hexdigest(),'capture_summary_sha256':sha(EVIDENCE/'summary.json'),
            'poster_source':str(poster_source),'poster_source_sha256':sha(poster_source)},
        'packet_validation':packet_stats,'motion_validation':motion_stats,
        'metrics':None,'metrics_reason':'Source capture is not a per-frame latency benchmark; MOT17-10 metrics must not be attributed to02.',
        'encode':{'argv':command,'wall_seconds':elapsed,'uses_gpu':False,
            'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
            'lossy_video':True,'color_policy':'full-rangeJPEG BT601 to limitedBT709, sRGB transfer metadata'},
        'ffprobe':probe,
        'generator':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(Path(__file__))},
        'artifacts':{name:{'bytes':(ASSETS/name).stat().st_size,'sha256':sha(ASSETS/name)} for name in
            ['data-02.json','data-02.js','source-02.mp4','poster-02.jpg','source-02-encode.log']},
        'limitations':['prerecorded result playback, not live inference','20-second source30FPS is not a throughput claim',
            'no new held-out evaluation','state-estimate boxes with no offline MOT interpolation in packet replay']}
    dump(ASSETS/'provenance-02.json',provenance)
    assert json.loads(compact)==json.loads((ASSETS/'data-02.json').read_text())
    print(json.dumps({'status':'PASS','frames':600,'rows':packet_stats['rows'],'motion':motion_stats['status_counts'],
        'video_bytes':video.stat().st_size,'duration':20,'metrics':None},indent=2),flush=True)


if __name__=='__main__':main()
