"""Single-file frozen inference in a new Python process; independent of train.py."""
import argparse,json
from pathlib import Path
from runtime import load_frozen,predict_files,read_amplitude


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--file',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();model,state,kind=load_frozen(args.checkpoint)
    amplitude=read_amplitude(args.file)
    device_name={232:'HP',64:'ESP32'}[amplitude.shape[0]]
    row={'file_path':str(args.file),'sample_id':args.file.stem,'device':device_name,'binary_label':'0'}
    score=predict_files(model,state,kind,[row])[0]['fall_score']
    result={'file':str(args.file.resolve()),'device':device_name,'fall_score':score,'threshold':state['selected_threshold'],
            'prediction':'Fall' if score>=state['selected_threshold'] else 'Nonfall','model_version':state['model_version'],
            'candidate_kind':kind,'seed':state['seed']}
    if args.output:
        if args.output.exists():raise FileExistsError('Inference output exists')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
