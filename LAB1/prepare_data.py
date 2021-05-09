import re
import sys
import argparse
import subprocess as sp

parser = argparse.ArgumentParser(description='Process captions file.')
parser.add_argument('--vid','-v',required=True,help='.mp4 video file')
parser.add_argument('--rt','-t',required=True,help='Transcription file')
parser.add_argument('--poi',required=True,help='POI ID')
parser.add_argument('--counter',default=0,help='Utterance counter (for more videos)')

args = parser.parse_args()

# frame counter
i=int(args.counter)

#
activation = False

spk = 'id' + (args.poi).zfill(2)

#sp.Popen(["mkdir",spk]).wait()

with open(args.rt,'r') as rt, open('mp4.scp','a') as scp, open('text','a') as txt , open('utt2spk','a') as utt2spk:

	for line in rt:
		# read times from line
		times = re.findall('\d\d:[0-5]\d:[0-5]\d.\d\d\d',line)

		# if this is a times line
		if len(times) > 0:

			# if this is POI speech
			if line.split()[-1] == '1':
				beg_t = times[0]
				end_t = times[1]

				utt = spk + '-' + str(i).zfill(5)
				out_vid_file = spk + '/' + utt + '.mp4'
				utt2spk.write(utt + ' ' + spk + '\n')
				scp.write(utt + ' ' + out_vid_file + '\n')
				
				sp.Popen(['ffmpeg -y -i', args.vid,'-ss',beg_t,'-to',end_t,out_vid_file]).wait()
				#sp.Popen(['ffmpeg', args.vid,'-ss',beg_t,'-to',end_t,out_vid_file]).wait()

				activation = True
				i+=1

		# transcription line
		elif len(line.split()) > 0:
			if activation:
				txt.write(utt + ' ' + line)
				activation = False