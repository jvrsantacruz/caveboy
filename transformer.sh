#!/bin/bash

# Resize videos and slice it into images
# Usage: transformer.sh DIR [outdir] [WxH] [fps]
# Javier Santacruz
# Alberto Montes

# Default vars
VIDEO_DIR=
OUTDIR=images
RES=320x240
FPS=20
# Flags -y: Overwrite -an No audio
FFMPEG_FLAGS=-y -an -f rawvideo -vcodec png

if [ $# -lt 1 ]; then
	echo "Missing arguments."
	echo "Usage: $0 video_dir [outdir=${OUTDIR}] [WxH=${RES}] [fps=${FPS}]"
	exit 1
fi

VIDEO_DIR=$1;
if [ $# -gt 1 ]; then OUTDIR=$2; fi
if [ $# -gt 2 ]; then RES=$3; fi
if [ $# -gt 3 ]; then FPS=$4; fi

for VIDEO_PATH in $VIDEO_DIR/*.*
do
	#creamos el subdirectorio para ese vídeo
	VIDEO=${VIDEO_PATH%.*}
	OUTDIR_VIDEO=$OUTDIR/$(basename ${VIDEO})
	mkdir -p $OUTDIR_VIDEO

	echo "Encoding $VIDEO to ${OUTDIR_VIDEO}"

	#dividimos el video en imagenes
	ffmpeg $FFMPEG_FLAGS -i $VIDEO_PATH -r $FPS -s $RES $OUTDIR_VIDEO/image%04d.png

	echo OK $VIDEO
done

echo OK ALL
