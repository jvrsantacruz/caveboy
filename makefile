DEPS := pnglite/pnglite.h perceptron/pattern.h perceptron/perceptron.h

SRC := caveboy.c
OBJS := $(SRC:.c=.o)
OBJS += $(DEPS:.h=.o)
TARGETS := $(SRC:%.c=%)

EXE := caveboy

TRANSFORMER:= ./transformer.sh
FRAMES_DIR := ~/commercials_images
VIDEOS_DIR := ~/commercials
VIDEOS := ${VIDEOS_DIR}/*.*

CC=gcc
# Debug flags
CFLAGS := -g -pg -enable-checking -ggdb -Wall -O0 -pedantic -std=c99 -DDEBUG -Iperceptron 
# Production flags
#CFLAGS := -Wall -O3 -pedantic -std=c99 -Iperceptron 
LDFLAGS := -lm -lz 

.PHONY: deps clean slice_videos analyze

all: ${TARGETS}

deps:
	make -C zlib

${EXE}: ${OBJS}
	@echo Compiling ${EXE}
	$(CC) -o $@ $^ $(LDFLAGS)

slice_videos: ${VIDEOS}
	@echo Slicing videos...
	./${TRANSFORMER} ${VIDEOS_DIR} ${FRAMES_DIR} 128x64 3
	@echo Done slicing

analyze: caveboy
	@echo Analyzing...
	@echo Starting training...
	./${EXE} ${FRAMES_DIR} -w weights.dat -e error.dat -m 20 -t
	@echo Done training.
	@echo Starting testing...
	./${EXE} ${FRAMES_DIR} -w weights.dat -e error.dat -m 20
	@echo Done testing.

clean:
	@echo Cleaning caveboy objects
	rm -fr ${EXE} ${OBJS} ${TARGETS}

clean-all: clean
	@echo Cleaning zlib objects
	make -C zlib clean
	rm -rf ${FRAMES_DIR}
