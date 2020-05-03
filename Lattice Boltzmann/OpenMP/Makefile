# Makefile

EXE=d2q9-bgk

COMPILER?=icc
OptimisationLevel ?= Ofast

ifeq ($(COMPILER), icc)
CC=icc
CFLAGS= -std=c99 -Wall -xhost -$(OptimisationLevel) -qopenmp
endif
ifeq ($(COMPILER), gcc)
CC=gcc
CFLAGS= -std=c99 -Wall -$(OptimisationLevel) -march=native -fopenmp
endif

LIBS = -lm

CheckSize?=128x128
FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=../check/$(CheckSize).final_state.dat
REF_AV_VELS_FILE=../check/$(CheckSize).av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python ../check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE) av_vels.dat final_state.dat
