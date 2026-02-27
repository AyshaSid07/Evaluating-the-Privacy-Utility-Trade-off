#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -n 64
#SBATCH -t 0-00:10:00

module 

python3 tokenization-defense.py