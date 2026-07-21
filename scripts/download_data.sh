#!/bin/bash
hf download FrenzyMath/Herald_translator --local-dir data/herald
hf download JohnYang88/lean-dojo-mathlib4 --repo-type dataset --local-dir data/raw/lean-dojo-mathlib4
hf download internlm/Lean-Workbook --repo-type dataset --local-dir data/raw/Lean-Workbook
