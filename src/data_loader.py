"""Dataset Loader & Preprocessing Utility for Autoformalization.

This module normalizes raw dataset downloads (Herald, Lean Workbook, LeanDojo, miniF2F)
into standardized, isolated datasets located in `data/syntax` and `data/rlcf`.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataLoader")


class DatasetPreparer:
    """Manages parsing and standardization of autoformalization datasets."""

    def __init__(self, data_root: str = "data") -> None:
        """Initialize directory paths.

        Args:
            data_root: Root directory containing raw and processed data.
        """
        self.data_root = Path(data_root)
        self.syntax_dir = self.data_root / "syntax"
        self.rlcf_dir = self.data_root / "rlcf"
        
        self.syntax_dir.mkdir(parents=True, exist_ok=True)
        self.rlcf_dir.mkdir(parents=True, exist_ok=True)

    def process_herald(self) -> Path:
        """Process Herald Statements dataset for Stage 1 Syntax Tuning."""
        raw_path = self.data_root / "herald" / "data" / "train-00000-of-00001.parquet"
        out_path = self.syntax_dir / "herald.jsonl"
        
        if not raw_path.exists():
            logger.warning(f"Herald dataset not found at {raw_path}")
            return out_path

        logger.info(f"Processing Herald dataset from {raw_path}...")
        df = pd.read_parquet(raw_path)
        
        records = []
        for idx, row in df.iterrows():
            records.append({
                "problem_id": f"herald_{row.get('id', idx)}",
                "informal_statement": str(row.get("informal_statement", "")).strip(),
                "formal_statement": str(row.get("formal_statement", "")).strip(),
                "source": "herald"
            })
            
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        logger.info(f"Successfully exported {len(records)} Herald entries to {out_path}")
        return out_path

    def process_lean_workbook(self) -> Path:
        """Process Lean Workbook dataset for Stage 1 Syntax Tuning."""
        raw_path = self.data_root / "lean-workbook" / "lean_workbook.json"
        out_path = self.syntax_dir / "lean_workbook.jsonl"
        
        if not raw_path.exists():
            logger.warning(f"Lean Workbook dataset not found at {raw_path}")
            return out_path

        logger.info(f"Processing Lean Workbook dataset from {raw_path}...")
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        records = []
        for idx, item in enumerate(data):
            nl_stmt = item.get("natural_language_statement") or item.get("informal_statement", "")
            formal_stmt = item.get("formal_statement", "")
            records.append({
                "problem_id": f"wkbk_{idx}",
                "informal_statement": str(nl_stmt).strip(),
                "formal_statement": str(formal_stmt).strip(),
                "source": "lean_workbook"
            })
            
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        logger.info(f"Successfully exported {len(records)} Lean Workbook entries to {out_path}")
        return out_path

    def process_lean_dojo(self) -> Path:
        """Process LeanDojo dataset for Stage 2 RLCF."""
        dojo_files = list((self.data_root / "lean-dojo" / "data").glob("*.parquet"))
        out_path = self.rlcf_dir / "lean_dojo.jsonl"
        
        if not dojo_files:
            logger.warning(f"No LeanDojo parquet files found.")
            return out_path

        logger.info(f"Processing {len(dojo_files)} LeanDojo files for RLCF...")
        records = []
        for pfile in dojo_files:
            df = pd.read_parquet(pfile)
            for idx, row in df.iterrows():
                full_name = str(row.get("full_name", f"dojo_{idx}"))
                file_path = str(row.get("file_path", ""))
                
                # LeanDojo theorem premises
                records.append({
                    "problem_id": f"dojo_{full_name}",
                    "informal_statement": None,
                    "formal_statement": f"-- File: {file_path}\n-- Theorem: {full_name}",
                    "source": "lean_dojo"
                })
                
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        logger.info(f"Successfully exported {len(records)} LeanDojo entries to {out_path}")
        return out_path

    def process_minif2f(self) -> Path:
        """Process miniF2F Lean files for Stage 2 RLCF."""
        minif2f_dir = self.data_root / "miniF2F"
        out_path = self.rlcf_dir / "minif2f.jsonl"
        
        lean_files = list(minif2f_dir.rglob("*.lean"))
        valid_files = [f for f in lean_files if f.name in ("Valid.lean", "Test.lean")]
        
        if not valid_files:
            logger.warning("No miniF2F Lean files found.")
            return out_path

        logger.info(f"Extracting miniF2F benchmarks from {valid_files}...")
        records = []
        theorem_pattern = re.compile(
            r"(/\*.*?\*/\s*)?(theorem\s+[\s\S]+?:=\s*by\s*sorry)", re.DOTALL
        )
        
        for lfile in valid_files:
            content = lfile.read_text(encoding="utf-8")
            blocks = re.findall(r"(?:/--([\s\S]*?)-/)?\s*(theorem\s+([a-zA-Z0-9_]+)[\s\S]*?:=\s*by)", content)
            
            for doc, stmt, name in blocks:
                records.append({
                    "problem_id": f"minif2f_{name}",
                    "informal_statement": doc.strip() if doc else None,
                    "formal_statement": stmt.strip() + " sorry",
                    "source": f"minif2f_{lfile.stem.lower()}"
                })
                
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        logger.info(f"Successfully exported {len(records)} miniF2F benchmarks to {out_path}")
        return out_path

    def run_all(self) -> None:
        """Run standard preprocessing across all dataset stages."""
        logger.info("=== Starting Preprocessing for Syntax (Stage 1) ===")
        self.process_herald()
        self.process_lean_workbook()
        
        logger.info("\n=== Starting Preprocessing for RLCF (Stage 2) ===")
        self.process_lean_dojo()
        self.process_minif2f()
        logger.info("=== Dataset Preparation Complete ===")


if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.run_all()
