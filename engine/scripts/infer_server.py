#!/usr/bin/env python3
"""TCP JSON-lines inference server. Loads the QLoRA model once, serves completions."""
import asyncio
import json
import logging
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("infer_server")

MODEL_PATH = os.environ.get("MODEL_PATH", "path_to_model")
HOST = os.environ.get("INFER_HOST", "127.0.0.1")
PORT = int(os.environ.get("INFER_PORT", "9876"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str):
    log.info("loading tokenizer from %s", path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    log.info("loading base model (NF4)")
    base = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        log.info("loading PEFT adapter")
        model = PeftModel.from_pretrained(base, path)
        model.eval()
    else:
        model = base
    log.info("model ready on %s", DEVICE)
    return tok, model


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 tok, model) -> None:
    addr = writer.get_extra_info("peername")
    try:
        line = await reader.readline()
        req = json.loads(line.decode())
        messages = req.get("messages", [])
        temperature = float(req.get("temperature", 0.7))
        max_new_tokens = int(req.get("max_new_tokens", 512))

        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(DEVICE)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        completion = tok.decode(new_tokens, skip_special_tokens=True)

        resp = json.dumps({"completion": completion}) + "\n"
        writer.write(resp.encode())
        await writer.drain()
        log.info("served %d new tokens to %s", len(new_tokens), addr)
    except Exception as exc:
        log.exception("error handling request from %s", addr)
        writer.write((json.dumps({"error": str(exc)}) + "\n").encode())
        await writer.drain()
    finally:
        writer.close()


async def main():
    tok, model = load_model(MODEL_PATH)
    server = await asyncio.start_server(
        lambda r, w: handle(r, w, tok, model), HOST, PORT
    )
    log.info("inference server listening on %s:%d", HOST, PORT)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
