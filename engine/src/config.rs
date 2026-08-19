use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub title: String,
    pub problem: String,
    pub model_path: String,
    pub max_iters: usize,
    pub output_format: String,
    pub output_dir: String,
    pub lean_header: String,
    pub retrieval_emb: String,
    pub retrieval_meta: String,
    pub retrieval_bm25: String,
    pub retrieval_k: usize,
    pub infer_host: String,
    pub infer_port: u16,
    pub temperature: f32,
    pub max_new_tokens: usize,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config: {path}"))?;
        toml::from_str(&raw).context("parsing config.toml")
    }

    pub fn output_path(&self) -> PathBuf {
        let ext = match self.output_format.as_str() {
            "yaml" => "yaml",
            "json" => "json",
            _ => "txt",
        };
        PathBuf::from(&self.output_dir)
            .join(&self.title)
            .with_extension(ext)
    }
}
