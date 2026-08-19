/// Serialise and persist agent results in the configured format.
use crate::{agent::AgentResult, config::Config};
use anyhow::{Context, Result};
use chrono::Utc;
use serde::Serialize;
use std::{fs, path::Path};
use tracing::info;

#[derive(Serialize)]
pub struct OutputRecord<'a> {
    pub title: &'a str,
    pub problem: &'a str,
    pub solved: bool,
    pub iters: usize,
    pub formal_code: &'a str,
    pub errors: &'a [String],
    pub timestamp: String,
}

pub fn write_result(config: &Config, result: &AgentResult) -> Result<()> {
    let record = OutputRecord {
        title: &config.title,
        problem: config.problem.trim(),
        solved: result.solved,
        iters: result.iters,
        formal_code: &result.final_code,
        errors: &result.final_repl.errors,
        timestamp: Utc::now().to_rfc3339(),
    };

    let path = config.output_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }

    let content = match config.output_format.as_str() {
        "yaml" => serde_yaml::to_string(&record).context("yaml serialise")?,
        "json" => serde_json::to_string_pretty(&record).context("json serialise")?,
        _ => format_txt(&record),
    };

    fs::write(&path, &content).with_context(|| format!("write {}", path.display()))?;
    info!(path = %path.display(), format = %config.output_format, "result written");
    Ok(())
}

fn format_txt(r: &OutputRecord) -> String {
    format!(
        "title:   {}\nproblem: {}\nsolved:  {}\niters:   {}\ntimestamp: {}\n\n--- Lean 4 Code ---\n{}\n\n--- Errors ---\n{}\n",
        r.title, r.problem, r.solved, r.iters, r.timestamp, r.formal_code,
        if r.errors.is_empty() { "none".into() } else { r.errors.join("\n") }
    )
}

/// Resolve absolute output path from a workspace root.
pub fn resolve_path(config: &Config, workspace: &Path) -> std::path::PathBuf {
    let rel = config.output_path();
    if rel.is_absolute() { rel } else { workspace.join(rel) }
}
