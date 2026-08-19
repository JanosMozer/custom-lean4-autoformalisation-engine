/// Persistent REPL subprocess client — imports Mathlib once, amortises over all calls.
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tracing::{debug, info, instrument, warn};

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ReplResult {
    pub ok: bool,
    pub errors: Vec<String>,
    pub goal: Option<String>,
}

struct Inner {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// Wraps a single long-lived `repl_server.py` process.
pub struct LeanRepl {
    inner: Arc<Mutex<Inner>>,
}

impl LeanRepl {
    pub async fn spawn(python: &str, workspace: &PathBuf, server_script: &str) -> Result<Self> {
        let mut child = Command::new(python)
            .current_dir(workspace)
            .env("LEANBENCH_ROOT", workspace)
            .arg(server_script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn repl_server {server_script}"))?;

        let stdin = child.stdin.take().context("repl stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("repl stdout")?);
        info!("repl_server spawned");
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner { child, stdin, stdout })),
        })
    }

    #[instrument(skip(self, code), fields(code_len = code.len()))]
    pub async fn run(&self, code: &str) -> Result<ReplResult> {
        let req = serde_json::json!({ "code": code });
        let mut payload = req.to_string();
        payload.push('\n');

        let mut guard = self.inner.lock().await;
        guard.stdin.write_all(payload.as_bytes()).await.context("write to repl_server")?;

        let mut line = String::new();
        guard.stdout.read_line(&mut line).await.context("read from repl_server")?;
        drop(guard);

        if line.is_empty() {
            bail!("repl_server closed unexpectedly");
        }
        let res: ReplResult = serde_json::from_str(line.trim()).context("parse REPL result")?;
        debug!(ok = res.ok, n_errors = res.errors.len(), "REPL check done");
        Ok(res)
    }

    /// Send shutdown signal and wait for the process to exit.
    pub async fn shutdown(self) {
        let mut guard = self.inner.lock().await;
        let _ = guard.stdin.write_all(b"{\"shutdown\":true}\n").await;
        let _ = guard.stdin.flush().await;
        drop(guard);
        // Give the server a moment to exit cleanly.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let mut guard = self.inner.lock().await;
        match guard.child.try_wait() {
            Ok(Some(s)) => info!(status = ?s, "repl_server exited"),
            _ => {
                warn!("repl_server did not exit — killing");
                let _ = guard.child.kill().await;
            }
        }
    }
}

/// Extract unknown-identifier names from Lean compiler error messages.
pub fn extract_unknown_idents(errors: &[String]) -> Vec<String> {
    let re = regex::Regex::new(r"unknown (?:identifier|constant) '([^']+)'").unwrap();
    errors
        .iter()
        .flat_map(|e| re.captures_iter(e).map(|c| c[1].to_owned()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect()
}
