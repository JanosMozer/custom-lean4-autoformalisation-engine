/// Persistent retrieval subprocess client — loads index once, batches System A lookups.
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tracing::{debug, info, instrument, warn};

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct Premise {
    pub name: String,
    pub signature: String,
    pub slogan: String,
}

struct Inner {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// Wraps a single long-lived `retrieval_server.py` process.
pub struct Retriever {
    inner: Arc<Mutex<Inner>>,
    retrieval_k: usize,
}

impl Retriever {
    pub async fn spawn(
        python: &str,
        workspace: &PathBuf,
        server_script: &str,
        k: usize,
    ) -> Result<Self> {
        let mut child = Command::new(python)
            .current_dir(workspace)
            .env("LEANBENCH_ROOT", workspace)
            .arg(server_script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn retrieval_server {server_script}"))?;

        let stdin = child.stdin.take().context("retrieval stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("retrieval stdout")?);
        info!("retrieval_server spawned");
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner { child, stdin, stdout })),
            retrieval_k: k,
        })
    }

    async fn roundtrip(&self, req: serde_json::Value) -> Result<String> {
        let mut payload = req.to_string();
        payload.push('\n');

        let mut guard = self.inner.lock().await;
        guard.stdin.write_all(payload.as_bytes()).await.context("write to retrieval_server")?;

        let mut line = String::new();
        guard.stdout.read_line(&mut line).await.context("read from retrieval_server")?;
        drop(guard);

        if line.is_empty() {
            bail!("retrieval_server closed unexpectedly");
        }
        Ok(line)
    }

    /// System B — hybrid dense+BM25 retrieval for initial prompt grounding.
    #[instrument(skip(self, query), fields(query_len = query.len()))]
    pub async fn retrieve_b(&self, query: &str) -> Result<Vec<Premise>> {
        let line = self
            .roundtrip(serde_json::json!({"op":"retrieve","query":query,"k":self.retrieval_k}))
            .await?;
        let premises: Vec<Premise> =
            serde_json::from_str(line.trim()).context("parse system-B output")?;
        debug!(n = premises.len(), "system-B premises retrieved");
        Ok(premises)
    }

    /// System A — batch fuzzy lookup for all unknown idents simultaneously.
    #[instrument(skip(self, idents), fields(n = idents.len()))]
    pub async fn lookup_a_batch(&self, idents: &[String]) -> Result<HashMap<String, Vec<String>>> {
        if idents.is_empty() {
            return Ok(HashMap::new());
        }
        let line = self
            .roundtrip(serde_json::json!({"op":"lookup","idents":idents,"k":5}))
            .await?;
        let map: HashMap<String, Vec<String>> =
            serde_json::from_str(line.trim()).context("parse system-A output")?;
        debug!(n = map.len(), "system-A batch done");
        Ok(map)
    }

    /// Format System-B premises into a context block for the initial prompt.
    pub fn format_premises(premises: &[Premise]) -> String {
        premises
            .iter()
            .map(|p| format!("-- {}\n{}", p.slogan, p.signature))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Send shutdown signal and wait for the process to exit.
    pub async fn shutdown(self) {
        let mut guard = self.inner.lock().await;
        let _ = guard.stdin.write_all(b"{\"op\":\"shutdown\"}\n").await;
        let _ = guard.stdin.flush().await;
        drop(guard);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let mut guard = self.inner.lock().await;
        match guard.child.try_wait() {
            Ok(Some(s)) => info!(status = ?s, "retrieval_server exited"),
            _ => {
                warn!("retrieval_server did not exit — killing");
                let _ = guard.child.kill().await;
            }
        }
    }
}
