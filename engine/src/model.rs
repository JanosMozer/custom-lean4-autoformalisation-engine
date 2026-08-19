/// Async client for the Python inference server — reuses a single TCP connection.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tracing::{debug, info, instrument};

#[derive(Serialize)]
struct InferRequest<'a> {
    messages: &'a [Message],
    temperature: f32,
    max_new_tokens: usize,
}

#[derive(Serialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
struct InferResponse {
    completion: Option<String>,
    error: Option<String>,
}

struct Conn {
    writer: tokio::net::tcp::OwnedWriteHalf,
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
}

/// Holds one persistent TCP connection to the inference server.
/// Reconnects automatically if the connection drops.
pub struct InferenceClient {
    addr: String,
    pub temperature: f32,
    pub max_new_tokens: usize,
    conn: Arc<Mutex<Option<Conn>>>,
}

impl InferenceClient {
    pub async fn connect(host: &str, port: u16, temperature: f32, max_new_tokens: usize) -> Result<Self> {
        let addr = format!("{host}:{port}");
        let conn = Self::open_conn(&addr).await?;
        info!(%addr, "inference server connected");
        Ok(Self {
            addr,
            temperature,
            max_new_tokens,
            conn: Arc::new(Mutex::new(Some(conn))),
        })
    }

    async fn open_conn(addr: &str) -> Result<Conn> {
        let stream = TcpStream::connect(addr)
            .await
            .with_context(|| format!("connect to inference server {addr}"))?;
        let (r, w) = stream.into_split();
        Ok(Conn { writer: w, reader: BufReader::new(r) })
    }

    #[instrument(skip(self, messages), fields(n_msgs = messages.len()))]
    pub async fn generate(&self, messages: &[Message]) -> Result<String> {
        let req = InferRequest {
            messages,
            temperature: self.temperature,
            max_new_tokens: self.max_new_tokens,
        };
        let mut payload = serde_json::to_string(&req)?;
        payload.push('\n');

        let mut guard = self.conn.lock().await;

        // Reconnect if connection was lost.
        if guard.is_none() {
            *guard = Some(Self::open_conn(&self.addr).await?);
            info!("inference server reconnected");
        }

        let conn = guard.as_mut().unwrap();
        if let Err(e) = conn.writer.write_all(payload.as_bytes()).await {
            // Connection broken — drop and retry once.
            *guard = None;
            drop(guard);
            bail!("inference write failed ({e}); will reconnect on next call");
        }

        let mut line = String::new();
        if conn.reader.read_line(&mut line).await? == 0 {
            *guard = None;
            bail!("inference server closed connection");
        }

        let resp: InferResponse =
            serde_json::from_str(line.trim()).context("parse inference response")?;
        if let Some(err) = resp.error {
            bail!("inference server error: {err}");
        }
        let completion = resp.completion.unwrap_or_default();
        debug!(len = completion.len(), "received completion");
        Ok(completion)
    }

    pub async fn shutdown(&self) {
        // The inference server is external — just drop the connection; it manages its own lifecycle.
        let mut guard = self.conn.lock().await;
        *guard = None;
        info!("inference client disconnected");
    }
}
