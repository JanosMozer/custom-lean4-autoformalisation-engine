mod agent;
mod config;
mod model;
mod output;
mod repl;
mod retrieval;

use agent::Agent;
use anyhow::{Context, Result};
use config::Config;
use model::InferenceClient;
use output::write_result;
use repl::LeanRepl;
use retrieval::Retriever;
use std::{env, path::PathBuf};
use tracing::{error, info};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("lean_engine=info".parse()?))
        .with_target(false)
        .compact()
        .init();

    let config_path = env::args().nth(1).unwrap_or_else(|| "engine/config.toml".into());
    let config = Config::load(&config_path)?;

    info!(title = %config.title, max_iters = config.max_iters, "engine start");

    let workspace = PathBuf::from(env::var("LEANBENCH_ROOT").unwrap_or_else(|_| ".".into()));
    let python = resolve_python(&workspace);
    let engine_dir = workspace.join("engine");
    let repl_script = engine_dir.join("scripts/repl_server.py");
    let retrieval_script = engine_dir.join("scripts/retrieval_server.py");

    // Spawn persistent subprocess servers.
    let repl = LeanRepl::spawn(&python, &workspace, repl_script.to_str().unwrap()).await?;
    let retriever = Retriever::spawn(
        &python,
        &workspace,
        retrieval_script.to_str().unwrap(),
        config.retrieval_k,
    )
    .await?;

    // Connect to the (externally managed) inference server.
    let client =
        InferenceClient::connect(&config.infer_host, config.infer_port, config.temperature, config.max_new_tokens)
            .await
            .context("connect inference server — is scripts/infer_server.py running?")?;

    let run_result = run_agent(&config, &client, &repl, &retriever, &workspace).await;

    // Graceful shutdown — always reached, even on error.
    info!("shutting down subprocesses");
    client.shutdown().await;
    retriever.shutdown().await;
    repl.shutdown().await;

    let result = run_result?;

    if result.solved {
        info!(iters = result.iters, "SOLVED — formal statement written");
    } else {
        error!(iters = result.iters, "UNSOLVED after max iterations");
        std::process::exit(1);
    }

    Ok(())
}

async fn run_agent(
    config: &Config,
    client: &InferenceClient,
    repl: &LeanRepl,
    retriever: &Retriever,
    workspace: &PathBuf,
) -> Result<agent::AgentResult> {
    let agent = Agent::new(config, client, repl, retriever);
    let result = agent.run().await.context("agent loop")?;

    let out_path = output::resolve_path(config, workspace);
    let mut cfg_abs = config.clone();
    cfg_abs.output_dir = out_path
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| config.output_dir.clone());

    write_result(&cfg_abs, &result)?;
    Ok(result)
}

fn resolve_python(workspace: &PathBuf) -> String {
    let venv = workspace.join(".venv/bin/python");
    if venv.exists() {
        return venv.to_string_lossy().into_owned();
    }
    "python3".into()
}
