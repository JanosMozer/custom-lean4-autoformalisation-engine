/// Agentic compiler-feedback loop (A + B system, up to max_iters repair turns).
use crate::{
    config::Config,
    model::{InferenceClient, Message},
    repl::{extract_unknown_idents, LeanRepl, ReplResult},
    retrieval::Retriever,
};
use anyhow::Result;
use tracing::{info, warn};

pub struct AgentResult {
    pub solved: bool,
    pub iters: usize,
    pub final_code: String,
    pub final_repl: ReplResult,
}

const SYSTEM_PROMPT: &str = "\
You are a Lean 4 autoformalization expert. \
Translate the given informal mathematical statement into a single valid Lean 4 theorem \
declaration ending with `:= by sorry`. \
Output only the Lean 4 code, no explanation, no markdown fences.";

pub struct Agent<'a> {
    config: &'a Config,
    client: &'a InferenceClient,
    repl: &'a LeanRepl,
    retriever: &'a Retriever,
}

impl<'a> Agent<'a> {
    pub fn new(
        config: &'a Config,
        client: &'a InferenceClient,
        repl: &'a LeanRepl,
        retriever: &'a Retriever,
    ) -> Self {
        Self { config, client, repl, retriever }
    }

    pub async fn run(&self) -> Result<AgentResult> {
        let problem = self.config.problem.trim().to_owned();

        // System B: grounding premises for Turn 1
        info!("System B: retrieving premises");
        let premises = self.retriever.retrieve_b(&problem).await.unwrap_or_default();
        let premise_block = Retriever::format_premises(&premises);

        let augmented_user = if premise_block.is_empty() {
            problem.clone()
        } else {
            format!("-- Relevant Mathlib declarations:\n{premise_block}\n\n{problem}")
        };

        let mut messages: Vec<Message> = vec![
            Message { role: "system".into(), content: SYSTEM_PROMPT.into() },
            Message { role: "user".into(), content: augmented_user },
        ];

        let mut final_code = String::new();
        let mut final_repl = ReplResult { ok: false, errors: vec![], goal: None };

        for iter in 1..=self.config.max_iters {
            info!(iter, "generating completion");
            let completion = self.client.generate(&messages).await?;
            let code = strip_fences(&completion);

            let lean_input = format!("{}{}", self.config.lean_header, code);
            info!(iter, "REPL check");
            let repl_res = self.repl.run(&lean_input).await?;

            if repl_res.ok {
                info!(iter, "solved");
                return Ok(AgentResult {
                    solved: true,
                    iters: iter,
                    final_code: code,
                    final_repl: repl_res,
                });
            }

            warn!(iter, errors = ?repl_res.errors, "compile failed");

            // System A: batch lookup for all unknown idents in one round-trip.
            let idents = extract_unknown_idents(&repl_res.errors);
            let lookup = self.retriever.lookup_a_batch(&idents).await.unwrap_or_default();

            let mut repair_ctx = String::new();
            for ident in &idents {
                if let Some(cands) = lookup.get(ident) {
                    if !cands.is_empty() {
                        repair_ctx.push_str(&format!(
                            "Did you mean '{}' → {}?\n",
                            ident,
                            cands.join(", ")
                        ));
                    }
                }
            }

            let error_text = repl_res.errors.join("\n").chars().take(800).collect::<String>();
            let repair_msg = format!(
                "Compilation failed (iteration {iter}).\nErrors:\n{error_text}\n{repair_ctx}\n\
                 Please correct the Lean 4 statement. Output only the corrected code."
            );

            messages.push(Message { role: "assistant".into(), content: completion });
            messages.push(Message { role: "user".into(), content: repair_msg });

            final_code = code;
            final_repl = repl_res;
        }

        Ok(AgentResult { solved: false, iters: self.config.max_iters, final_code, final_repl })
    }
}

fn strip_fences(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_prefix("```lean").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim().to_owned()
}
