//! Token-ID driven thinking budget tracker (T3).
//!
//! SPEC: `SPEC/04-API-DESIGN.md §3.2 Thinking Budget` (REQ-API-THINKING-BUDGET).
//!
//! Responsibility
//! --------------
//! During generation, the executor samples one token at a time. Some models
//! (Qwen3 thinking mode, Gemma thinking variants, Kimi-K2 etc.) wrap an
//! internal chain-of-thought with dedicated `<thinking>` / `</thinking>`
//! special tokens. Users want to bound the cost of this phase with
//! `GenerationBuilder::thinking_budget(n)`. This module implements the
//! token-level state machine that:
//!
//! 1. Detects entry (`<thinking>`) / exit (`</thinking>`) by exact token ID.
//! 2. Counts thinking tokens so we can expose `thinking_token_count` in
//!    `GenerationResponse`.
//! 3. Signals a forced exit when the budget is exhausted, so the executor
//!    can overwrite the sampled token with `</thinking>` and return to the
//!    normal generation path (NO_SILENT_FALLBACK).
//!
//! The tracker is intentionally tokenizer-backed rather than text-based:
//! the text-matching approach in `crate::generation` was an island module
//! with no production caller, and text matching is fragile against partial
//! BPE merges (e.g. the token for `<thinking>` may not be a full substring
//! boundary in the decoded text).
//!
//! Design notes
//! ------------
//! - When the model's tokenizer has no `<thinking>` / `</thinking>` tokens,
//!   the tracker degrades to a permanent `Normal` state: every observe()
//!   returns `TokenClass::Normal` and `budget_exhausted()` is false. This
//!   is the explicit "model has no thinking" contract — not a silent
//!   fallback, since the caller is not promised anything about the budget
//!   in that case.
//! - The tracker itself is `#[derive(Debug, Clone)]`; construction requires
//!   a borrowed `TokenizerHandle` but the tracker does not retain it.

use crate::tokenizer::TokenizerHandle;

/// Classification of a single observed token.
///
/// Returned by [`ThinkingTracker::observe`] after each sampled token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenClass {
    /// Token is part of the model's normal output (non-thinking).
    Normal,
    /// Token is part of a thinking span (between `<thinking>` and
    /// `</thinking>`, inclusive of the opening tag but excluding the
    /// closing tag — the closing tag transitions back to `Normal`).
    Thinking,
    /// The thinking budget has been exhausted on this token. The executor
    /// MUST overwrite the sampled token with the `</thinking>` token ID
    /// returned by [`ThinkingTracker::end_token`] to force exit.
    ExitThinkingForced,
}

/// Thinking phase state (internal — exposed for tests / observability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingState {
    /// Normal generation phase (before `<thinking>` or after `</thinking>`).
    Normal,
    /// Inside the thinking span.
    Thinking,
}

/// Per-request thinking tracker (T3).
#[derive(Debug, Clone)]
pub struct ThinkingTracker {
    state: ThinkingState,
    thinking_count: usize,
    budget: Option<usize>,
    start_token: Option<u32>,
    end_token: Option<u32>,
}

impl ThinkingTracker {
    /// Construct a tracker. `budget == Some(0)` disables thinking entirely
    /// (any attempted entry into `<thinking>` is immediately force-exited).
    ///
    /// Looks up the `<thinking>` / `</thinking>` token IDs from the
    /// tokenizer. If either lookup fails (single-token encoding not
    /// produced), the tracker degrades to permanent `Normal` state.
    pub fn new(budget: Option<usize>, tokenizer: &TokenizerHandle) -> Self {
        let start_token = tokenizer.single_token_id("<thinking>");
        let end_token = tokenizer.single_token_id("</thinking>");
        Self {
            state: ThinkingState::Normal,
            thinking_count: 0,
            budget,
            start_token,
            end_token,
        }
    }

    /// Construct a tracker with explicit token IDs (used by unit tests that
    /// do not have a real tokenizer).
    pub fn with_tokens(
        budget: Option<usize>,
        start_token: Option<u32>,
        end_token: Option<u32>,
    ) -> Self {
        Self {
            state: ThinkingState::Normal,
            thinking_count: 0,
            budget,
            start_token,
            end_token,
        }
    }

    /// Advance the state machine with the freshly sampled token.
    ///
    /// Returns how the executor should treat `token`. See [`TokenClass`].
    pub fn observe(&mut self, token: u32) -> TokenClass {
        // Model has no thinking tokens → everything is Normal.
        if self.start_token.is_none() || self.end_token.is_none() {
            return TokenClass::Normal;
        }
        let start = self.start_token.expect("checked above");
        let end = self.end_token.expect("checked above");

        match self.state {
            ThinkingState::Normal => {
                if token == start {
                    // Enter thinking. If the budget is zero, the very next
                    // call will force-exit on the first thinking token; we
                    // still count the opening tag so the caller can see it.
                    self.state = ThinkingState::Thinking;
                    self.thinking_count += 1;
                    // Budget == 0 means "no thinking tokens allowed" — the
                    // opening tag itself consumes the only slot, so on the
                    // following observe we must force-exit immediately. The
                    // opening tag counts as thinking (it is emitted inside
                    // the thinking span).
                    if self.budget == Some(0) {
                        // Already exhausted — the caller should force an
                        // immediate exit on the next step.
                        return TokenClass::Thinking;
                    }
                    TokenClass::Thinking
                } else {
                    TokenClass::Normal
                }
            }
            ThinkingState::Thinking => {
                if token == end {
                    // Model emitted </thinking> on its own — simply exit.
                    self.state = ThinkingState::Normal;
                    return TokenClass::Normal;
                }
                // Budget check BEFORE counting: if we are already at the
                // budget ceiling, force the caller to exit on this token.
                if let Some(max) = self.budget {
                    if self.thinking_count >= max {
                        // Force exit. The caller replaces `token` with
                        // `end_token` and we transition back to Normal.
                        self.state = ThinkingState::Normal;
                        return TokenClass::ExitThinkingForced;
                    }
                }
                self.thinking_count += 1;
                TokenClass::Thinking
            }
        }
    }

    /// Returns the `</thinking>` token ID used to force-exit. The caller
    /// invokes this when [`observe`](Self::observe) returns
    /// [`TokenClass::ExitThinkingForced`] to overwrite the freshly sampled
    /// token. Guaranteed to be `Some` when `ExitThinkingForced` is
    /// returned, because force-exit can only happen after the tracker
    /// observed a real `<thinking>` token — which requires
    /// `start_token.is_some()` AND `end_token.is_some()`.
    pub fn end_token(&self) -> Option<u32> {
        self.end_token
    }

    /// Total number of tokens classified as `Thinking` so far. Does NOT
    /// include force-exit synthetic `</thinking>` (the executor writes
    /// that back to the output stream but it is not counted as a thinking
    /// token — it terminates the thinking phase).
    pub fn thinking_count(&self) -> usize {
        self.thinking_count
    }

    /// True once a force-exit has been issued (state went back to Normal
    /// via `ExitThinkingForced`). Useful for assertions / telemetry.
    pub fn budget_exhausted(&self) -> bool {
        // The tracker transitions back to Normal on force-exit, so a pure
        // state check is insufficient; we rely on count vs budget.
        matches!(self.budget, Some(max) if self.thinking_count >= max)
            && self.start_token.is_some()
    }

    /// Current state (for tests / diagnostics).
    pub fn state(&self) -> ThinkingState {
        self.state
    }

    /// True if the model has thinking tokens registered (i.e. the tracker
    /// is not in the permanent `Normal` degraded state).
    pub fn has_thinking_support(&self) -> bool {
        self.start_token.is_some() && self.end_token.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: u32 = 1000;
    const END: u32 = 1001;

    #[test]
    fn thinking_tracker_transitions_on_tags() {
        let mut tracker = ThinkingTracker::with_tokens(None, Some(START), Some(END));
        // Pre-thinking: normal tokens stay Normal.
        assert_eq!(tracker.observe(42), TokenClass::Normal);
        assert_eq!(tracker.state(), ThinkingState::Normal);

        // Enter thinking via <thinking>.
        assert_eq!(tracker.observe(START), TokenClass::Thinking);
        assert_eq!(tracker.state(), ThinkingState::Thinking);

        // Two tokens inside the span.
        assert_eq!(tracker.observe(100), TokenClass::Thinking);
        assert_eq!(tracker.observe(101), TokenClass::Thinking);

        // Exit via </thinking> — that token itself is Normal (terminator).
        assert_eq!(tracker.observe(END), TokenClass::Normal);
        assert_eq!(tracker.state(), ThinkingState::Normal);

        // Post-thinking tokens are Normal.
        assert_eq!(tracker.observe(200), TokenClass::Normal);

        // Count = 3 (<thinking> + two inner tokens; </thinking> is NOT
        // counted because it terminates).
        assert_eq!(tracker.thinking_count(), 3);
    }

    #[test]
    fn thinking_tracker_forces_exit_on_budget() {
        // Budget = 3: model can emit <thinking> + 2 inner tokens, then the
        // third inner token must be force-exited.
        let mut tracker = ThinkingTracker::with_tokens(Some(3), Some(START), Some(END));
        assert_eq!(tracker.observe(START), TokenClass::Thinking); // 1
        assert_eq!(tracker.observe(111), TokenClass::Thinking); // 2
        assert_eq!(tracker.observe(222), TokenClass::Thinking); // 3 (ceiling)

        // 4th thinking token — budget reached, must force exit.
        assert_eq!(tracker.observe(333), TokenClass::ExitThinkingForced);
        assert_eq!(tracker.state(), ThinkingState::Normal);
        assert!(tracker.budget_exhausted());
        assert_eq!(tracker.end_token(), Some(END));

        // After force-exit we are Normal again.
        assert_eq!(tracker.observe(444), TokenClass::Normal);
        // The forced-exit token is NOT counted as thinking.
        assert_eq!(tracker.thinking_count(), 3);
    }

    #[test]
    fn thinking_tracker_noop_when_model_has_no_thinking() {
        // No start/end tokens → every observation is Normal, no counting.
        let mut tracker = ThinkingTracker::with_tokens(Some(5), None, None);
        assert_eq!(tracker.observe(42), TokenClass::Normal);
        assert_eq!(tracker.observe(1000), TokenClass::Normal);
        assert_eq!(tracker.observe(999), TokenClass::Normal);
        assert_eq!(tracker.thinking_count(), 0);
        assert!(!tracker.has_thinking_support());
        assert!(!tracker.budget_exhausted());
    }

    #[test]
    fn thinking_tracker_noop_when_only_start_token_available() {
        // Pathological tokenizer: start token resolves but end token does
        // not. We must not force-exit because we have no token ID to write.
        let mut tracker = ThinkingTracker::with_tokens(Some(1), Some(START), None);
        assert_eq!(tracker.observe(START), TokenClass::Normal);
        assert!(!tracker.has_thinking_support());
    }

    #[test]
    fn thinking_tracker_budget_zero_forces_exit_on_next_token() {
        // Budget == 0: <thinking> itself counts, the very next thinking
        // token is force-exited.
        let mut tracker = ThinkingTracker::with_tokens(Some(0), Some(START), Some(END));
        assert_eq!(tracker.observe(START), TokenClass::Thinking); // counts as 1, but budget=0
        // thinking_count (1) >= budget (0) → next observe forces exit.
        assert_eq!(tracker.observe(99), TokenClass::ExitThinkingForced);
        assert_eq!(tracker.state(), ThinkingState::Normal);
    }

    #[test]
    fn thinking_tracker_no_budget_allows_unbounded_thinking() {
        let mut tracker = ThinkingTracker::with_tokens(None, Some(START), Some(END));
        assert_eq!(tracker.observe(START), TokenClass::Thinking);
        for i in 0..100 {
            assert_eq!(tracker.observe(i), TokenClass::Thinking);
        }
        assert_eq!(tracker.observe(END), TokenClass::Normal);
        assert_eq!(tracker.thinking_count(), 101); // <thinking> + 100
        assert!(!tracker.budget_exhausted());
    }
}
