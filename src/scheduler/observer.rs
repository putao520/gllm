use super::jit_types::SystemState;

/// Error type for observer operations.
#[derive(Debug, Clone)]
pub enum ObserverError {
    BackendUnavailable(String),
}

impl std::fmt::Display for ObserverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendUnavailable(msg) => write!(f, "backend unavailable: {msg}"),
        }
    }
}

impl std::error::Error for ObserverError {}

/// Runtime observer trait.
pub trait RuntimeObserver {
    fn capture(&self) -> Result<SystemState, ObserverError>;
}

/// Basic observer that holds the last captured state.
/// The executor updates fields before calling capture().
pub struct BasicObserver {
    pub last_state: SystemState,
}

impl Default for BasicObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicObserver {
    pub fn new() -> Self {
        Self {
            last_state: SystemState::default(),
        }
    }

    /// Update resource metrics from external sources.
    /// Called by executor before policy decision.
    pub fn update_memory_pressure(&mut self, pressure: Result<f32, String>) -> Result<(), ObserverError> {
        match pressure {
            Ok(p) => {
                self.last_state.memory_pressure = p;
                Ok(())
            }
            Err(e) => Err(ObserverError::BackendUnavailable(e)),
        }
    }

    pub fn update_scheduler_metrics(
        &mut self,
        waiting_queue_len: usize,
        current_running_len: usize,
        current_batch_size: usize,
        mean_context_len: usize,
    ) {
        self.last_state.waiting_queue_len = waiting_queue_len;
        self.last_state.current_running_len = current_running_len;
        self.last_state.current_batch_size = current_batch_size;
        self.last_state.mean_context_len = mean_context_len;
    }

    pub fn update_kv_fragmentation(&mut self, fragmentation: f32) {
        self.last_state.kv_fragmentation = fragmentation;
    }

    pub fn update_swap_io_rate(&mut self, rate: f32) {
        self.last_state.swap_io_rate = rate;
    }

    pub fn update_logits_entropy(&mut self, entropy: f32) {
        self.last_state.logits_entropy = entropy;
    }

    pub fn update_attention_sparsity(&mut self, sparsity: f32) {
        self.last_state.attention_sparsity = sparsity;
    }
}

impl RuntimeObserver for BasicObserver {
    fn capture(&self) -> Result<SystemState, ObserverError> {
        Ok(self.last_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_updated_state() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.75)).unwrap();
        obs.update_scheduler_metrics(10, 5, 8, 128);
        obs.update_kv_fragmentation(0.3);
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.75).abs() < f32::EPSILON);
        assert_eq!(state.waiting_queue_len, 10);
        assert_eq!(state.current_running_len, 5);
        assert_eq!(state.current_batch_size, 8);
        assert_eq!(state.mean_context_len, 128);
        assert!((state.kv_fragmentation - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn memory_pressure_error_propagation() {
        let mut obs = BasicObserver::new();
        let result = obs.update_memory_pressure(Err("gpu unavailable".into()));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ObserverError::BackendUnavailable(_)));
    }

    #[test]
    fn phase2_setters() {
        let mut obs = BasicObserver::new();
        obs.update_swap_io_rate(42.5);
        obs.update_logits_entropy(3.14);
        obs.update_attention_sparsity(0.85);
        let state = obs.capture().unwrap();
        assert!((state.swap_io_rate - 42.5).abs() < f32::EPSILON);
        assert!((state.logits_entropy - 3.14).abs() < f32::EPSILON);
        assert!((state.attention_sparsity - 0.85).abs() < f32::EPSILON);
    }
}
