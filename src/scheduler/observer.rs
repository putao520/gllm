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
}

impl RuntimeObserver for BasicObserver {
    fn capture(&self) -> Result<SystemState, ObserverError> {
        Ok(self.last_state)
    }
}
