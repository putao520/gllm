use super::jit_types::SystemState;

/// Observer trait for capturing system state.
pub trait RuntimeObserver {
    fn capture(&self) -> SystemState;
}

/// Basic observer implementation.
pub struct BasicObserver {
    // We will inject closures or references to get real data
    // For MVP, we might just pass values in, or hold refs to components if possible.
    // However, to keep it decoupled and avoid borrow hell, we'll design it to be updated
    // or pull from a shared stats source.

    // For now, let's assume it gets updated by the executor before decision.
    pub last_state: SystemState,
}

impl BasicObserver {
    pub fn new() -> Self {
        Self {
            last_state: SystemState::default(),
        }
    }

    pub fn update(&mut self, state: SystemState) {
        self.last_state = state;
    }
}

impl RuntimeObserver for BasicObserver {
    fn capture(&self) -> SystemState {
        self.last_state
    }
}
