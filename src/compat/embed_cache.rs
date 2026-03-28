#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::collections::{HashMap, VecDeque};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
/// Tier I.6: LRU Cache with capacity 256 for fused Embed-RmsNorm ops.
pub struct EmbedNormCache {
    map: HashMap<u32, Vec<u8>>,
    order: VecDeque<u32>,
    capacity: usize,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl EmbedNormCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    pub fn get(&mut self, token: u32) -> Option<Vec<u8>> {
        if self.map.contains_key(&token) {
            if let Some(pos) = self.order.iter().position(|&t| t == token) {
                self.order.remove(pos);
                self.order.push_back(token);
            }
            self.map.get(&token).cloned()
        } else {
            None
        }
    }

    pub fn insert(&mut self, token: u32, data: Vec<u8>) {
        if self.map.contains_key(&token) {
            if let Some(pos) = self.order.iter().position(|&t| t == token) {
                self.order.remove(pos);
            }
            self.order.push_back(token);
            self.map.insert(token, data);
            return;
        }
        if self.map.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.order.push_back(token);
        self.map.insert(token, data);
    }
    
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}
