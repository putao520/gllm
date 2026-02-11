use std::collections::HashMap;

use crate::scheduler::memory_manager::VirtualPageId;

pub type TokenId = u32;

/// 前缀匹配结果
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    pub matched_tokens: usize,
    pub matched_pages: Vec<VirtualPageId>,
}

/// KV Cache 前缀树索引 (ARCH-SCHED-PREFIX-INDEX)
#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<TokenId, TrieNode>,
    page_ref: Option<(VirtualPageId, usize)>,
}

#[derive(Debug, Default)]
pub struct KvPrefixIndex {
    root: TrieNode,
}

impl KvPrefixIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// O(n) 查找最长匹配前缀
    pub fn find_longest_prefix(&self, tokens: &[TokenId]) -> Option<PrefixMatch> {
        let mut node = &self.root;
        let mut matched_pages = Vec::new();
        let mut best_tokens = 0usize;
        let mut best_page_len = 0usize;

        for (idx, token) in tokens.iter().enumerate() {
            let Some(next) = node.children.get(token) else {
                break;
            };
            node = next;

            if let Some((page_id, offset_in_page)) = node.page_ref {
                // 命中页面有效性校验：offset 必须不超过当前前缀索引
                if offset_in_page <= idx {
                    if matched_pages.last().copied() != Some(page_id) {
                        matched_pages.push(page_id);
                    }
                    best_tokens = idx.saturating_add(1);
                    best_page_len = matched_pages.len();
                }
            }
        }

        if best_tokens == 0 {
            return None;
        }
        matched_pages.truncate(best_page_len);
        Some(PrefixMatch {
            matched_tokens: best_tokens,
            matched_pages,
        })
    }

    /// 插入新的 token 序列
    pub fn insert(&mut self, tokens: &[TokenId], pages: &[VirtualPageId]) {
        if tokens.is_empty() {
            return;
        }

        let mut node = &mut self.root;
        for (idx, token) in tokens.iter().enumerate() {
            node = node.children.entry(*token).or_default();
            if let Some(page_id) = pages.get(idx).copied() {
                node.page_ref = Some((page_id, idx));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_prefix_supports_append_reuse() {
        let mut index = KvPrefixIndex::new();
        let tokens = vec![11, 22, 33];
        let pages = vec![
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ];
        index.insert(&tokens, &pages);

        let matched = index.find_longest_prefix(&[11, 22, 33, 44]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages, pages);
    }

    #[test]
    fn longest_prefix_returns_none_on_miss() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[VirtualPageId::new(1, 0)]);
        assert!(index.find_longest_prefix(&[9, 9, 9]).is_none());
    }
}
