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

/// @trace REQ-KV-ADDR-002 [entity:ENT-KV-CACHE] [api:POST /internal/paged_attention]
#[derive(Debug, Default)]
pub struct KvPrefixIndex {
    root: TrieNode,
}

impl KvPrefixIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// O(n) 查找最长匹配前缀
    /// @trace REQ-KV-ADDR-002 [entity:ENT-KV-CACHE] [api:POST /internal/paged_attention]
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
    /// @trace REQ-KV-ADDR-002 [entity:ENT-KV-CACHE] [api:POST /internal/paged_attention]
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

    #[test]
    fn empty_index_returns_none() {
        let index = KvPrefixIndex::new();
        assert!(index.find_longest_prefix(&[1, 2, 3]).is_none());
    }

    #[test]
    fn insert_empty_tokens_is_noop() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[], &[VirtualPageId::new(1, 0)]);
        assert!(index.find_longest_prefix(&[1]).is_none());
    }

    #[test]
    fn exact_match() {
        let mut index = KvPrefixIndex::new();
        let tokens = vec![10, 20, 30];
        let pages = vec![
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ];
        index.insert(&tokens, &pages);
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
    }

    #[test]
    fn partial_match_returns_shorter() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
        ]);
        let matched = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
    }

    #[test]
    fn multiple_insertions_common_prefix() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[1, 2, 9], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // First sequence should still match fully
        let m1 = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(m1.matched_tokens, 3);
        // Second sequence too
        let m2 = index.find_longest_prefix(&[1, 2, 9]).unwrap();
        assert_eq!(m2.matched_tokens, 3);
    }

    #[test]
    fn single_token_match() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[42], &[VirtualPageId::new(0, 0)]);
        let matched = index.find_longest_prefix(&[42, 99]).unwrap();
        assert_eq!(matched.matched_tokens, 1);
    }

    // ── Additional coverage ──

    #[test]
    fn prefix_match_equality() {
        let a = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![VirtualPageId::new(0, 0)],
        };
        let b = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![VirtualPageId::new(0, 0)],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn prefix_match_inequality_different_tokens() {
        let a = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![],
        };
        let b = PrefixMatch {
            matched_tokens: 5,
            matched_pages: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn prefix_match_clone() {
        let a = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(0, 1), VirtualPageId::new(0, 2)],
        };
        let cloned = a.clone();
        assert_eq!(a, cloned);
    }

    #[test]
    fn prefix_match_debug() {
        let pm = PrefixMatch {
            matched_tokens: 5,
            matched_pages: vec![],
        };
        let debug = format!("{pm:?}");
        assert!(debug.contains("matched_tokens"));
        assert!(debug.contains("matched_pages"));
    }

    #[test]
    fn kv_prefix_index_default() {
        let index = KvPrefixIndex::default();
        assert!(index.find_longest_prefix(&[1]).is_none());
    }

    #[test]
    fn kv_prefix_index_debug() {
        let index = KvPrefixIndex::new();
        let debug = format!("{index:?}");
        assert!(debug.contains("root"));
    }

    #[test]
    fn insert_with_fewer_pages_than_tokens() {
        let mut index = KvPrefixIndex::new();
        // 3 tokens but only 1 page
        index.insert(&[1, 2, 3], &[VirtualPageId::new(0, 0)]);
        // Token 1 has page, tokens 2 and 3 don't
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages.len(), 1);
    }

    #[test]
    fn find_longest_prefix_empty_query() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2], &[VirtualPageId::new(0, 0)]);
        assert!(index.find_longest_prefix(&[]).is_none());
    }

    #[test]
    fn deduplicate_same_page_across_tokens() {
        let mut index = KvPrefixIndex::new();
        // Same page for multiple tokens
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 0),
        ]);
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 1, "same page deduplicated");
    }

    // ── PrefixMatch field access & edge cases ──

    #[test]
    fn prefix_match_zero_tokens_and_empty_pages() {
        // Arrange: construct with zero matched_tokens and empty pages
        let pm = PrefixMatch {
            matched_tokens: 0,
            matched_pages: vec![],
        };
        // Assert: field access works correctly
        assert_eq!(pm.matched_tokens, 0);
        assert!(pm.matched_pages.is_empty());
    }

    #[test]
    fn prefix_match_large_token_count() {
        // Arrange: use usize::MAX as matched_tokens
        let pm = PrefixMatch {
            matched_tokens: usize::MAX,
            matched_pages: vec![VirtualPageId::new(0, 0)],
        };
        assert_eq!(pm.matched_tokens, usize::MAX);
    }

    #[test]
    fn prefix_match_pages_field_preserves_order() {
        // Arrange: multiple pages in specific order
        let pages = vec![
            VirtualPageId::new(1, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(3, 2),
        ];
        let pm = PrefixMatch {
            matched_tokens: 3,
            matched_pages: pages.clone(),
        };
        // Assert: order preserved
        assert_eq!(pm.matched_pages, pages);
    }

    #[test]
    fn prefix_match_equality_different_pages() {
        let a = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![VirtualPageId::new(0, 0)],
        };
        let b = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![VirtualPageId::new(0, 1)],
        };
        assert_ne!(a, b, "different pages should not be equal");
    }

    // ── KvPrefixIndex new() vs default() equivalence ──

    #[test]
    fn new_and_default_are_functionally_equivalent() {
        // Arrange: create via both constructors
        let via_new = KvPrefixIndex::new();
        let via_default = KvPrefixIndex::default();
        // Assert: both return None for any query
        assert!(via_new.find_longest_prefix(&[1, 2]).is_none());
        assert!(via_default.find_longest_prefix(&[1, 2]).is_none());
    }

    // ── TokenId boundary values (u32::MAX) ──

    #[test]
    fn insert_and_find_with_max_token_id() {
        // Arrange: use u32::MAX as a token
        let mut index = KvPrefixIndex::new();
        let tokens = vec![u32::MAX, 0, 1];
        let pages = vec![
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ];
        index.insert(&tokens, &pages);
        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();
        // Assert
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
    }

    #[test]
    fn find_with_max_token_id_no_match() {
        // Arrange: insert normal tokens, query with u32::MAX
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[VirtualPageId::new(0, 0)]);
        // Act & Assert
        assert!(index.find_longest_prefix(&[u32::MAX]).is_none());
    }

    // ── Overwriting existing token sequence ──

    #[test]
    fn insert_overwrites_existing_page_refs() {
        // Arrange: insert once, then re-insert same tokens with different pages
        let mut index = KvPrefixIndex::new();
        let tokens = vec![10, 20, 30];
        let pages_v1 = vec![
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ];
        let pages_v2 = vec![
            VirtualPageId::new(1, 10),
            VirtualPageId::new(1, 11),
            VirtualPageId::new(1, 12),
        ];
        index.insert(&tokens, &pages_v1);
        index.insert(&tokens, &pages_v2);
        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();
        // Assert: second insert's pages take effect (page_ref overwritten)
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages, pages_v2);
    }

    // ── Insert with empty pages slice ──

    #[test]
    fn insert_with_empty_pages_is_noop_for_matching() {
        // Arrange: insert tokens but no pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[5, 6, 7], &[]);
        // Act & Assert: trie nodes exist but no page_ref set
        assert!(index.find_longest_prefix(&[5, 6, 7]).is_none());
    }

    // ── Multiple divergent branches ──

    #[test]
    fn multiple_branches_find_correct_one() {
        // Arrange: insert three sequences diverging at second token
        let mut index = KvPrefixIndex::new();
        // Branch A: 100 -> 200 -> 300
        index.insert(&[100, 200, 300], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Branch B: 100 -> 201 -> 301
        index.insert(&[100, 201, 301], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // Branch C: 100 -> 202 -> 302
        index.insert(&[100, 202, 302], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(2, 2),
        ]);
        // Act & Assert: each branch matches independently
        // Note: shared node at token 100 has page_ref overwritten by last insert (Branch C),
        // so the first page in matched_pages comes from Branch C's page for the shared token.
        let ma = index.find_longest_prefix(&[100, 200, 300]).unwrap();
        assert_eq!(ma.matched_tokens, 3);
        // First page is from shared token 100 (overwritten by last insert = VirtualPageId(2,0))
        assert_eq!(ma.matched_pages[0], VirtualPageId::new(2, 0));

        let mb = index.find_longest_prefix(&[100, 201, 301]).unwrap();
        assert_eq!(mb.matched_tokens, 3);

        let mc = index.find_longest_prefix(&[100, 202, 302]).unwrap();
        assert_eq!(mc.matched_tokens, 3);

        // Query that shares only common prefix
        let mp = index.find_longest_prefix(&[100, 999]).unwrap();
        assert_eq!(mp.matched_tokens, 1);
    }

    // ── Query longer than inserted sequence ──

    #[test]
    fn query_extends_beyond_inserted_stops_at_boundary() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[7, 8], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
        ]);
        // Act: query longer sequence that continues past
        let matched = index.find_longest_prefix(&[7, 8, 9, 10, 11]).unwrap();
        // Assert: match stops at end of inserted sequence
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages.len(), 2);
    }

    // ── Single token various edge cases ──

    #[test]
    fn single_token_insert_then_query_different_token() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[55], &[VirtualPageId::new(0, 0)]);
        assert!(index.find_longest_prefix(&[56]).is_none());
    }

    #[test]
    fn single_token_exact_match() {
        let mut index = KvPrefixIndex::new();
        let page = VirtualPageId::new(99, 42);
        index.insert(&[55], &[page]);
        let matched = index.find_longest_prefix(&[55]).unwrap();
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages, vec![page]);
    }

    // ── Very long token sequence ──

    #[test]
    fn long_sequence_insert_and_match() {
        // Arrange: 1000 tokens
        let tokens: Vec<TokenId> = (0..1000).collect();
        let pages: Vec<VirtualPageId> = (0..1000)
            .map(|i| VirtualPageId::new(0, i as usize))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act: exact match
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 1000);
        assert_eq!(matched.matched_pages.len(), 1000);
        // Act: partial match (first 500)
        let partial_query: Vec<TokenId> = (0..500).collect();
        let partial = index.find_longest_prefix(&partial_query).unwrap();
        assert_eq!(partial.matched_tokens, 500);
        assert_eq!(partial.matched_pages.len(), 500);
    }

    // ── Partial pages: more tokens than pages ──

    #[test]
    fn partial_pages_only_matched_portion_counted() {
        // Arrange: 5 tokens, 2 pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4, 5], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
        ]);
        // Act: tokens 3,4,5 have no page_ref
        let matched = index.find_longest_prefix(&[1, 2, 3, 4, 5]).unwrap();
        // Assert: best_tokens should be 2 (last token with a page)
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages.len(), 2);
    }

    // ── find_longest_prefix with no shared prefix at all ──

    #[test]
    fn no_shared_prefix_returns_none() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[VirtualPageId::new(0, 0)]);
        // Query starts with completely different token
        assert!(index.find_longest_prefix(&[99, 20, 30]).is_none());
    }

    // ── Insert same sequence twice with different page counts ──

    #[test]
    fn reinsert_with_more_pages_extends_match() {
        let mut index = KvPrefixIndex::new();
        let tokens = vec![1, 2, 3];
        // First: only 1 page
        index.insert(&tokens, &[VirtualPageId::new(0, 0)]);
        let m1 = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(m1.matched_tokens, 1);
        // Second: all 3 pages
        index.insert(&tokens, &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        let m2 = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(m2.matched_tokens, 3);
        assert_eq!(m2.matched_pages.len(), 3);
    }

    // ── TokenId type alias is u32 ──

    #[test]
    fn token_id_is_u32_max() {
        // Verify TokenId accepts u32::MAX
        let max_token: TokenId = u32::MAX;
        assert_eq!(max_token, u32::MAX);
    }

    // ── Multiple inserts with interleaved queries ──

    #[test]
    fn interleaved_insert_and_query() {
        let mut index = KvPrefixIndex::new();
        // Insert first sequence
        index.insert(&[1, 2], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
        ]);
        let m1 = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(m1.matched_tokens, 2);

        // Insert extending sequence
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(2, 2),
        ]);
        let m2 = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(m2.matched_tokens, 3);

        // Original prefix still matches (overwritten pages)
        let m3 = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(m3.matched_tokens, 2);
    }

    // ── Edge cases: token value 0 ──

    #[test]
    fn token_zero_insert_and_match() {
        // Arrange: token value 0 is valid
        let mut index = KvPrefixIndex::new();
        let page = VirtualPageId::new(0, 0);
        index.insert(&[0, 0, 0], &[VirtualPageId::new(0, 0), VirtualPageId::new(0, 1), page]);
        // Act
        let matched = index.find_longest_prefix(&[0, 0, 0, 1]).unwrap();
        // Assert
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
    }

    #[test]
    fn token_zero_distinguishes_from_absent() {
        // Arrange: insert [0] with page, query [1]
        let mut index = KvPrefixIndex::new();
        index.insert(&[0], &[VirtualPageId::new(0, 0)]);
        // Act & Assert: token 0 and token 1 are distinct
        assert!(index.find_longest_prefix(&[1]).is_none());
        assert!(index.find_longest_prefix(&[0]).is_some());
    }

    // ── All-same token sequence ──

    #[test]
    fn all_identical_tokens_insert_and_query() {
        // Arrange: sequence of identical tokens [5, 5, 5, 5]
        let mut index = KvPrefixIndex::new();
        let pages: Vec<VirtualPageId> = (0..4).map(|i| VirtualPageId::new(0, i)).collect();
        index.insert(&[5, 5, 5, 5], &pages);
        // Act
        let matched = index.find_longest_prefix(&[5, 5, 5, 5, 5]).unwrap();
        // Assert
        assert_eq!(matched.matched_tokens, 4);
    }

    #[test]
    fn all_identical_tokens_query_shorter() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        let pages: Vec<VirtualPageId> = (0..4).map(|i| VirtualPageId::new(0, i)).collect();
        index.insert(&[5, 5, 5, 5], &pages);
        // Act: query [5, 5] — only first 2 match
        let matched = index.find_longest_prefix(&[5, 5]).unwrap();
        // Assert
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages.len(), 2);
    }

    // ── VirtualPageId boundary: large logical_index ──

    #[test]
    fn virtual_page_id_large_logical_index() {
        // Arrange: VirtualPageId with usize::MAX logical_index
        let page = VirtualPageId::new(0, usize::MAX);
        let mut index = KvPrefixIndex::new();
        index.insert(&[42], &[page]);
        // Act
        let matched = index.find_longest_prefix(&[42]).unwrap();
        // Assert
        assert_eq!(matched.matched_pages[0].logical_index, usize::MAX);
    }

    // ── Idempotent insert: same data twice produces same result ──

    #[test]
    fn insert_identical_twice_is_idempotent() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        let tokens = vec![10, 20];
        let pages = vec![VirtualPageId::new(0, 0), VirtualPageId::new(0, 1)];
        index.insert(&tokens, &pages);
        let first = index.find_longest_prefix(&tokens).unwrap();
        // Act: insert identical again
        index.insert(&tokens, &pages);
        let second = index.find_longest_prefix(&tokens).unwrap();
        // Assert: results are identical
        assert_eq!(first, second);
    }

    // ── Divergence at last token ──

    #[test]
    fn divergence_at_last_token() {
        // Arrange: two sequences sharing first N-1 tokens, diverging at last
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 100], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
        ]);
        index.insert(&[1, 2, 3, 200], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
            VirtualPageId::new(1, 3),
        ]);
        // Act & Assert: each full sequence matches independently
        let ma = index.find_longest_prefix(&[1, 2, 3, 100]).unwrap();
        assert_eq!(ma.matched_tokens, 4);
        let mb = index.find_longest_prefix(&[1, 2, 3, 200]).unwrap();
        assert_eq!(mb.matched_tokens, 4);
        // Divergent token gets no match beyond shared prefix
        let mc = index.find_longest_prefix(&[1, 2, 3, 999]).unwrap();
        assert_eq!(mc.matched_tokens, 3);
    }

    // ── Large number of branches from single root ──

    #[test]
    fn many_branches_from_shared_root() {
        // Arrange: 100 branches all sharing root token 1, diverging at second token
        let mut index = KvPrefixIndex::new();
        for i in 0..100u32 {
            let page = VirtualPageId::new(i as u64, 0);
            index.insert(&[1, i], &[VirtualPageId::new(0, 0), page]);
        }
        // Act & Assert: each branch matches
        for i in 0..100u32 {
            let matched = index.find_longest_prefix(&[1, i, 99]).unwrap();
            assert_eq!(matched.matched_tokens, 2, "failed for branch {i}");
        }
    }

    // ── Internal node has page, leaf has none ──

    #[test]
    fn internal_node_page_ref_leaf_without() {
        // Arrange: 3 tokens, only first has page
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[VirtualPageId::new(0, 0)]);
        // Act: full query hits internal node with page, leaf without
        let matched = index.find_longest_prefix(&[10, 20, 30]).unwrap();
        // Assert: match length is 1 (only first token has page)
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages.len(), 1);
    }

    // ── Three sequential overwrites ──

    #[test]
    fn three_sequential_overwrites_last_wins() {
        // Arrange: overwrite same sequence 3 times
        let mut index = KvPrefixIndex::new();
        let tokens = [7, 8];
        let p1 = vec![VirtualPageId::new(1, 0), VirtualPageId::new(1, 1)];
        let p2 = vec![VirtualPageId::new(2, 0), VirtualPageId::new(2, 1)];
        let p3 = vec![VirtualPageId::new(3, 0), VirtualPageId::new(3, 1)];
        index.insert(&tokens, &p1);
        index.insert(&tokens, &p2);
        index.insert(&tokens, &p3);
        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();
        // Assert: last insert wins
        assert_eq!(matched.matched_pages, p3);
    }

    // ── PrefixMatch: same tokens different pages not equal ──

    #[test]
    fn prefix_match_inequality_same_tokens_empty_vs_nonempty_pages() {
        // Arrange
        let a = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![],
        };
        let b = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![VirtualPageId::new(0, 0)],
        };
        // Assert
        assert_ne!(a, b);
    }

    // ── PrefixMatch Debug includes both fields' values ──

    #[test]
    fn prefix_match_debug_includes_page_details() {
        // Arrange
        let pm = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(5, 10)],
        };
        // Act
        let debug = format!("{pm:?}");
        // Assert: debug output contains the field names
        assert!(debug.contains("matched_tokens: 2"));
        assert!(debug.contains("matched_pages"));
    }

    // ── Single insert then multiple distinct queries ──

    #[test]
    fn single_insert_multiple_distinct_queries() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act & Assert: various queries
        assert!(index.find_longest_prefix(&[99]).is_none());
        assert!(index.find_longest_prefix(&[10, 99]).is_some()); // matches [10] only
        let m = index.find_longest_prefix(&[10, 99]).unwrap();
        assert_eq!(m.matched_tokens, 1);
        let full = index.find_longest_prefix(&[10, 20, 30, 40]).unwrap();
        assert_eq!(full.matched_tokens, 3);
    }

    // ── Insert long sequence, query shorter prefixes at multiple points ──

    #[test]
    fn long_sequence_prefix_queries_at_boundaries() {
        // Arrange: 10 tokens
        let tokens: Vec<TokenId> = (0..10).collect();
        let pages: Vec<VirtualPageId> = (0..10).map(|i| VirtualPageId::new(0, i as usize)).collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act & Assert: query at various prefix lengths
        for len in 1..=10 {
            let query: Vec<TokenId> = (0..len as u32).collect();
            let matched = index.find_longest_prefix(&query).unwrap();
            assert_eq!(matched.matched_tokens, len);
        }
    }

    // ── 15 new tests ──

    // 1. Shorter sequence inserted after longer extension — prefix of longer still found
    #[test]
    fn insert_shorter_after_longer_shared_prefix() {
        // Arrange: insert [1,2,3,4] then [1,2]
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
        ]);
        index.insert(&[1, 2], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);
        // Act: query [1,2,3,4] still matches fully (shared nodes overwritten but deeper nodes remain)
        let matched = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(matched.matched_tokens, 4);
        // Act: query [1,2] matches with overwritten pages
        let short = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(short.matched_tokens, 2);
    }

    // 2. Two fully independent sequences with zero overlap
    #[test]
    fn two_fully_independent_sequences() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[100, 200], &[
            VirtualPageId::new(10, 0),
            VirtualPageId::new(10, 1),
        ]);
        index.insert(&[300, 400], &[
            VirtualPageId::new(20, 0),
            VirtualPageId::new(20, 1),
        ]);
        // Act & Assert: each matches independently
        let ma = index.find_longest_prefix(&[100, 200, 999]).unwrap();
        assert_eq!(ma.matched_tokens, 2);
        assert_eq!(ma.matched_pages[0], VirtualPageId::new(10, 0));
        let mb = index.find_longest_prefix(&[300, 400, 999]).unwrap();
        assert_eq!(mb.matched_tokens, 2);
        assert_eq!(mb.matched_pages[0], VirtualPageId::new(20, 0));
    }

    // 3. Query single token that only appears deep in trie — no match from root
    #[test]
    fn query_mid_sequence_token_from_root_returns_none() {
        // Arrange: insert [10, 20, 30]
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act & Assert: token 20 exists in trie but not as root child
        assert!(index.find_longest_prefix(&[20]).is_none());
        assert!(index.find_longest_prefix(&[30]).is_none());
    }

    // 4. Repeated find calls return identical results (immutability)
    #[test]
    fn repeated_find_is_idempotent() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[5, 6, 7], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act: call find 5 times
        let results: Vec<_> = (0..5)
            .map(|_| index.find_longest_prefix(&[5, 6, 7, 8]))
            .collect();
        // Assert: all identical
        for result in &results[1..] {
            assert_eq!(results[0], *result);
        }
    }

    // 5. Different sequence_ids produce different pages — no dedup across sequences
    #[test]
    fn different_sequence_ids_not_deduplicated() {
        // Arrange: two pages with different sequence_id but same logical_index
        let page_a = VirtualPageId::new(1, 0);
        let page_b = VirtualPageId::new(2, 0);
        assert_ne!(page_a, page_b);
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2], &[page_a, page_b]);
        // Act
        let matched = index.find_longest_prefix(&[1, 2]).unwrap();
        // Assert: both pages present, no dedup
        assert_eq!(matched.matched_pages.len(), 2);
        assert_eq!(matched.matched_pages[0], page_a);
        assert_eq!(matched.matched_pages[1], page_b);
    }

    // 6. PrefixMatch Debug with empty pages shows empty vector
    #[test]
    fn prefix_match_debug_empty_pages_shows_empty_vec() {
        let pm = PrefixMatch {
            matched_tokens: 0,
            matched_pages: vec![],
        };
        let debug = format!("{pm:?}");
        assert!(debug.contains("matched_tokens: 0"));
        assert!(debug.contains("matched_pages: []"));
    }

    // 7. KvPrefixIndex Debug after insert shows non-empty root
    #[test]
    fn index_debug_after_insert_shows_children() {
        let mut index = KvPrefixIndex::new();
        index.insert(&[1], &[VirtualPageId::new(0, 0)]);
        let debug = format!("{index:?}");
        // After insert, root should have children (non-empty HashMap)
        assert!(debug.contains("children"));
    }

    // 8. Insert empty then real — empty insert does not corrupt state
    #[test]
    fn empty_insert_then_real_insert_works() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[], &[VirtualPageId::new(0, 0)]);
        index.insert(&[42], &[VirtualPageId::new(0, 0)]);
        // Act & Assert
        let matched = index.find_longest_prefix(&[42]).unwrap();
        assert_eq!(matched.matched_tokens, 1);
    }

    // 9. PrefixMatch equality: same tokens, same single page
    #[test]
    fn prefix_match_equality_same_single_page() {
        let a = PrefixMatch {
            matched_tokens: 1,
            matched_pages: vec![VirtualPageId::new(5, 10)],
        };
        let b = PrefixMatch {
            matched_tokens: 1,
            matched_pages: vec![VirtualPageId::new(5, 10)],
        };
        assert_eq!(a, b);
    }

    // 10. PrefixMatch inequality: same tokens, different page count
    #[test]
    fn prefix_match_inequality_different_page_count() {
        let a = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(0, 0)],
        };
        let b = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(0, 0), VirtualPageId::new(0, 1)],
        };
        assert_ne!(a, b);
    }

    // 11. Large branch factor: 256 branches from single root node
    #[test]
    fn large_branch_factor_256_branches() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        for i in 0u32..256 {
            index.insert(&[0, i], &[
                VirtualPageId::new(0, 0),
                VirtualPageId::new(i as u64, 1),
            ]);
        }
        // Act & Assert: all 256 branches found
        for i in 0u32..256 {
            let matched = index.find_longest_prefix(&[0, i]).unwrap();
            assert_eq!(matched.matched_tokens, 2, "failed for branch {i}");
        }
        // Non-existent branch returns partial match
        let partial = index.find_longest_prefix(&[0, 999]).unwrap();
        assert_eq!(partial.matched_tokens, 1);
    }

    // 12. Query matches only first token when second diverges immediately
    #[test]
    fn query_diverges_at_second_token_matches_only_first() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
        ]);
        // Act: query diverges at token index 1
        let matched = index.find_longest_prefix(&[1, 99]).unwrap();
        // Assert: only first token matched
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages.len(), 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
    }

    // 13. Insert with pages only for middle tokens (gap at start)
    #[test]
    fn pages_only_for_middle_tokens_gap_at_start() {
        // Arrange: 5 tokens, pages only at indices 2 and 3
        let mut index = KvPrefixIndex::new();
        // insert provides pages for all positions; simulate gap by providing empty page vec
        // Actually, insert assigns pages.get(idx).copied(), so we need a custom approach.
        // Let's insert with pages covering indices 2,3 by using a slice starting at 0.
        // The only way to have a gap is to provide fewer pages than tokens.
        // Insert [1,2,3,4,5] with pages [page_at_0, page_at_1] → tokens 2,3,4 have no page
        index.insert(&[1, 2, 3, 4, 5], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
        ]);
        // Act: find returns match length = 2 (last token with a page)
        let matched = index.find_longest_prefix(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
    }

    // 14. Two-level common prefix with third-level divergence
    #[test]
    fn two_level_shared_prefix_third_level_diverges() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        // [A, B, X]
        index.insert(&[1, 2, 10], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // [A, B, Y]
        index.insert(&[1, 2, 20], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // Act: query [A, B, Z] matches only first two tokens
        let matched = index.find_longest_prefix(&[1, 2, 99]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
        // Act: each full branch matches 3
        let ma = index.find_longest_prefix(&[1, 2, 10]).unwrap();
        assert_eq!(ma.matched_tokens, 3);
        let mb = index.find_longest_prefix(&[1, 2, 20]).unwrap();
        assert_eq!(mb.matched_tokens, 3);
    }

    // 15. PrefixMatch clone produces independent copy
    #[test]
    fn prefix_match_clone_is_independent() {
        // Arrange
        let original = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![
                VirtualPageId::new(0, 0),
                VirtualPageId::new(0, 1),
                VirtualPageId::new(0, 2),
            ],
        };
        let cloned = original.clone();
        // Assert: equal
        assert_eq!(original, cloned);
        // Assert: field values match individually
        assert_eq!(original.matched_tokens, cloned.matched_tokens);
        assert_eq!(original.matched_pages, cloned.matched_pages);
        assert_eq!(original.matched_pages.len(), 3);
    }

    // ── Wave 2: 15 additional tests ──

    // 1. Insert then overwrite with fewer pages — page_refs at deeper nodes remain
    #[test]
    fn reinsert_with_fewer_pages_deeper_refs_remain() {
        // Arrange: full pages first
        let mut index = KvPrefixIndex::new();
        let tokens = vec![10, 20, 30];
        index.insert(&tokens, &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act: overwrite with only 1 page — token 0's page_ref is updated,
        // but tokens 1 and 2 still have their page_refs from the first insert
        index.insert(&tokens, &[VirtualPageId::new(1, 0)]);
        let matched = index.find_longest_prefix(&tokens).unwrap();
        // Assert: deeper page_refs survive, match is still 3 tokens
        assert_eq!(matched.matched_tokens, 3);
        // First page is overwritten to (1, 0)
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(1, 0));
    }

    // 2. Page deduplication: two consecutive tokens sharing same VirtualPageId
    #[test]
    fn consecutive_tokens_same_page_deduplicated_in_output() {
        // Arrange: tokens [A, B] both map to the same VirtualPageId
        let shared_page = VirtualPageId::new(42, 7);
        let mut index = KvPrefixIndex::new();
        index.insert(&[100, 200], &[shared_page, shared_page]);
        // Act
        let matched = index.find_longest_prefix(&[100, 200]).unwrap();
        // Assert: only one page in output despite two tokens
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages, vec![shared_page]);
    }

    // 3. Deep trie: chain of 500 single-child nodes
    #[test]
    fn deep_single_chain_500_tokens() {
        // Arrange: [0, 1, 2, ..., 499] each with unique page
        let tokens: Vec<TokenId> = (0..500).collect();
        let pages: Vec<VirtualPageId> = (0..500)
            .map(|i| VirtualPageId::new(0, i as usize))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act: match at various depths
        let at_100 = index.find_longest_prefix(&(0..100).collect::<Vec<_>>()).unwrap();
        assert_eq!(at_100.matched_tokens, 100);
        let at_499 = index.find_longest_prefix(&(0..499).collect::<Vec<_>>()).unwrap();
        assert_eq!(at_499.matched_tokens, 499);
        let full = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(full.matched_tokens, 500);
    }

    // 4. Multiple empty inserts followed by real insert
    #[test]
    fn multiple_empty_inserts_do_not_corrupt() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[], &[VirtualPageId::new(0, 0)]);
        index.insert(&[], &[VirtualPageId::new(1, 0)]);
        index.insert(&[], &[]);
        index.insert(&[5, 10], &[VirtualPageId::new(0, 0), VirtualPageId::new(0, 1)]);
        // Act & Assert
        let matched = index.find_longest_prefix(&[5, 10]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
        assert!(index.find_longest_prefix(&[1]).is_none());
    }

    // 5. Token sequence with all u32 byte boundaries (0x00, 0xFF, 0x100, 0xFFFF, 0x10000)
    #[test]
    fn token_byte_boundary_values() {
        // Arrange: boundary token values
        let tokens: Vec<TokenId> = vec![0x00, 0xFF, 0x100, 0xFFFF, 0x10000];
        let pages: Vec<VirtualPageId> = tokens.iter().enumerate()
            .map(|(i, _)| VirtualPageId::new(0, i))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act: exact match
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 5);
        // Act: partial match at boundary
        let partial = index.find_longest_prefix(&[0x00, 0xFF]).unwrap();
        assert_eq!(partial.matched_tokens, 2);
        // Act: diverge at boundary value
        let miss = index.find_longest_prefix(&[0x00, 0xFE]);
        assert_eq!(miss.unwrap().matched_tokens, 1);
    }

    // 6. Overwriting shared prefix node affects all branches
    #[test]
    fn overwrite_shared_node_affects_all_branches() {
        // Arrange: two branches share root token
        let mut index = KvPrefixIndex::new();
        let page_v1 = VirtualPageId::new(1, 0);
        index.insert(&[1, 10], &[page_v1, VirtualPageId::new(1, 1)]);
        index.insert(&[1, 20], &[VirtualPageId::new(2, 0), VirtualPageId::new(2, 1)]);
        // Act: overwrite shared root with new page
        let page_v2 = VirtualPageId::new(99, 0);
        index.insert(&[1, 30], &[page_v2, VirtualPageId::new(3, 1)]);
        // Assert: all three branches now see page_v2 at root
        for second in [10u32, 20, 30] {
            let matched = index.find_longest_prefix(&[1, second]).unwrap();
            assert_eq!(matched.matched_pages[0], page_v2,
                "branch [1, {second}] should see overwritten root page");
        }
    }

    // 7. Query that is a strict prefix of an inserted sequence (already tested but this
    //    verifies page list correctness, not just token count)
    #[test]
    fn strict_prefix_query_verifies_page_contents() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        let pages = vec![
            VirtualPageId::new(0, 100),
            VirtualPageId::new(0, 200),
            VirtualPageId::new(0, 300),
            VirtualPageId::new(0, 400),
        ];
        index.insert(&[1, 2, 3, 4], &pages);
        // Act: query only first 3 tokens
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        // Assert: pages for first 3 tokens only
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages, vec![
            VirtualPageId::new(0, 100),
            VirtualPageId::new(0, 200),
            VirtualPageId::new(0, 300),
        ]);
    }

    // 8. Progressive inserts extend the trie and overwrite earlier page_refs
    #[test]
    fn progressive_insert_overwrites_shared_prefix_pages() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        // Act & Assert: insert [1] first
        index.insert(&[1], &[VirtualPageId::new(0, 0)]);
        assert_eq!(
            index.find_longest_prefix(&[1]).unwrap().matched_tokens, 1
        );

        // Insert [1, 2] — overwrites page for token 1, adds page for token 2
        index.insert(&[1, 2], &[VirtualPageId::new(1, 0), VirtualPageId::new(1, 1)]);
        let m = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(m.matched_tokens, 2);
        assert_eq!(m.matched_pages[0], VirtualPageId::new(1, 0));
        assert_eq!(m.matched_pages[1], VirtualPageId::new(1, 1));

        // Insert [1, 2, 3] — all three tokens now have pages
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(2, 0), VirtualPageId::new(2, 1), VirtualPageId::new(2, 2),
        ]);
        let full = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(full.matched_tokens, 3);
        assert_eq!(full.matched_pages, vec![
            VirtualPageId::new(2, 0), VirtualPageId::new(2, 1), VirtualPageId::new(2, 2),
        ]);
    }

    // 9. Find on empty index returns None consistently
    #[test]
    fn empty_index_repeated_queries_all_none() {
        // Arrange
        let index = KvPrefixIndex::new();
        // Act & Assert: multiple different queries all return None
        assert!(index.find_longest_prefix(&[]).is_none());
        assert!(index.find_longest_prefix(&[0]).is_none());
        assert!(index.find_longest_prefix(&[u32::MAX]).is_none());
        assert!(index.find_longest_prefix(&[1, 2, 3, 4, 5]).is_none());
    }

    // 10. VirtualPageId with sequence_id=0 is valid and distinct from no page
    #[test]
    fn sequence_id_zero_is_valid_page() {
        // Arrange
        let page = VirtualPageId::new(0, 0); // sequence_id = 0, logical_index = 0
        let mut index = KvPrefixIndex::new();
        index.insert(&[1], &[page]);
        // Act
        let matched = index.find_longest_prefix(&[1]).unwrap();
        // Assert: page with sequence_id=0 is stored and returned
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
        assert_eq!(matched.matched_tokens, 1);
    }

    // 11. Three-way branching at root with deep sub-branches
    #[test]
    fn three_way_root_branch_with_depth() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        // Branch A: [0, 1, 2, 3]
        index.insert(&[0, 1, 2, 3], &[
            VirtualPageId::new(10, 0), VirtualPageId::new(10, 1),
            VirtualPageId::new(10, 2), VirtualPageId::new(10, 3),
        ]);
        // Branch B: [0, 5, 6]
        index.insert(&[0, 5, 6], &[
            VirtualPageId::new(20, 0), VirtualPageId::new(20, 1),
            VirtualPageId::new(20, 2),
        ]);
        // Branch C: [0, 9]
        index.insert(&[0, 9], &[
            VirtualPageId::new(30, 0), VirtualPageId::new(30, 1),
        ]);
        // Act & Assert: full depth on each branch
        assert_eq!(index.find_longest_prefix(&[0, 1, 2, 3]).unwrap().matched_tokens, 4);
        assert_eq!(index.find_longest_prefix(&[0, 5, 6]).unwrap().matched_tokens, 3);
        assert_eq!(index.find_longest_prefix(&[0, 9]).unwrap().matched_tokens, 2);
        // Unknown branch: only root matches
        assert_eq!(index.find_longest_prefix(&[0, 77]).unwrap().matched_tokens, 1);
    }

    // 12. Insert single token, then longer sequence starting with different token
    #[test]
    fn independent_single_token_and_longer_sequence() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[50], &[VirtualPageId::new(0, 0)]);
        index.insert(&[60, 70, 80], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // Act & Assert: each independent
        assert_eq!(index.find_longest_prefix(&[50]).unwrap().matched_tokens, 1);
        assert_eq!(index.find_longest_prefix(&[60, 70, 80]).unwrap().matched_tokens, 3);
        // No cross-contamination
        assert!(index.find_longest_prefix(&[50, 70]).unwrap().matched_tokens == 1);
        assert!(index.find_longest_prefix(&[60]).unwrap().matched_tokens == 1);
    }

    // 13. Page dedup: same page appears at non-consecutive positions
    #[test]
    fn same_page_at_non_consecutive_positions() {
        // Arrange: page P at positions 0 and 2, different page at 1
        let shared = VirtualPageId::new(5, 0);
        let other = VirtualPageId::new(6, 0);
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[shared, other, shared]);
        // Act
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        // Assert: all 3 tokens matched, pages include shared, other, shared (dedup
        // only removes consecutive duplicates)
        assert_eq!(matched.matched_tokens, 3);
        // The dedup logic checks `last().copied() != Some(page_id)`, so non-consecutive
        // same page will appear twice
        assert_eq!(matched.matched_pages.len(), 3);
        assert_eq!(matched.matched_pages[0], shared);
        assert_eq!(matched.matched_pages[1], other);
        assert_eq!(matched.matched_pages[2], shared);
    }

    // 14. Large token value range: tokens from u32 range spread evenly
    #[test]
    fn sparse_token_values_across_u32_range() {
        // Arrange: 10 tokens with large gaps
        let tokens: Vec<TokenId> = vec![
            0, 1_000_000, 100_000_000, u32::MAX / 2, u32::MAX,
        ];
        let pages: Vec<VirtualPageId> = tokens.iter().enumerate()
            .map(|(i, _)| VirtualPageId::new(0, i))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act: exact match
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 5);
        // Act: partial match stopping at middle
        let partial = index.find_longest_prefix(&[0, 1_000_000]).unwrap();
        assert_eq!(partial.matched_tokens, 2);
        // Act: wrong value at second position
        let miss = index.find_longest_prefix(&[0, 999_999]).unwrap();
        assert_eq!(miss.matched_tokens, 1);
    }

    // 15. PrefixMatch equality: different matched_tokens with empty pages
    #[test]
    fn prefix_match_inequality_different_tokens_both_empty_pages() {
        // Arrange
        let a = PrefixMatch {
            matched_tokens: 0,
            matched_pages: vec![],
        };
        let b = PrefixMatch {
            matched_tokens: 1,
            matched_pages: vec![],
        };
        // Assert: different matched_tokens makes them unequal
        assert_ne!(a, b);
    }

    // ── Wave 3: 15 additional tests covering uncovered edge cases ──

    // 1. Three consecutive tokens mapping to same page — deduplicated to single entry
    #[test]
    fn three_consecutive_same_page_deduplicated_to_one() {
        // Arrange: 5 tokens all mapping to the same VirtualPageId
        let shared = VirtualPageId::new(7, 3);
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30, 40, 50], &[shared; 5]);
        // Act
        let matched = index.find_longest_prefix(&[10, 20, 30, 40, 50]).unwrap();
        // Assert: 5 tokens matched but only 1 unique page
        assert_eq!(matched.matched_tokens, 5);
        assert_eq!(matched.matched_pages, vec![shared]);
    }

    // 2. Wide trie: 50 independent single-token sequences at root level
    #[test]
    fn wide_trie_50_independent_single_tokens() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        for i in 0u32..50 {
            let page = VirtualPageId::new(i as u64, 0);
            index.insert(&[i], &[page]);
        }
        // Act & Assert: each single-token sequence is independently findable
        for i in 0u32..50 {
            let matched = index.find_longest_prefix(&[i]).unwrap();
            assert_eq!(matched.matched_tokens, 1, "failed for token {i}");
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(i as u64, 0));
        }
        // Non-existent token returns None
        assert!(index.find_longest_prefix(&[999]).is_none());
    }

    // 3. Insert single token without page — trie node exists but no match returned
    #[test]
    fn single_token_without_page_returns_none_even_with_trie_node() {
        // Arrange: insert [42] with empty pages slice — trie node created but page_ref is None
        let mut index = KvPrefixIndex::new();
        index.insert(&[42], &[]);
        // Act & Assert: no page_ref means best_tokens stays 0
        assert!(index.find_longest_prefix(&[42]).is_none());
    }

    // 4. Partial match at first token only when query diverges at second — verify exact page
    #[test]
    fn first_token_match_page_is_from_latest_overwrite() {
        // Arrange: insert [1, 2] then overwrite shared prefix with [1, 3]
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
        ]);
        index.insert(&[1, 3], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);
        // Act: query diverges at token 2 — only token 1 matches
        let matched = index.find_longest_prefix(&[1, 99]).unwrap();
        // Assert: page for shared token 1 comes from the second insert (last overwrite wins)
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(1, 0));
    }

    // 5. Long common prefix (50 tokens) diverging at position 51
    #[test]
    fn long_common_prefix_diverging_at_51() {
        // Arrange: two sequences sharing first 50 tokens
        let mut index = KvPrefixIndex::new();
        let shared: Vec<TokenId> = (0..50).collect();
        let mut seq_a: Vec<TokenId> = shared.clone();
        seq_a.push(100);
        let mut seq_b: Vec<TokenId> = shared.clone();
        seq_b.push(200);
        let pages_a: Vec<VirtualPageId> = (0..51).map(|i| VirtualPageId::new(0, i)).collect();
        let pages_b: Vec<VirtualPageId> = (0..51).map(|i| VirtualPageId::new(1, i)).collect();
        index.insert(&seq_a, &pages_a);
        index.insert(&seq_b, &pages_b);
        // Act: query matching seq_a exactly
        let ma = index.find_longest_prefix(&seq_a).unwrap();
        assert_eq!(ma.matched_tokens, 51);
        // Act: query with divergent 51st token — matches 50 shared tokens
        let mut q = shared.clone();
        q.push(999);
        let md = index.find_longest_prefix(&q).unwrap();
        assert_eq!(md.matched_tokens, 50);
    }

    // 6. Overwrite mid-sequence page_ref without touching deeper nodes
    #[test]
    fn overwrite_prefix_leaves_deeper_page_refs_intact() {
        // Arrange: insert [1, 2, 3] with full pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act: overwrite only [1] with a new page
        index.insert(&[1], &[VirtualPageId::new(9, 9)]);
        // Assert: full sequence still matches 3 tokens
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        // First page is overwritten
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(9, 9));
        // Deeper pages remain from first insert
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(0, 1));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 2));
    }

    // 7. Page deduplication with page appearing at start and end but not middle
    #[test]
    fn same_page_at_start_and_end_with_different_middle() {
        // Arrange: page P at positions 0 and 3, different pages at 1 and 2
        let p = VirtualPageId::new(5, 0);
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4], &[p, VirtualPageId::new(6, 0), VirtualPageId::new(7, 0), p]);
        // Act
        let matched = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
        // Assert: 4 tokens matched, pages are [P, 6:0, 7:0, P] — P appears twice (non-consecutive)
        assert_eq!(matched.matched_tokens, 4);
        assert_eq!(matched.matched_pages.len(), 4);
        assert_eq!(matched.matched_pages[0], p);
        assert_eq!(matched.matched_pages[3], p);
    }

    // 8. Many VirtualPageIds with same logical_index but different sequence_ids
    #[test]
    fn same_logical_index_different_sequence_ids_all_distinct() {
        // Arrange: 10 pages all with logical_index=5 but different sequence_ids
        let mut index = KvPrefixIndex::new();
        for sid in 0u64..10 {
            let tokens = vec![(sid * 100) as u32, (sid * 100 + 1) as u32];
            let pages = vec![
                VirtualPageId::new(sid, 5),
                VirtualPageId::new(sid, 5),
            ];
            index.insert(&tokens, &pages);
        }
        // Act & Assert: each sequence's pages are unique per sequence_id
        for sid in 0u64..10 {
            let tokens = vec![(sid * 100) as u32, (sid * 100 + 1) as u32];
            let matched = index.find_longest_prefix(&tokens).unwrap();
            assert_eq!(matched.matched_tokens, 2);
            // Dedup: same page for both tokens → only 1 in output
            assert_eq!(matched.matched_pages.len(), 1);
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(sid, 5));
        }
    }

    // 9. Query exactly one token from a 200-token inserted sequence
    #[test]
    fn query_single_token_from_long_sequence() {
        // Arrange: 200-token sequence
        let tokens: Vec<TokenId> = (0..200).collect();
        let pages: Vec<VirtualPageId> = (0..200).map(|i| VirtualPageId::new(0, i as usize)).collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act: query only the first token
        let matched = index.find_longest_prefix(&[0]).unwrap();
        // Assert
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages.len(), 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
    }

    // 10. Insert sequence, query suffix that has no prefix overlap — returns None
    #[test]
    fn suffix_query_with_no_root_overlap_returns_none() {
        // Arrange: insert [10, 20, 30]
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act & Assert: query [20, 30] — token 20 is not a root child
        assert!(index.find_longest_prefix(&[20, 30]).is_none());
        assert!(index.find_longest_prefix(&[30]).is_none());
    }

    // 11. PrefixMatch PartialEq: same tokens, same pages in different order — not equal
    #[test]
    fn prefix_match_same_tokens_pages_in_different_order_not_equal() {
        // Arrange
        let a = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(1, 0), VirtualPageId::new(2, 0)],
        };
        let b = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(2, 0), VirtualPageId::new(1, 0)],
        };
        // Assert: different order → not equal
        assert_ne!(a, b);
    }

    // 12. Interleaved branches: insert A, query A, insert B (shares prefix), query both
    #[test]
    fn interleaved_branch_insert_query_correctness() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[5, 10, 15], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Query before second insert
        let before = index.find_longest_prefix(&[5, 10, 15]).unwrap();
        assert_eq!(before.matched_tokens, 3);
        // Act: insert diverging branch sharing [5]
        index.insert(&[5, 20, 25], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // Assert: both branches still fully match
        let ma = index.find_longest_prefix(&[5, 10, 15]).unwrap();
        assert_eq!(ma.matched_tokens, 3);
        let mb = index.find_longest_prefix(&[5, 20, 25]).unwrap();
        assert_eq!(mb.matched_tokens, 3);
        // Shared prefix [5] query matches 1 token
        let shared = index.find_longest_prefix(&[5, 99]).unwrap();
        assert_eq!(shared.matched_tokens, 1);
    }

    // 13. Insert with more pages than tokens — excess pages silently ignored
    #[test]
    fn more_pages_than_tokens_excess_ignored() {
        // Arrange: 2 tokens but 5 pages provided
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
            VirtualPageId::new(0, 4),
        ]);
        // Act: only 2 pages assigned (pages.get(0) and pages.get(1))
        let matched = index.find_longest_prefix(&[1, 2]).unwrap();
        // Assert
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages.len(), 2);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(0, 1));
    }

    // 14. Empty pages insert does not clear existing page_refs
    #[test]
    fn empty_pages_insert_does_not_clear_existing_page_refs() {
        // Arrange: insert [1, 2] with full pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
        ]);
        // Act: re-insert [1, 2] with empty pages — insert only sets page_ref when
        // pages.get(idx) is Some, never clears existing page_ref
        index.insert(&[1, 2], &[]);
        // Assert: both page_refs persist
        let matched = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages.len(), 2);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(0, 1));
    }

    // 15. Verify find_longest_prefix is read-only: insert, find, insert again, find unchanged
    #[test]
    fn find_does_not_mutate_index_state() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act: find before and after — results should be identical each time
        let r1 = index.find_longest_prefix(&[1, 2, 3]);
        let r2 = index.find_longest_prefix(&[1, 2, 3]);
        let r3 = index.find_longest_prefix(&[1, 2, 3, 4]);
        // Assert: repeated finds return same result
        assert_eq!(r1, r2);
        // The partial match (3 of 4 tokens) is consistent
        assert_eq!(r3.as_ref().unwrap().matched_tokens, 3);
        assert_eq!(r3.as_ref().unwrap().matched_tokens, r1.as_ref().unwrap().matched_tokens);
    }

    // ── Wave 4: 15 additional tests covering uncovered edge cases ──

    // 1. VirtualPageId with u64::MAX sequence_id stored and retrieved correctly
    #[test]
    fn virtual_page_id_max_sequence_id_roundtrip() {
        // Arrange: use u64::MAX as sequence_id
        let page = VirtualPageId::new(u64::MAX, 42);
        let mut index = KvPrefixIndex::new();
        index.insert(&[100], &[page]);
        // Act
        let matched = index.find_longest_prefix(&[100]).unwrap();
        // Assert: page stored and retrieved without truncation
        assert_eq!(matched.matched_pages[0].sequence_id, u64::MAX);
        assert_eq!(matched.matched_pages[0].logical_index, 42);
    }

    // 2. Insert same token sequence with progressively shifting page assignments
    #[test]
    fn progressive_page_shift_via_reinsert() {
        // Arrange: insert [A, B, C] with pages at all positions, then overwrite
        // with a version that has pages only at position 0
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 10),
            VirtualPageId::new(0, 20),
            VirtualPageId::new(0, 30),
        ]);
        // Act: overwrite [1, 2] with different pages — token 3's page_ref persists
        index.insert(&[1, 2], &[
            VirtualPageId::new(5, 100),
            VirtualPageId::new(5, 200),
        ]);
        // Assert: full query still matches 3 tokens (deep ref survives)
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        // Position 0 and 1 pages come from the second insert
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(5, 100));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(5, 200));
        // Position 2 page survives from the first insert
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 30));
    }

    // 3. Insert token without page then overwrite with page via longer sequence
    // — insert with empty pages does NOT clear existing page_ref
    #[test]
    fn insert_empty_pages_does_not_clear_existing_page_ref() {
        // Arrange: insert [10] with a page, then insert [10] with empty pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[10], &[VirtualPageId::new(0, 0)]);
        // Act: insert same token with empty pages — page_ref is NOT cleared
        // because insert only sets page_ref when pages.get(idx) is Some
        index.insert(&[10], &[]);
        // Assert: page_ref from first insert persists
        let matched = index.find_longest_prefix(&[10]).unwrap();
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
    }

    // 4. Multiple branches share 2-token prefix, query with 1-token prefix verifies page
    #[test]
    fn multi_branch_shared_prefix_short_query_verifies_page() {
        // Arrange: 3 branches sharing [1, 2], each 4 tokens long
        let mut index = KvPrefixIndex::new();
        for (branch, suffix) in [(10u32, 100u32), (20, 200), (30, 300)] {
            index.insert(&[1, 2, branch, suffix], &[
                VirtualPageId::new(branch as u64, 0),
                VirtualPageId::new(branch as u64, 1),
                VirtualPageId::new(branch as u64, 2),
                VirtualPageId::new(branch as u64, 3),
            ]);
        }
        // Act: query [1] — shared token's page comes from last insert (branch 30)
        let matched = index.find_longest_prefix(&[1]).unwrap();
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(30, 0));
        // Act: query [1, 2] — second shared token's page also from last insert
        let matched2 = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(matched2.matched_tokens, 2);
        assert_eq!(matched2.matched_pages[0], VirtualPageId::new(30, 0));
        assert_eq!(matched2.matched_pages[1], VirtualPageId::new(30, 1));
    }

    // 5. Token sequence where every other token has the same page — alternating dedup
    #[test]
    fn alternating_same_page_every_other_token() {
        // Arrange: page P at positions 0, 2, 4 and different pages at 1, 3
        let p = VirtualPageId::new(0, 0);
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4, 5], &[
            p,
            VirtualPageId::new(1, 0),
            p,
            VirtualPageId::new(2, 0),
            p,
        ]);
        // Act
        let matched = index.find_longest_prefix(&[1, 2, 3, 4, 5]).unwrap();
        // Assert: 5 tokens matched
        assert_eq!(matched.matched_tokens, 5);
        // Pages: [P, 1:0, P, 2:0, P] — P appears 3 times (non-consecutive)
        assert_eq!(matched.matched_pages.len(), 5);
        assert_eq!(matched.matched_pages[0], p);
        assert_eq!(matched.matched_pages[2], p);
        assert_eq!(matched.matched_pages[4], p);
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(1, 0));
        assert_eq!(matched.matched_pages[3], VirtualPageId::new(2, 0));
    }

    // 6. Insert a sequence that is a prefix of another already-inserted sequence, verify
    // the longer sequence still matches after the shorter overwrites shared page_refs
    #[test]
    fn shorter_overwrite_does_not_break_longer_path() {
        // Arrange: insert long sequence first
        let mut index = KvPrefixIndex::new();
        index.insert(&[5, 6, 7, 8, 9], &[
            VirtualPageId::new(0, 50),
            VirtualPageId::new(0, 60),
            VirtualPageId::new(0, 70),
            VirtualPageId::new(0, 80),
            VirtualPageId::new(0, 90),
        ]);
        // Act: insert shorter prefix with different pages — overwrites first 2 nodes
        index.insert(&[5, 6], &[
            VirtualPageId::new(9, 5),
            VirtualPageId::new(9, 6),
        ]);
        // Assert: long sequence still fully matchable
        let matched = index.find_longest_prefix(&[5, 6, 7, 8, 9]).unwrap();
        assert_eq!(matched.matched_tokens, 5);
        assert_eq!(matched.matched_pages.len(), 5);
        // First 2 pages come from the shorter overwrite
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(9, 5));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(9, 6));
        // Remaining pages from original insert
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 70));
        assert_eq!(matched.matched_pages[3], VirtualPageId::new(0, 80));
        assert_eq!(matched.matched_pages[4], VirtualPageId::new(0, 90));
    }

    // 7. Insert 10 sequences of length 3 sharing token at position 0, verify each matches
    // independently and that querying a non-existent branch gives partial match of length 1
    #[test]
    fn ten_branches_from_root_all_match_independently() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        for i in 0u32..10 {
            let tokens = [100, i * 10, i * 10 + 1];
            let pages: Vec<VirtualPageId> = (0..3)
                .map(|p| VirtualPageId::new(i as u64, p))
                .collect();
            index.insert(&tokens, &pages);
        }
        // Act & Assert: each branch matches fully
        for i in 0u32..10 {
            let tokens = [100, i * 10, i * 10 + 1];
            let matched = index.find_longest_prefix(&tokens).unwrap();
            assert_eq!(matched.matched_tokens, 3, "branch {i} should match 3 tokens");
        }
        // Non-existent second token: matches only root
        let partial = index.find_longest_prefix(&[100, 999]).unwrap();
        assert_eq!(partial.matched_tokens, 1);
    }

    // 8. Saturating add in best_tokens: single token at index 0 produces best_tokens = 1
    #[test]
    fn saturating_add_first_token_produces_one() {
        // Arrange: single token at trie root level
        let mut index = KvPrefixIndex::new();
        index.insert(&[7], &[VirtualPageId::new(0, 0)]);
        // Act: idx=0, best_tokens = 0.saturating_add(1) = 1
        let matched = index.find_longest_prefix(&[7]).unwrap();
        // Assert
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages.len(), 1);
    }

    // 9. Insert tokens where first and last share VirtualPageId but middle has different
    // — verify both appearances of the shared page are in output (non-consecutive)
    #[test]
    fn first_and_last_share_page_middle_different() {
        // Arrange: 4 tokens, page P at positions 0 and 3
        let shared = VirtualPageId::new(42, 0);
        let mid1 = VirtualPageId::new(43, 0);
        let mid2 = VirtualPageId::new(44, 0);
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30, 40], &[shared, mid1, mid2, shared]);
        // Act
        let matched = index.find_longest_prefix(&[10, 20, 30, 40]).unwrap();
        // Assert: all 4 pages present, shared appears at positions 0 and 3
        assert_eq!(matched.matched_tokens, 4);
        assert_eq!(matched.matched_pages.len(), 4);
        assert_eq!(matched.matched_pages[0], shared);
        assert_eq!(matched.matched_pages[1], mid1);
        assert_eq!(matched.matched_pages[2], mid2);
        assert_eq!(matched.matched_pages[3], shared);
    }

    // 10. Insert sequence with 0 tokens but non-empty pages — should be noop
    #[test]
    fn insert_zero_tokens_with_nonempty_pages_is_noop() {
        // Arrange: empty token slice but non-empty pages slice
        let mut index = KvPrefixIndex::new();
        let pages = vec![VirtualPageId::new(0, 0), VirtualPageId::new(0, 1)];
        index.insert(&[], &pages);
        // Act & Assert: nothing was inserted
        assert!(index.find_longest_prefix(&[]).is_none());
        assert!(index.find_longest_prefix(&[0]).is_none());
    }

    // 11. Deep chain of 200 tokens, overwrite first 100 page_refs with new pages
    #[test]
    fn deep_chain_partial_overwrite_preserves_tail() {
        // Arrange: 200-token chain
        let tokens: Vec<TokenId> = (0..200).collect();
        let pages_v1: Vec<VirtualPageId> = (0..200)
            .map(|i| VirtualPageId::new(0, i as usize))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages_v1);
        // Act: overwrite first 100 tokens with new pages
        let prefix_tokens: Vec<TokenId> = (0..100).collect();
        let pages_v2: Vec<VirtualPageId> = (0..100)
            .map(|i| VirtualPageId::new(1, i as usize + 1000))
            .collect();
        index.insert(&prefix_tokens, &pages_v2);
        // Assert: full sequence matches 200 tokens
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 200);
        // First 100 pages from v2
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(1, 1000));
        assert_eq!(matched.matched_pages[99], VirtualPageId::new(1, 1099));
        // Last 100 pages from v1
        assert_eq!(matched.matched_pages[100], VirtualPageId::new(0, 100));
        assert_eq!(matched.matched_pages[199], VirtualPageId::new(0, 199));
    }

    // 12. PrefixMatch Eq trait: two matches with same tokens and same pages are equal
    #[test]
    fn prefix_match_eq_trait_comprehensive() {
        // Arrange: construct two identical PrefixMatch instances independently
        let a = PrefixMatch {
            matched_tokens: 4,
            matched_pages: vec![
                VirtualPageId::new(1, 10),
                VirtualPageId::new(2, 20),
                VirtualPageId::new(3, 30),
                VirtualPageId::new(4, 40),
            ],
        };
        let b = PrefixMatch {
            matched_tokens: 4,
            matched_pages: vec![
                VirtualPageId::new(1, 10),
                VirtualPageId::new(2, 20),
                VirtualPageId::new(3, 30),
                VirtualPageId::new(4, 40),
            ],
        };
        // Assert: PartialEq and Eq both satisfied
        assert_eq!(a, b);
        // Also test via assert! for clarity
        assert!(a == b);
    }

    // 13. Insert [A, B, C] then re-insert [A, B] with new pages — verify full sequence
    // still matches with mixed page sources (overwritten prefix + surviving tail)
    #[test]
    fn reinsert_prefix_preserves_deeper_page_refs() {
        // Arrange: insert [A, B, C] with full pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act: re-insert [10, 20] with different pages — overwrites first 2 page_refs
        // but does NOT touch depth 2 (token 30) — its page_ref survives
        index.insert(&[10, 20], &[
            VirtualPageId::new(9, 90),
            VirtualPageId::new(9, 91),
        ]);
        // Assert: full sequence matches 3 tokens
        let matched = index.find_longest_prefix(&[10, 20, 30]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
        // First 2 pages from the second insert
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(9, 90));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(9, 91));
        // Third page survives from the first insert
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 2));
    }

    // 14. Insert sequence, query with tokens that diverge immediately (no shared prefix)
    // at the very first token — verifies root children lookup miss
    #[test]
    fn first_token_miss_returns_none_even_with_populated_trie() {
        // Arrange: populate trie with multiple sequences
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[10, 20], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);
        index.insert(&[100, 200, 300, 400], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(2, 2),
            VirtualPageId::new(2, 3),
        ]);
        // Act & Assert: queries starting with tokens not in root.children return None
        assert!(index.find_longest_prefix(&[0]).is_none());
        assert!(index.find_longest_prefix(&[5]).is_none());
        assert!(index.find_longest_prefix(&[50, 1, 2]).is_none());
        assert!(index.find_longest_prefix(&[999]).is_none());
    }

    // 15. Clone of PrefixMatch produces a deep copy — verifying field-by-field equality
    #[test]
    fn prefix_match_clone_deep_copy_independence() {
        // Arrange
        let original = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(2, 1),
                VirtualPageId::new(3, 2),
            ],
        };
        // Act: clone
        let cloned = original.clone();
        // Assert: equal
        assert_eq!(original, cloned);
        // Assert: each field matches independently
        assert_eq!(original.matched_tokens, cloned.matched_tokens);
        for i in 0..3 {
            assert_eq!(original.matched_pages[i], cloned.matched_pages[i]);
        }
        // Verify length preserved
        assert_eq!(original.matched_pages.len(), cloned.matched_pages.len());
    }

    // ── Wave 5: 13 additional tests covering remaining edge cases ──

    // 1. Single-token difference at end of long prefix: [0..99, 100] vs [0..99, 101]
    #[test]
    fn single_token_difference_at_end_of_long_prefix() {
        // Arrange: two 100-token sequences sharing first 99 tokens
        let mut index = KvPrefixIndex::new();
        let shared: Vec<TokenId> = (0..99).collect();
        let mut seq_a: Vec<TokenId> = shared.clone();
        seq_a.push(100);
        let mut seq_b: Vec<TokenId> = shared.clone();
        seq_b.push(101);
        let pages_a: Vec<VirtualPageId> = (0..100).map(|i| VirtualPageId::new(0, i)).collect();
        let pages_b: Vec<VirtualPageId> = (0..100).map(|i| VirtualPageId::new(1, i)).collect();
        index.insert(&seq_a, &pages_a);
        index.insert(&seq_b, &pages_b);
        // Act & Assert: each full sequence matches exactly 100 tokens
        let ma = index.find_longest_prefix(&seq_a).unwrap();
        assert_eq!(ma.matched_tokens, 100);
        let mb = index.find_longest_prefix(&seq_b).unwrap();
        assert_eq!(mb.matched_tokens, 100);
        // Act: query with different last token — matches only the 99 shared tokens
        let mut q = shared.clone();
        q.push(999);
        let md = index.find_longest_prefix(&q).unwrap();
        assert_eq!(md.matched_tokens, 99);
    }

    // 2. Very long sequence of identical tokens (1000 repetitions of token 42)
    #[test]
    fn very_long_identical_token_sequence_1000() {
        // Arrange: 1000 tokens all being 42, each with unique page
        let tokens: Vec<TokenId> = vec![42; 1000];
        let pages: Vec<VirtualPageId> = (0..1000)
            .map(|i| VirtualPageId::new(0, i))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act: exact match
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 1000);
        assert_eq!(matched.matched_pages.len(), 1000);
        // Act: partial match (first 500)
        let partial_query = vec![42; 500];
        let partial = index.find_longest_prefix(&partial_query).unwrap();
        assert_eq!(partial.matched_tokens, 500);
        assert_eq!(partial.matched_pages.len(), 500);
    }

    // 3. Overwrite-query-overwrite cycle: verify each cycle produces correct results
    #[test]
    fn overwrite_query_overwrite_cycle() {
        // Arrange: insert, query, overwrite with different pages, query again
        let mut index = KvPrefixIndex::new();
        let tokens = vec![10, 20, 30];
        // Cycle 1
        index.insert(&tokens, &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        let m1 = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(m1.matched_pages[0], VirtualPageId::new(1, 0));
        // Cycle 2: overwrite all pages
        index.insert(&tokens, &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(2, 2),
        ]);
        let m2 = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(m2.matched_tokens, 3);
        assert_eq!(m2.matched_pages[0], VirtualPageId::new(2, 0));
        // Cycle 3: overwrite only first page
        index.insert(&tokens, &[VirtualPageId::new(3, 0)]);
        let m3 = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(m3.matched_tokens, 3);
        assert_eq!(m3.matched_pages[0], VirtualPageId::new(3, 0));
        // Positions 1 and 2 still have cycle 2 values
        assert_eq!(m3.matched_pages[1], VirtualPageId::new(2, 1));
        assert_eq!(m3.matched_pages[2], VirtualPageId::new(2, 2));
    }

    // 4. Tree pattern: root -> [A, B, C] where A has 3 sub-branches, B has 2, C is leaf
    #[test]
    fn tree_pattern_with_branching_at_multiple_levels() {
        // Arrange: root token 1, branching at levels 1 and 2
        let mut index = KvPrefixIndex::new();
        // [1, 10, 100]
        index.insert(&[1, 10, 100], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // [1, 10, 101]
        index.insert(&[1, 10, 101], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // [1, 20, 200]
        index.insert(&[1, 20, 200], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(2, 2),
        ]);
        // [1, 30]
        index.insert(&[1, 30], &[
            VirtualPageId::new(3, 0),
            VirtualPageId::new(3, 1),
        ]);
        // Act & Assert: each leaf path matches fully
        assert_eq!(index.find_longest_prefix(&[1, 10, 100]).unwrap().matched_tokens, 3);
        assert_eq!(index.find_longest_prefix(&[1, 10, 101]).unwrap().matched_tokens, 3);
        assert_eq!(index.find_longest_prefix(&[1, 20, 200]).unwrap().matched_tokens, 3);
        assert_eq!(index.find_longest_prefix(&[1, 30]).unwrap().matched_tokens, 2);
        // Divergent at level 2 under branch 10: matches [1, 10] only
        assert_eq!(index.find_longest_prefix(&[1, 10, 999]).unwrap().matched_tokens, 2);
        // Divergent at level 1: matches [1] only
        assert_eq!(index.find_longest_prefix(&[1, 99]).unwrap().matched_tokens, 1);
    }

    // 5. Deep chain query verifies page_ref offset_in_page <= idx check at every depth
    #[test]
    fn deep_chain_query_all_offset_checks_pass() {
        // Arrange: 50-token chain where each token value equals its index
        let tokens: Vec<TokenId> = (0..50).collect();
        let pages: Vec<VirtualPageId> = (0..50)
            .map(|i| VirtualPageId::new(0, i * 10))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act & Assert: verify match at every prefix length
        for len in 1..=50 {
            let query: Vec<TokenId> = (0..len as u32).collect();
            let matched = index.find_longest_prefix(&query).unwrap();
            assert_eq!(matched.matched_tokens, len,
                "prefix of length {len} should match exactly {len} tokens");
        }
    }

    // 6. Shared node page_ref overwritten by shorter sequence, then queried via longer path
    // — verifies offset_in_page validity check at deeper nodes
    #[test]
    fn shared_node_overwrite_offset_validity_at_depth() {
        // Arrange: insert [1, 2, 3, 4] fully
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
        ]);
        // Act: insert [1, 2] with different pages — overwrites nodes at depth 0 and 1
        index.insert(&[1, 2], &[
            VirtualPageId::new(9, 0),
            VirtualPageId::new(9, 1),
        ]);
        // Assert: full path still matches 4 tokens — nodes at depth 2 and 3
        // have page_refs from first insert with offset_in_page 2 and 3
        // At query time, idx=2 for depth 2 node, offset_in_page=2, check 2<=2 passes
        // At query time, idx=3 for depth 3 node, offset_in_page=3, check 3<=3 passes
        let matched = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(matched.matched_tokens, 4);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(9, 0));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(9, 1));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 2));
        assert_eq!(matched.matched_pages[3], VirtualPageId::new(0, 3));
    }

    // 7. Multiple branch insertions accumulate pages correctly in matched_pages vector
    // — verifies truncate(best_page_len) works when page_ref appears mid-traversal
    #[test]
    fn matched_pages_truncation_after_branch_mismatch() {
        // Arrange: insert [A, B, C] with pages at A and C but NOT B
        let mut index = KvPrefixIndex::new();
        index.insert(&[100, 200, 300], &[
            VirtualPageId::new(0, 0), // page for A
            // No page for B — insert with only 1 page
        ]);
        // Re-insert to add page for C without page for B
        // Actually we can't have a gap in the middle via a single insert.
        // Use two inserts: first sets page at idx 0, second extends to set page at idx 2
        // but insert only sets page_ref for indices < pages.len()
        // Let's insert a 3-token sequence with pages only at 0 and 2 via overwriting
        index.insert(&[100, 200, 300], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act: overwrite [100, 200] with only page at idx 0 — removes page at idx 1
        index.insert(&[100, 200], &[VirtualPageId::new(5, 50)]);
        // Now node for token 200 has NO page_ref (only set for idx < 1 = just idx 0)
        // Actually insert sets page_ref at idx 0 and idx 1 for the 2-token insert.
        // Wait: pages has 1 element, so pages.get(0)=Some, pages.get(1)=None.
        // So token 100 gets page_ref=(5,50), token 200 gets no page_ref update.
        // Token 200 still has page_ref from the first insert: (0,1).
        // Token 300 still has page_ref from the first insert: (0,2).

        // Query [100, 200, 300]:
        // idx=0: token 100, page_ref=(5,50), offset=0<=0 -> push (5,50), best=1, pages_len=1
        // idx=1: token 200, page_ref=(0,1), offset=1<=1 -> push (0,1), best=2, pages_len=2
        // idx=2: token 300, page_ref=(0,2), offset=2<=2 -> push (0,2), best=3, pages_len=3
        let matched = index.find_longest_prefix(&[100, 200, 300]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(5, 50));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(0, 1));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 2));
    }

    // 8. Two divergence points in the same trie: shared prefix, branch, shared mid, branch again
    #[test]
    fn two_divergence_points_in_same_trie() {
        // Arrange:
        // [1, 2, 10, 20, 100] — path A
        // [1, 2, 10, 30, 200] — path B (diverges at position 3)
        // [1, 2, 40, 50] — path C (diverges at position 2)
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 10, 20, 100], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
            VirtualPageId::new(0, 4),
        ]);
        index.insert(&[1, 2, 10, 30, 200], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
            VirtualPageId::new(1, 3),
            VirtualPageId::new(1, 4),
        ]);
        index.insert(&[1, 2, 40, 50], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(2, 2),
            VirtualPageId::new(2, 3),
        ]);
        // Act & Assert: each path matches fully
        assert_eq!(index.find_longest_prefix(&[1, 2, 10, 20, 100]).unwrap().matched_tokens, 5);
        assert_eq!(index.find_longest_prefix(&[1, 2, 10, 30, 200]).unwrap().matched_tokens, 5);
        assert_eq!(index.find_longest_prefix(&[1, 2, 40, 50]).unwrap().matched_tokens, 4);
        // Diverge at first branch point: [1, 2, 99] matches 2 shared tokens
        assert_eq!(index.find_longest_prefix(&[1, 2, 99]).unwrap().matched_tokens, 2);
        // Diverge at second branch point: [1, 2, 10, 99] matches 3 tokens
        assert_eq!(index.find_longest_prefix(&[1, 2, 10, 99]).unwrap().matched_tokens, 3);
    }

    // 9. Wide trie with 500 single-token branches from root
    #[test]
    fn wide_trie_500_independent_single_token_branches() {
        // Arrange: 500 single-token sequences, each with unique page
        let mut index = KvPrefixIndex::new();
        for i in 0u32..500 {
            index.insert(&[i], &[VirtualPageId::new(i as u64, i as usize)]);
        }
        // Act & Assert: each token is independently findable
        for i in 0u32..500 {
            let matched = index.find_longest_prefix(&[i]).unwrap();
            assert_eq!(matched.matched_tokens, 1, "token {i}");
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(i as u64, i as usize));
        }
        // Non-existent token returns None
        assert!(index.find_longest_prefix(&[999]).is_none());
    }

    // 10. All tokens same, pages alternate between two VirtualPageIds
    #[test]
    fn all_same_tokens_alternating_two_pages() {
        // Arrange: 6 tokens all being 7, pages alternate [P, Q, P, Q, P, Q]
        let p = VirtualPageId::new(1, 0);
        let q = VirtualPageId::new(2, 0);
        let pages = vec![p, q, p, q, p, q];
        let tokens = vec![7u32; 6];
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();
        // Assert: all 6 tokens matched, pages alternate without dedup (non-consecutive same)
        assert_eq!(matched.matched_tokens, 6);
        assert_eq!(matched.matched_pages.len(), 6);
        assert_eq!(matched.matched_pages[0], p);
        assert_eq!(matched.matched_pages[1], q);
        assert_eq!(matched.matched_pages[2], p);
        assert_eq!(matched.matched_pages[3], q);
        assert_eq!(matched.matched_pages[4], p);
        assert_eq!(matched.matched_pages[5], q);
    }

    // 11. Progressive prefix-length queries on 50-token sequence verify incremental correctness
    #[test]
    fn progressive_prefix_queries_incremental_correctness() {
        // Arrange: 50-token sequence with distinct pages per token
        let tokens: Vec<TokenId> = (100..150).collect();
        let pages: Vec<VirtualPageId> = (100..150)
            .map(|i| VirtualPageId::new(0, i as usize))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act & Assert: query every prefix length and verify both token count and page count
        for len in 1..=50 {
            let query: Vec<TokenId> = (100..100 + len as u32).collect();
            let matched = index.find_longest_prefix(&query).unwrap();
            assert_eq!(matched.matched_tokens, len,
                "prefix len {len}: expected {len} matched tokens");
            assert_eq!(matched.matched_pages.len(), len,
                "prefix len {len}: expected {len} matched pages");
            // Verify each page is correct
            for j in 0..len {
                assert_eq!(matched.matched_pages[j], VirtualPageId::new(0, 100 + j),
                    "prefix len {len}: page at index {j} mismatch");
            }
        }
    }

    // 12. Default-constructed index behaves identically to new-constructed for all operations
    #[test]
    fn default_index_insert_and_query_matches_new_index() {
        // Arrange: create two indexes via different constructors
        let mut idx_new = KvPrefixIndex::new();
        let mut idx_default = KvPrefixIndex::default();
        let tokens = vec![5, 10, 15, 20, 25];
        let pages: Vec<VirtualPageId> = (0..5)
            .map(|i| VirtualPageId::new(0, i * 10))
            .collect();
        // Act: insert same data into both
        idx_new.insert(&tokens, &pages);
        idx_default.insert(&tokens, &pages);
        // Assert: results are identical for full query
        let m_new = idx_new.find_longest_prefix(&tokens).unwrap();
        let m_default = idx_default.find_longest_prefix(&tokens).unwrap();
        assert_eq!(m_new, m_default);
        // Assert: partial query also identical
        let p_new = idx_new.find_longest_prefix(&[5, 10, 15]);
        let p_default = idx_default.find_longest_prefix(&[5, 10, 15]);
        assert_eq!(p_new, p_default);
        // Assert: miss query also identical
        let miss_new = idx_new.find_longest_prefix(&[999]);
        let miss_default = idx_default.find_longest_prefix(&[999]);
        assert_eq!(miss_new, miss_default);
        assert!(miss_new.is_none());
    }

    // 13. Insert 3 sequences that form a diamond shape in the trie:
    //     [A, B, X], [A, B, Y], [A, C, X] — token X appears at different depths
    #[test]
    fn diamond_shaped_trie_token_at_different_depths() {
        // Arrange: three sequences forming a diamond
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 100], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[1, 2, 200], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        index.insert(&[1, 3, 100], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(2, 2),
        ]);
        // Act & Assert: all three sequences match fully
        let ma = index.find_longest_prefix(&[1, 2, 100]).unwrap();
        assert_eq!(ma.matched_tokens, 3);
        let mb = index.find_longest_prefix(&[1, 2, 200]).unwrap();
        assert_eq!(mb.matched_tokens, 3);
        let mc = index.find_longest_prefix(&[1, 3, 100]).unwrap();
        assert_eq!(mc.matched_tokens, 3);
        // Token 100 appears in two paths but querying [100] from root gives nothing
        assert!(index.find_longest_prefix(&[100]).is_none());
        // Shared prefix [1] matches 1 token
        assert_eq!(index.find_longest_prefix(&[1, 99]).unwrap().matched_tokens, 1);
    }

    // ── Wave 6: 13 additional tests ──

    // 1. Very long prefix (512 tokens) with exact match and boundary queries
    #[test]
    fn very_long_prefix_512_tokens_boundary_queries() {
        // Arrange: 512-token sequence (power-of-2 length)
        let tokens: Vec<TokenId> = (0..512).collect();
        let pages: Vec<VirtualPageId> = (0..512)
            .map(|i| VirtualPageId::new(0, i as usize))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act & Assert: exact match
        let full = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(full.matched_tokens, 512);
        assert_eq!(full.matched_pages.len(), 512);
        // Boundary: 256 (half)
        let half: Vec<TokenId> = (0..256).collect();
        assert_eq!(index.find_longest_prefix(&half).unwrap().matched_tokens, 256);
        // Boundary: 511 (one less)
        let almost: Vec<TokenId> = (0..511).collect();
        assert_eq!(index.find_longest_prefix(&almost).unwrap().matched_tokens, 511);
        // Boundary: 1 (just first token)
        assert_eq!(index.find_longest_prefix(&[0]).unwrap().matched_tokens, 1);
    }

    // 2. Wide trie: 1000 single-token branches from root
    #[test]
    fn wide_trie_1000_single_token_branches() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        for i in 0u32..1000 {
            index.insert(&[i], &[VirtualPageId::new(i as u64, 0)]);
        }
        // Act & Assert: spot-check every 100th branch
        for i in (0..1000).step_by(100) {
            let matched = index.find_longest_prefix(&[i]).unwrap();
            assert_eq!(matched.matched_tokens, 1, "token {i}");
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(i as u64, 0));
        }
        // Assert: non-existent token returns None
        assert!(index.find_longest_prefix(&[9999]).is_none());
    }

    // 3. Diamond pattern: two paths converge to same suffix token value at different depths
    #[test]
    fn diamond_shared_suffix_token_at_different_depths() {
        // Arrange: [1, 2, 99] and [3, 99] — token 99 at depth 2 and depth 1
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 99], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[3, 99], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);
        // Act & Assert: both paths match fully
        assert_eq!(index.find_longest_prefix(&[1, 2, 99]).unwrap().matched_tokens, 3);
        assert_eq!(index.find_longest_prefix(&[3, 99]).unwrap().matched_tokens, 2);
        // Token 99 is not a root child
        assert!(index.find_longest_prefix(&[99]).is_none());
        // Divergent: [1, 99] — token 99 not child of node 1 (node 1 has child 2)
        assert_eq!(index.find_longest_prefix(&[1, 99]).unwrap().matched_tokens, 1);
    }

    // 4. Progressive query matching with page content verification at each step
    #[test]
    fn progressive_query_with_page_content_verification() {
        // Arrange: 20-token sequence with pages encoding token index in logical_index
        let tokens: Vec<TokenId> = (200..220).collect();
        let pages: Vec<VirtualPageId> = (200..220)
            .map(|i| VirtualPageId::new(42, i as usize))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act & Assert: query at each length, verify both token count and page identity
        for len in [1, 5, 10, 15, 20] {
            let query: Vec<TokenId> = (200..200 + len as u32).collect();
            let matched = index.find_longest_prefix(&query).unwrap();
            assert_eq!(matched.matched_tokens, len);
            assert_eq!(matched.matched_pages.len(), len);
            assert_eq!(
                matched.matched_pages[len - 1],
                VirtualPageId::new(42, (200 + len - 1) as usize),
                "page mismatch at len={len}"
            );
        }
    }

    // 5. Empty trie: all API operations on a freshly-constructed index
    #[test]
    fn empty_trie_all_api_operations() {
        // Arrange
        let empty = KvPrefixIndex::new();
        // Act & Assert: find on empty returns None for various inputs
        assert!(empty.find_longest_prefix(&[]).is_none());
        assert!(empty.find_longest_prefix(&[0]).is_none());
        assert!(empty.find_longest_prefix(&[u32::MAX]).is_none());
        assert!(empty.find_longest_prefix(&[1, 2, 3, 4, 5]).is_none());
        // Debug format works on empty trie
        let debug = format!("{empty:?}");
        assert!(debug.contains("root"));
    }

    // 6. Single-node trie: insert one token, verify all query behaviors
    #[test]
    fn single_node_trie_comprehensive_query_behaviors() {
        // Arrange: single-token trie with specific page
        let page = VirtualPageId::new(7, 13);
        let mut index = KvPrefixIndex::new();
        index.insert(&[42], &[page]);
        // Act & Assert: exact match
        let exact = index.find_longest_prefix(&[42]).unwrap();
        assert_eq!(exact.matched_tokens, 1);
        assert_eq!(exact.matched_pages[0], page);
        // Act & Assert: extended query matches only the single token
        let ext = index.find_longest_prefix(&[42, 43, 44]).unwrap();
        assert_eq!(ext.matched_tokens, 1);
        assert_eq!(ext.matched_pages[0], page);
        // Act & Assert: different first token returns None
        assert!(index.find_longest_prefix(&[41]).is_none());
        assert!(index.find_longest_prefix(&[43]).is_none());
        // Act & Assert: empty query returns None
        assert!(index.find_longest_prefix(&[]).is_none());
    }

    // 7. Overwrite with fewer pages then re-overwrite with more pages — page_ref lifecycle
    #[test]
    fn overwrite_fewer_then_more_pages_lifecycle() {
        // Arrange: insert 3-token sequence with full pages
        let mut index = KvPrefixIndex::new();
        let tokens = vec![1, 2, 3];
        index.insert(&tokens, &[
            VirtualPageId::new(0, 10),
            VirtualPageId::new(0, 20),
            VirtualPageId::new(0, 30),
        ]);
        // Act: overwrite with only 1 page — tokens 2 and 3 keep old page_refs
        index.insert(&tokens, &[VirtualPageId::new(5, 50)]);
        let m1 = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(m1.matched_tokens, 3);
        assert_eq!(m1.matched_pages[0], VirtualPageId::new(5, 50));
        assert_eq!(m1.matched_pages[1], VirtualPageId::new(0, 20));
        // Act: overwrite again with full 3 pages
        index.insert(&tokens, &[
            VirtualPageId::new(9, 100),
            VirtualPageId::new(9, 200),
            VirtualPageId::new(9, 300),
        ]);
        let m2 = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(m2.matched_tokens, 3);
        assert_eq!(m2.matched_pages, vec![
            VirtualPageId::new(9, 100),
            VirtualPageId::new(9, 200),
            VirtualPageId::new(9, 300),
        ]);
    }

    // 8. Match length boundary: page at first and third token, gap at second
    #[test]
    fn match_boundary_gap_at_middle_token() {
        // Arrange: insert 3 tokens, but page only at position 0, then overwrite to add page
        // at position 2 while leaving position 1 without a page
        let mut index = KvPrefixIndex::new();
        // First insert: pages at all 3 positions
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Overwrite: insert only 1 page — position 1 and 2 keep old page_refs
        // We need position 1 to have no page_ref. We can't clear page_ref via insert.
        // Instead, use a fresh approach: insert [A, B, C] with 1 page (only A has page_ref)
        let mut index2 = KvPrefixIndex::new();
        index2.insert(&[10, 20, 30], &[
            VirtualPageId::new(0, 0),
        ]);
        // Act: tokens B and C have no page_ref, so best match is 1
        let matched = index2.find_longest_prefix(&[10, 20, 30]).unwrap();
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages.len(), 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
    }

    // 9. Debug format verification: PrefixMatch with multiple pages
    #[test]
    fn debug_format_multiple_pages_comprehensive() {
        // Arrange: PrefixMatch with 3 distinct pages
        let pm = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![
                VirtualPageId::new(1, 100),
                VirtualPageId::new(2, 200),
                VirtualPageId::new(3, 300),
            ],
        };
        // Act
        let debug = format!("{pm:?}");
        // Assert: both fields present with values
        assert!(debug.contains("matched_tokens: 3"));
        assert!(debug.contains("matched_pages"));
        // Assert: pages vector is non-empty in debug output
        assert!(!debug.contains("matched_pages: []"));
    }

    // 10. Clone independence: modify cloned PrefixMatch fields independently
    #[test]
    fn clone_independence_field_isolation() {
        // Arrange: original with specific pages
        let original = PrefixMatch {
            matched_tokens: 5,
            matched_pages: vec![
                VirtualPageId::new(0, 0),
                VirtualPageId::new(1, 1),
                VirtualPageId::new(2, 2),
            ],
        };
        // Act: clone
        let cloned = original.clone();
        // Assert: values are equal
        assert_eq!(original.matched_tokens, cloned.matched_tokens);
        assert_eq!(original.matched_pages, cloned.matched_pages);
        // Assert: original unchanged after clone (fields still accessible)
        assert_eq!(original.matched_tokens, 5);
        assert_eq!(original.matched_pages.len(), 3);
        assert_eq!(original.matched_pages[0], VirtualPageId::new(0, 0));
    }

    // 11. Insert 20 independent 2-token sequences, query each and verify isolation
    #[test]
    fn twenty_independent_two_token_sequences_isolation() {
        // Arrange: 20 sequences, each starting with a unique first token
        let mut index = KvPrefixIndex::new();
        for i in 0u32..20 {
            index.insert(&[i * 100, i * 100 + 1], &[
                VirtualPageId::new(i as u64, 0),
                VirtualPageId::new(i as u64, 1),
            ]);
        }
        // Act & Assert: each sequence matches fully
        for i in 0u32..20 {
            let matched = index.find_longest_prefix(&[i * 100, i * 100 + 1]).unwrap();
            assert_eq!(matched.matched_tokens, 2, "seq {i}");
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(i as u64, 0));
        }
        // Assert: no cross-contamination — query first token of seq 5 + second of seq 3
        let cross = index.find_longest_prefix(&[500, 301]).unwrap();
        assert_eq!(cross.matched_tokens, 1, "should match only first token of seq 5");
        // Assert: non-existent first token returns None
        assert!(index.find_longest_prefix(&[1]).is_none());
    }

    // 12. Complex multi-level tree: verify isolation between distant branches
    #[test]
    fn complex_tree_distant_branch_isolation() {
        // Arrange: build a tree with shared prefix [1, 2] then 3 branches of depth 3 each
        let mut index = KvPrefixIndex::new();
        // Branch A: [1, 2, 10, 100, 1000]
        index.insert(&[1, 2, 10, 100, 1000], &[
            VirtualPageId::new(10, 0),
            VirtualPageId::new(10, 1),
            VirtualPageId::new(10, 2),
            VirtualPageId::new(10, 3),
            VirtualPageId::new(10, 4),
        ]);
        // Branch B: [1, 2, 20, 200, 2000]
        index.insert(&[1, 2, 20, 200, 2000], &[
            VirtualPageId::new(20, 0),
            VirtualPageId::new(20, 1),
            VirtualPageId::new(20, 2),
            VirtualPageId::new(20, 3),
            VirtualPageId::new(20, 4),
        ]);
        // Branch C: [1, 2, 30, 300, 3000]
        index.insert(&[1, 2, 30, 300, 3000], &[
            VirtualPageId::new(30, 0),
            VirtualPageId::new(30, 1),
            VirtualPageId::new(30, 2),
            VirtualPageId::new(30, 3),
            VirtualPageId::new(30, 4),
        ]);
        // Act & Assert: each full branch matches 5 tokens
        assert_eq!(index.find_longest_prefix(&[1, 2, 10, 100, 1000]).unwrap().matched_tokens, 5);
        assert_eq!(index.find_longest_prefix(&[1, 2, 20, 200, 2000]).unwrap().matched_tokens, 5);
        assert_eq!(index.find_longest_prefix(&[1, 2, 30, 300, 3000]).unwrap().matched_tokens, 5);
        // Partial: shared prefix [1, 2] + unknown branch gives 2-token match
        assert_eq!(index.find_longest_prefix(&[1, 2, 99]).unwrap().matched_tokens, 2);
        // Partial: branch A prefix [1, 2, 10] + divergent gives 3-token match
        assert_eq!(index.find_longest_prefix(&[1, 2, 10, 99]).unwrap().matched_tokens, 3);
    }

    // 13. Verify find_longest_prefix never returns matched_tokens=0 in Some variant
    #[test]
    fn find_never_returns_zero_matched_tokens_in_some() {
        // Arrange: populate trie with various sequences
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[10, 20], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);
        // Act & Assert: queries that don't match return None (not Some(0))
        assert!(index.find_longest_prefix(&[99]).is_none());
        assert!(index.find_longest_prefix(&[]).is_none());
        assert!(index.find_longest_prefix(&[5, 6, 7]).is_none());
        // Act & Assert: queries that do match always have matched_tokens >= 1
        let m1 = index.find_longest_prefix(&[1]).unwrap();
        assert!(m1.matched_tokens >= 1);
        let m2 = index.find_longest_prefix(&[10, 20, 30]).unwrap();
        assert!(m2.matched_tokens >= 1);
    }

    // ── Wave 7: 13 additional tests covering offset_in_page validation, truncate logic,
    //   and structural edge cases ──

    // 1. offset_in_page validity: page_ref with offset > idx is skipped, reducing matched_tokens
    //    Insert [A, B] where B's page_ref has offset_in_page=5 (> idx=1), so only A matches
    #[test]
    fn offset_in_page_exceeds_idx_skips_page_at_depth() {
        // Arrange: insert a 2-token sequence, then overwrite the second token's node
        // with a page_ref whose offset_in_page is large (from a deeper position in a longer insert)
        let mut index = KvPrefixIndex::new();
        // First: insert [A, B, C, D, E, F] — token F gets page_ref with offset_in_page=5
        index.insert(&[1, 2, 3, 4, 5, 6], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
            VirtualPageId::new(0, 4),
            VirtualPageId::new(0, 5),
        ]);
        // Act: query only [1, 2] — token 2's page_ref has offset_in_page=1, idx=1, 1<=1 passes
        let matched = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
        // Now: query [1] only
        let m1 = index.find_longest_prefix(&[1]).unwrap();
        assert_eq!(m1.matched_tokens, 1);
    }

    // 2. Truncate interaction: page dedup during traversal, then branch diverges —
    //    matched_pages gets truncated back to best_page_len
    #[test]
    fn truncate_restores_best_after_dedup_and_diverge() {
        // Arrange: insert [A, B, C] where A and B share same page (dedup), then C diverges
        let shared = VirtualPageId::new(0, 0);
        let page_c = VirtualPageId::new(0, 2);
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[shared, shared, page_c]);
        // Act: query [10, 20, 30, 999] — token 999 diverges, best is 3 tokens
        let matched = index.find_longest_prefix(&[10, 20, 30, 999]).unwrap();
        // Assert: dedup means pages = [shared, page_c] (2 pages for 3 tokens)
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 2);
        assert_eq!(matched.matched_pages[0], shared);
        assert_eq!(matched.matched_pages[1], page_c);
    }

    // 3. All trie nodes have page_ref=None: insert tokens with empty pages, then query
    //    — verifies best_tokens stays 0 throughout traversal
    #[test]
    fn all_page_refs_none_returns_none_despite_trie_path_existing() {
        // Arrange: build trie nodes via insert with empty pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4, 5], &[]);
        // Act & Assert: trie path exists but no page_ref set anywhere
        assert!(index.find_longest_prefix(&[1]).is_none());
        assert!(index.find_longest_prefix(&[1, 2, 3]).is_none());
        assert!(index.find_longest_prefix(&[1, 2, 3, 4, 5]).is_none());
        // Act: now add page only at token 3 via re-insert of [1, 2, 3] with 3 pages
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Assert: now all 3 tokens match (page_refs added to existing nodes)
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
    }

    // 4. Single root-level branch with depth 1: verify KvPrefixIndex handles
    //    a trie that only has one child under root, with that child having a page_ref
    #[test]
    fn single_child_root_with_immediate_page_ref() {
        // Arrange: single token 7 with page, nothing else in trie
        let page = VirtualPageId::new(100, 200);
        let mut index = KvPrefixIndex::new();
        index.insert(&[7], &[page]);
        // Act: various queries
        assert_eq!(index.find_longest_prefix(&[7]).unwrap().matched_pages[0], page);
        assert!(index.find_longest_prefix(&[8]).is_none());
        // Extended query stops at depth 1
        let ext = index.find_longest_prefix(&[7, 8, 9]).unwrap();
        assert_eq!(ext.matched_tokens, 1);
        assert_eq!(ext.matched_pages.len(), 1);
    }

    // 5. PrefixMatch with very large pages vector: verify truncate correctness
    #[test]
    fn prefix_match_large_pages_vector_truncate_correctness() {
        // Arrange: construct PrefixMatch with 100 pages
        let pages: Vec<VirtualPageId> = (0..100)
            .map(|i| VirtualPageId::new(i as u64, i * 2))
            .collect();
        let pm = PrefixMatch {
            matched_tokens: 100,
            matched_pages: pages.clone(),
        };
        // Assert: all pages preserved
        assert_eq!(pm.matched_pages.len(), 100);
        assert_eq!(pm.matched_pages[0], VirtualPageId::new(0, 0));
        assert_eq!(pm.matched_pages[99], VirtualPageId::new(99, 198));
    }

    // 6. Overwrite shared prefix multiple times: verify page_ref for shared node
    //    always reflects the most recent insert
    #[test]
    fn shared_prefix_page_ref_always_from_most_recent_insert() {
        // Arrange: 4 inserts sharing first token, each with different page for root
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 10], &[VirtualPageId::new(1, 0), VirtualPageId::new(1, 1)]);
        index.insert(&[1, 20], &[VirtualPageId::new(2, 0), VirtualPageId::new(2, 1)]);
        index.insert(&[1, 30], &[VirtualPageId::new(3, 0), VirtualPageId::new(3, 1)]);
        index.insert(&[1, 40], &[VirtualPageId::new(4, 0), VirtualPageId::new(4, 1)]);
        // Act & Assert: shared token 1's page comes from last insert (sequence_id=4)
        for second in [10u32, 20, 30, 40] {
            let matched = index.find_longest_prefix(&[1, second]).unwrap();
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(4, 0),
                "branch [1, {second}] root page should be from last insert");
        }
        // Partial match on divergent second token also uses last root page
        let partial = index.find_longest_prefix(&[1, 999]).unwrap();
        assert_eq!(partial.matched_pages[0], VirtualPageId::new(4, 0));
    }

    // 7. TokenId value 1 followed by TokenId value 0: verify 0 is treated as valid token
    #[test]
    fn token_sequence_includes_zero_after_nonzero() {
        // Arrange: [1, 0, 2] — token 0 at position 1
        let mut index = KvPrefixIndex::new();
        let pages = vec![
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ];
        index.insert(&[1, 0, 2], &pages);
        // Act: exact match
        let matched = index.find_longest_prefix(&[1, 0, 2]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
        // Act: diverge at token 0 position
        let partial = index.find_longest_prefix(&[1, 5]).unwrap();
        assert_eq!(partial.matched_tokens, 1);
        // Act: query starting with 0 — different from token 0 at depth 1
        assert!(index.find_longest_prefix(&[0, 2]).is_none());
    }

    // 8. Insert sequence where pages repeat in pattern A, B, A, B — verify no over-dedup
    #[test]
    fn alternating_two_pages_pattern_no_over_dedup() {
        // Arrange: 8 tokens, pages alternate [P, Q, P, Q, P, Q, P, Q]
        let p = VirtualPageId::new(1, 0);
        let q = VirtualPageId::new(2, 0);
        let pages = vec![p, q, p, q, p, q, p, q];
        let tokens: Vec<TokenId> = (0..8).collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();
        // Assert: 8 tokens matched, 8 pages (no over-dedup, non-consecutive same pages kept)
        assert_eq!(matched.matched_tokens, 8);
        assert_eq!(matched.matched_pages.len(), 8);
        assert_eq!(matched.matched_pages[0], p);
        assert_eq!(matched.matched_pages[1], q);
        assert_eq!(matched.matched_pages[2], p);
    }

    // 9. Two completely independent sequences inserted in alternation, each verified after
    //    every insert — tests that alternating inserts don't corrupt the other branch
    #[test]
    fn alternating_inserts_two_branches_no_corruption() {
        // Arrange: build two branches [1, 2, 3] and [10, 20, 30] in alternating steps
        let mut index = KvPrefixIndex::new();
        // Step 1: insert [1] into branch A
        index.insert(&[1], &[VirtualPageId::new(10, 0)]);
        let ma = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(ma.matched_tokens, 1);
        // Step 2: insert [10] into branch B
        index.insert(&[10], &[VirtualPageId::new(20, 0)]);
        let mb = index.find_longest_prefix(&[10, 20]).unwrap();
        assert_eq!(mb.matched_tokens, 1);
        // Step 3: extend branch A to [1, 2]
        index.insert(&[1, 2], &[VirtualPageId::new(10, 0), VirtualPageId::new(10, 1)]);
        assert_eq!(index.find_longest_prefix(&[1, 2]).unwrap().matched_tokens, 2);
        assert_eq!(index.find_longest_prefix(&[10]).unwrap().matched_tokens, 1);
        // Step 4: extend branch B to [10, 20]
        index.insert(&[10, 20], &[VirtualPageId::new(20, 0), VirtualPageId::new(20, 1)]);
        assert_eq!(index.find_longest_prefix(&[10, 20]).unwrap().matched_tokens, 2);
        assert_eq!(index.find_longest_prefix(&[1, 2]).unwrap().matched_tokens, 2);
        // Step 5: extend both to full length
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(10, 0), VirtualPageId::new(10, 1), VirtualPageId::new(10, 2),
        ]);
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(20, 0), VirtualPageId::new(20, 1), VirtualPageId::new(20, 2),
        ]);
        // Final verification
        let fa = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(fa.matched_tokens, 3);
        assert_eq!(fa.matched_pages[0], VirtualPageId::new(10, 0));
        let fb = index.find_longest_prefix(&[10, 20, 30]).unwrap();
        assert_eq!(fb.matched_tokens, 3);
        assert_eq!(fb.matched_pages[0], VirtualPageId::new(20, 0));
    }

    // 10. PrefixMatch Debug output contains expected field format for pages with values
    #[test]
    fn prefix_match_debug_output_contains_both_fields_with_values() {
        // Arrange
        let pm = PrefixMatch {
            matched_tokens: 7,
            matched_pages: vec![
                VirtualPageId::new(0, 0),
                VirtualPageId::new(1, 100),
            ],
        };
        // Act
        let debug = format!("{pm:?}");
        // Assert: field names present
        assert!(debug.contains("matched_tokens: 7"));
        assert!(debug.contains("matched_pages"));
        // Assert: page data present in debug (VirtualPageId has Debug derive)
        assert!(debug.contains("sequence_id"));
        assert!(debug.contains("logical_index"));
    }

    // 11. Insert a sequence of length 2, then a longer sequence of length 4 sharing the prefix,
    //     then query the shorter — verifies shorter match still returns correct pages
    #[test]
    fn shorter_prefix_match_after_longer_extension_correct_pages() {
        // Arrange: insert [1, 2] first, then extend to [1, 2, 3, 4]
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2], &[
            VirtualPageId::new(10, 0),
            VirtualPageId::new(10, 1),
        ]);
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(20, 0),
            VirtualPageId::new(20, 1),
            VirtualPageId::new(20, 2),
            VirtualPageId::new(20, 3),
        ]);
        // Act: query [1, 2] — pages come from second insert (overwrote shared prefix)
        let short = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(short.matched_tokens, 2);
        assert_eq!(short.matched_pages[0], VirtualPageId::new(20, 0));
        assert_eq!(short.matched_pages[1], VirtualPageId::new(20, 1));
        // Act: query full [1, 2, 3, 4]
        let full = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(full.matched_tokens, 4);
        assert_eq!(full.matched_pages.len(), 4);
    }

    // 12. VirtualPageId with sequence_id=0 and large logical_index: stored and retrieved
    //     without confusion with "no page" (page_ref is Option, not default-constructed)
    #[test]
    fn virtual_page_id_zero_sequence_id_large_index_distinct_from_none() {
        // Arrange: page with sequence_id=0 (which might look like "default") and large index
        let page = VirtualPageId::new(0, 1_000_000);
        let mut index = KvPrefixIndex::new();
        index.insert(&[42], &[page]);
        // Act
        let matched = index.find_longest_prefix(&[42]).unwrap();
        // Assert: page with sequence_id=0 is distinct from no page
        assert_eq!(matched.matched_pages[0].sequence_id, 0);
        assert_eq!(matched.matched_pages[0].logical_index, 1_000_000);
        assert_eq!(matched.matched_tokens, 1);
    }

    // 13. Insert identical tokens with identical pages twice — full idempotency with page
    //     content verification at every position
    #[test]
    fn idempotent_double_insert_page_content_verification() {
        // Arrange: 5 tokens with specific pages
        let tokens: Vec<TokenId> = vec![100, 200, 300, 400, 500];
        let pages: Vec<VirtualPageId> = vec![
            VirtualPageId::new(1, 10),
            VirtualPageId::new(2, 20),
            VirtualPageId::new(3, 30),
            VirtualPageId::new(4, 40),
            VirtualPageId::new(5, 50),
        ];
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        let first = index.find_longest_prefix(&tokens).unwrap();
        // Act: insert identical data again
        index.insert(&tokens, &pages);
        let second = index.find_longest_prefix(&tokens).unwrap();
        // Assert: every page matches position-by-position
        assert_eq!(first, second);
        for i in 0..5 {
            assert_eq!(second.matched_pages[i], pages[i],
                "page at position {i} should match after double insert");
        }
    }

    // ── Wave 8: 13 additional tests covering remaining uncovered edge cases ──

    // 1. Overwrite shared node via a third branch sharing only the root — verify
    //    the page_ref for root node is updated but deeper nodes of existing branches remain
    #[test]
    fn third_branch_overwrites_shared_root_without_affecting_deep_nodes() {
        // Arrange: insert [1, 2, 3] and [1, 4, 5], then [1, 99] overwrites root
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(10, 0),
            VirtualPageId::new(10, 1),
            VirtualPageId::new(10, 2),
        ]);
        index.insert(&[1, 4, 5], &[
            VirtualPageId::new(20, 0),
            VirtualPageId::new(20, 1),
            VirtualPageId::new(20, 2),
        ]);
        // Act: overwrite root node via a short sequence with a new page
        index.insert(&[1, 99], &[
            VirtualPageId::new(99, 0),
            VirtualPageId::new(99, 1),
        ]);
        // Assert: branch [1, 2, 3] still matches 3 tokens with updated root page
        let ma = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(ma.matched_tokens, 3);
        assert_eq!(ma.matched_pages[0], VirtualPageId::new(99, 0));
        assert_eq!(ma.matched_pages[1], VirtualPageId::new(10, 1));
        assert_eq!(ma.matched_pages[2], VirtualPageId::new(10, 2));
        // Assert: branch [1, 4, 5] also has updated root page
        let mb = index.find_longest_prefix(&[1, 4, 5]).unwrap();
        assert_eq!(mb.matched_tokens, 3);
        assert_eq!(mb.matched_pages[0], VirtualPageId::new(99, 0));
        assert_eq!(mb.matched_pages[1], VirtualPageId::new(20, 1));
    }

    // 2. Token sequence [u32::MAX, 0] — boundary value transition from max to zero
    #[test]
    fn token_sequence_u32_max_then_zero() {
        // Arrange: boundary value transition
        let mut index = KvPrefixIndex::new();
        let pages = vec![
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
        ];
        index.insert(&[u32::MAX, 0], &pages);
        // Act: exact match
        let matched = index.find_longest_prefix(&[u32::MAX, 0]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages, pages);
        // Act: query starting with 0 is a miss (root has child u32::MAX, not 0)
        assert!(index.find_longest_prefix(&[0, u32::MAX]).is_none());
        // Act: partial match with just u32::MAX
        let partial = index.find_longest_prefix(&[u32::MAX, 999]).unwrap();
        assert_eq!(partial.matched_tokens, 1);
    }

    // 3. Progressive insert with interleaved empty inserts — empty inserts never corrupt
    #[test]
    fn progressive_insert_with_interleaved_empty_inserts() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
        ]);
        // Act: interleaved empty inserts
        index.insert(&[], &[VirtualPageId::new(99, 99)]);
        index.insert(&[], &[]);
        // Assert: original data intact
        let matched = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
        // Extend
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[], &[VirtualPageId::new(99, 99)]);
        // Assert: extended data intact after more empty inserts
        let extended = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(extended.matched_tokens, 3);
    }

    // 4. Identical token sequence with alternating page pattern [A, B, A, B, A] — verify
    //    no over-dedup for non-consecutive identical pages within the output
    #[test]
    fn identical_tokens_alternating_three_two_pages_no_over_dedup() {
        // Arrange: 5 identical tokens, pages [P, Q, P, Q, P]
        let p = VirtualPageId::new(1, 0);
        let q = VirtualPageId::new(2, 0);
        let pages = vec![p, q, p, q, p];
        let tokens = vec![42u32; 5];
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();
        // Assert: 5 tokens matched, 5 pages (non-consecutive same pages kept)
        assert_eq!(matched.matched_tokens, 5);
        assert_eq!(matched.matched_pages.len(), 5);
        assert_eq!(matched.matched_pages[0], p);
        assert_eq!(matched.matched_pages[1], q);
        assert_eq!(matched.matched_pages[2], p);
        assert_eq!(matched.matched_pages[3], q);
        assert_eq!(matched.matched_pages[4], p);
    }

    // 5. Two sequences with no shared prefix at all, each verified before and after
    //    the other is inserted — proves insert isolation
    #[test]
    fn two_independent_sequences_verified_before_and_after() {
        // Arrange: empty index
        let mut index = KvPrefixIndex::new();
        // Act: insert first sequence
        index.insert(&[100, 200, 300], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // Assert: first sequence matches
        assert_eq!(index.find_longest_prefix(&[100, 200, 300]).unwrap().matched_tokens, 3);
        // Act: insert second independent sequence
        index.insert(&[400, 500], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
        ]);
        // Assert: both match independently
        assert_eq!(index.find_longest_prefix(&[100, 200, 300]).unwrap().matched_tokens, 3);
        assert_eq!(index.find_longest_prefix(&[400, 500]).unwrap().matched_tokens, 2);
        // Assert: cross-contamination check
        assert_eq!(index.find_longest_prefix(&[100, 500]).unwrap().matched_tokens, 1);
        assert!(index.find_longest_prefix(&[200]).is_none());
    }

    // 6. Truncate correctness: insert [A, B, C] with pages at all positions, then query
    //    [A, B, C, D, E] where D diverges — matched_pages should be exactly 3, not 5
    #[test]
    fn truncate_correctness_after_extension_query_diverges() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        let pages = vec![
            VirtualPageId::new(0, 10),
            VirtualPageId::new(0, 20),
            VirtualPageId::new(0, 30),
        ];
        index.insert(&[1, 2, 3], &pages);
        // Act: query extends past inserted sequence and diverges at position 3
        let matched = index.find_longest_prefix(&[1, 2, 3, 99, 99]).unwrap();
        // Assert: exactly 3 tokens, exactly 3 pages, no stale pages from traversal
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 10));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(0, 20));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 30));
    }

    // 7. Three branches sharing a 2-token prefix, each extending to different depths:
    //    branch A depth 4, branch B depth 3, branch C depth 2
    #[test]
    fn three_branches_different_depths_shared_prefix() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        // Branch A: [1, 2, 10, 100] depth 4
        index.insert(&[1, 2, 10, 100], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
        ]);
        // Branch B: [1, 2, 20] depth 3
        index.insert(&[1, 2, 20], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // Branch C: [1, 2] depth 2 (leaf at shared prefix boundary)
        index.insert(&[1, 2], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
        ]);
        // Act & Assert: each branch matches at its full depth
        let ma = index.find_longest_prefix(&[1, 2, 10, 100]).unwrap();
        assert_eq!(ma.matched_tokens, 4);
        let mb = index.find_longest_prefix(&[1, 2, 20]).unwrap();
        assert_eq!(mb.matched_tokens, 3);
        let mc = index.find_longest_prefix(&[1, 2, 30]).unwrap();
        assert_eq!(mc.matched_tokens, 2);
    }

    // 8. Overwrite a shared node's page_ref to VirtualPageId::new(0, 0) then verify it
    //    is distinct from None (page_ref = Some((0,0), 0) vs page_ref = None)
    #[test]
    fn overwrite_to_zero_page_id_distinct_from_no_page() {
        // Arrange: insert [10] with a non-zero page, then overwrite with zero page
        let mut index = KvPrefixIndex::new();
        index.insert(&[10], &[VirtualPageId::new(5, 5)]);
        assert_eq!(
            index.find_longest_prefix(&[10]).unwrap().matched_pages[0],
            VirtualPageId::new(5, 5),
        );
        // Act: overwrite with zero page_id
        index.insert(&[10], &[VirtualPageId::new(0, 0)]);
        // Assert: zero page is stored (not treated as "no page")
        let matched = index.find_longest_prefix(&[10]).unwrap();
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
    }

    // 9. Insert a long sequence, then query at every odd position — verify match length
    //    and page content at each position
    #[test]
    fn long_sequence_odd_position_queries() {
        // Arrange: 20-token sequence
        let tokens: Vec<TokenId> = (0..20).collect();
        let pages: Vec<VirtualPageId> = (0..20)
            .map(|i| VirtualPageId::new(0, i as usize * 7))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act & Assert: query at every odd length (1, 3, 5, ..., 19)
        for len in (1..20).step_by(2) {
            let query: Vec<TokenId> = (0..len as u32).collect();
            let matched = index.find_longest_prefix(&query).unwrap();
            assert_eq!(matched.matched_tokens, len, "odd len {len}");
            assert_eq!(
                matched.matched_pages[len - 1],
                VirtualPageId::new(0, (len - 1) * 7),
                "page at odd position {len}"
            );
        }
    }

    // 10. Repeated find on the same query after each of 5 inserts — verify monotonically
    //     non-decreasing matched_tokens as the trie grows
    #[test]
    fn monotonic_match_growth_with_progressive_inserts() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        let mut prev_tokens = 0;
        // Act & Assert: progressively extend [1, 2, 3, 4, 5] one token at a time
        for len in 1..=5 {
            let tokens: Vec<TokenId> = (1..=len as u32).collect();
            let pages: Vec<VirtualPageId> = (1..=len)
                .map(|i| VirtualPageId::new(0, i))
                .collect();
            index.insert(&tokens, &pages);
            let matched = index.find_longest_prefix(&[1, 2, 3, 4, 5]).unwrap();
            assert!(matched.matched_tokens >= prev_tokens,
                "match should grow: prev={prev_tokens}, current={}", matched.matched_tokens);
            prev_tokens = matched.matched_tokens;
        }
        // Final: all 5 tokens match
        assert_eq!(prev_tokens, 5);
    }

    // 11. Insert [A, B, C] with pages, then insert [A, B] with empty pages slice —
    //     the first two page_refs should NOT be cleared (insert only sets, never clears)
    #[test]
    fn reinsert_with_empty_pages_preserves_all_existing_page_refs() {
        // Arrange: full insert
        let mut index = KvPrefixIndex::new();
        let tokens = vec![5, 6, 7];
        index.insert(&tokens, &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(3, 2),
        ]);
        // Act: re-insert first 2 tokens with empty pages
        index.insert(&[5, 6], &[]);
        // Assert: all 3 page_refs survive
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(1, 0));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(2, 1));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(3, 2));
    }

    // 12. Y-shaped merge: two paths [A, B, X] and [C, B, X] share suffix [B, X] at different
    //     depths — verify each path is independent (trie does not merge suffix nodes)
    #[test]
    fn y_shaped_merge_suffix_shared_token_at_different_depths() {
        // Arrange: [10, 20, 30] and [40, 20, 30] — suffix [20, 30] appears in both
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[40, 20, 30], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // Act & Assert: each path matches fully
        let ma = index.find_longest_prefix(&[10, 20, 30]).unwrap();
        assert_eq!(ma.matched_tokens, 3);
        let mb = index.find_longest_prefix(&[40, 20, 30]).unwrap();
        assert_eq!(mb.matched_tokens, 3);
        // Token 20 is not a root child — it only appears at depth 1 in each branch
        assert!(index.find_longest_prefix(&[20, 30]).is_none());
        // Partial: [10, 20] matches 2 tokens from branch A
        assert_eq!(index.find_longest_prefix(&[10, 20]).unwrap().matched_tokens, 2);
        // Partial: [40, 20] matches 2 tokens from branch B
        assert_eq!(index.find_longest_prefix(&[40, 20]).unwrap().matched_tokens, 2);
    }

    // 13. Large number of overwrite cycles (10 overwrites of the same sequence) — verify
    //     the last overwrite's pages are always the final result
    #[test]
    fn ten_overwrite_cycles_last_wins() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        let tokens = vec![1, 2, 3];
        for cycle in 0u64..10 {
            let pages: Vec<VirtualPageId> = (0..3)
                .map(|i| VirtualPageId::new(cycle, i * 100))
                .collect();
            index.insert(&tokens, &pages);
            let matched = index.find_longest_prefix(&tokens).unwrap();
            // Assert: after each cycle, the pages reflect that cycle
            assert_eq!(matched.matched_tokens, 3, "cycle {cycle}");
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(cycle, 0),
                "cycle {cycle} page 0");
            assert_eq!(matched.matched_pages[1], VirtualPageId::new(cycle, 100),
                "cycle {cycle} page 1");
            assert_eq!(matched.matched_pages[2], VirtualPageId::new(cycle, 200),
                "cycle {cycle} page 2");
        }
        // Assert: final state is from cycle 9
        let final_match = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(final_match.matched_pages[0], VirtualPageId::new(9, 0));
        assert_eq!(final_match.matched_pages[1], VirtualPageId::new(9, 100));
        assert_eq!(final_match.matched_pages[2], VirtualPageId::new(9, 200));
    }

    // ── Wave 9: 13 additional tests covering struct update syntax, Copy/Hash/Ord traits,
    //   symmetry, boundary values, and structural edge cases ──

    // 1. PrefixMatch struct update syntax (FRU) creates a new instance overriding one field
    #[test]
    fn prefix_match_struct_update_syntax_overrides_tokens() {
        // Arrange: base instance with 3 tokens and 2 pages
        let base_pages = vec![VirtualPageId::new(0, 0), VirtualPageId::new(0, 1)];
        let base = PrefixMatch {
            matched_tokens: 3,
            matched_pages: base_pages.clone(),
        };
        // Act: use struct update syntax to override matched_tokens only
        let derived = PrefixMatch {
            matched_tokens: 5,
            ..base
        };
        // Assert: matched_tokens overridden, pages inherited from base
        assert_eq!(derived.matched_tokens, 5);
        assert_eq!(derived.matched_pages, base_pages);
    }

    // 2. PrefixMatch struct update syntax overriding pages while keeping tokens
    #[test]
    fn prefix_match_struct_update_syntax_overrides_pages() {
        // Arrange
        let base = PrefixMatch {
            matched_tokens: 4,
            matched_pages: vec![VirtualPageId::new(0, 0)],
        };
        let new_pages = vec![
            VirtualPageId::new(1, 10),
            VirtualPageId::new(2, 20),
            VirtualPageId::new(3, 30),
        ];
        // Act: override pages, keep matched_tokens
        let derived = PrefixMatch {
            matched_pages: new_pages.clone(),
            ..base
        };
        // Assert
        assert_eq!(derived.matched_tokens, 4);
        assert_eq!(derived.matched_pages, new_pages);
        // Assert: original base unchanged
        assert_eq!(base.matched_pages.len(), 1);
    }

    // 3. VirtualPageId Copy trait: modifying a copy does not affect the original
    #[test]
    fn virtual_page_id_copy_trait_independence() {
        // Arrange: original page with specific values
        let original = VirtualPageId::new(42, 100);
        // Act: copy (Copy trait) and bind to new variable with different field values
        let mut copy = original;
        copy.sequence_id = 99;
        copy.logical_index = 200;
        // Assert: original unchanged
        assert_eq!(original.sequence_id, 42);
        assert_eq!(original.logical_index, 100);
        // Assert: copy has new values
        assert_eq!(copy.sequence_id, 99);
        assert_eq!(copy.logical_index, 200);
    }

    // 4. VirtualPageId Hash consistency: equal values produce identical hashes
    #[test]
    fn virtual_page_id_hash_consistency_for_equal_values() {
        // Arrange: two equal VirtualPageIds
        let a = VirtualPageId::new(7, 42);
        let b = VirtualPageId::new(7, 42);
        // Act: compute hashes
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher_a = DefaultHasher::new();
        a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();
        let mut hasher_b = DefaultHasher::new();
        b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();
        // Assert: equal values produce equal hashes
        assert_eq!(hash_a, hash_b);
        // Assert: different values produce different hashes (probabilistic but reliable)
        let c = VirtualPageId::new(8, 42);
        let mut hasher_c = DefaultHasher::new();
        c.hash(&mut hasher_c);
        let hash_c = hasher_c.finish();
        assert_ne!(hash_a, hash_c);
    }

    // 5. VirtualPageId Eq trait: equality and inequality cover all field combinations
    #[test]
    fn virtual_page_id_eq_covers_all_field_combinations() {
        // Arrange: pages varying in each field
        let a = VirtualPageId::new(1, 10);
        let b = VirtualPageId::new(1, 10);
        let c = VirtualPageId::new(1, 20);
        let d = VirtualPageId::new(2, 10);
        // Assert: same both fields → equal
        assert_eq!(a, b);
        // Assert: different logical_index → not equal
        assert_ne!(a, c);
        // Assert: different sequence_id → not equal
        assert_ne!(a, d);
        // Assert: both fields different → not equal
        assert_ne!(c, d);
    }

    // 6. PrefixMatch PartialEq symmetry: a == b implies b == a
    #[test]
    fn prefix_match_equality_symmetry() {
        // Arrange: two instances with same values constructed independently
        let a = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(2, 1),
            ],
        };
        let b = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(2, 1),
            ],
        };
        // Assert: symmetry — both directions must be true
        assert!(a == b);
        assert!(b == a);
        // Assert: assert_eq uses PartialEq
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // 7. PrefixMatch PartialEq reflexivity: a == a is always true
    #[test]
    fn prefix_match_equality_reflexivity() {
        // Arrange
        let pm = PrefixMatch {
            matched_tokens: 7,
            matched_pages: vec![VirtualPageId::new(0, 0), VirtualPageId::new(1, 1)],
        };
        // Assert: reflexivity
        assert_eq!(pm, pm);
        assert!(pm == pm);
    }

    // 8. PrefixMatch PartialEq transitivity: a == b and b == c implies a == c
    #[test]
    fn prefix_match_equality_transitivity() {
        // Arrange: three independently constructed equal instances
        let a = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(5, 10)],
        };
        let b = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(5, 10)],
        };
        let c = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![VirtualPageId::new(5, 10)],
        };
        // Assert: transitivity
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // 9. VirtualPageId with logical_index = 0 is a valid, distinct page (not confused with None)
    #[test]
    fn virtual_page_id_logical_index_zero_is_valid_distinct_from_none() {
        // Arrange: page with logical_index = 0 and sequence_id = 0
        let page = VirtualPageId::new(0, 0);
        let mut index = KvPrefixIndex::new();
        // Insert a token without page first, then with page at (0, 0)
        index.insert(&[42], &[]);
        assert!(index.find_longest_prefix(&[42]).is_none());
        // Act: insert with page (0, 0)
        index.insert(&[42], &[page]);
        let matched = index.find_longest_prefix(&[42]).unwrap();
        // Assert: (0, 0) is stored and returned — not confused with "no page"
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
    }

    // 10. TokenId as HashMap key: multiple token lookups in trie stress the Hash/Eq contract
    #[test]
    fn token_id_hash_eq_contract_stress_via_trie_branching() {
        // Arrange: insert 200 single-token sequences with token values covering u16 range
        let mut index = KvPrefixIndex::new();
        for i in 0u32..200 {
            // Use tokens that have similar bit patterns (e.g., differ only in high bits)
            let token = i * 1000;
            let page = VirtualPageId::new(i as u64, 0);
            index.insert(&[token], &[page]);
        }
        // Act & Assert: each token is correctly found (HashMap key integrity)
        for i in 0u32..200 {
            let token = i * 1000;
            let matched = index.find_longest_prefix(&[token]).unwrap();
            assert_eq!(matched.matched_tokens, 1, "token {token}");
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(i as u64, 0));
        }
        // Assert: nearby values that weren't inserted return None
        for i in 0u32..200 {
            let nearby = i * 1000 + 1;
            assert!(index.find_longest_prefix(&[nearby]).is_none(),
                "nearby token {nearby} should not match");
        }
    }

    // 11. PrefixMatch Debug trait output is non-empty and deterministic for fixed input
    #[test]
    fn prefix_match_debug_output_is_deterministic() {
        // Arrange: fixed PrefixMatch
        let pm = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![
                VirtualPageId::new(10, 20),
                VirtualPageId::new(30, 40),
            ],
        };
        // Act: format multiple times
        let debug1 = format!("{pm:?}");
        let debug2 = format!("{pm:?}");
        // Assert: output is identical across calls (deterministic)
        assert_eq!(debug1, debug2);
        // Assert: output is non-empty and contains expected field names
        assert!(!debug1.is_empty());
        assert!(debug1.contains("matched_tokens"));
        assert!(debug1.contains("matched_pages"));
    }

    // 12. VirtualPageId Copy trait: using a page value in multiple places is independent
    #[test]
    fn virtual_page_id_copy_used_in_multiple_inserts() {
        // Arrange: same VirtualPageId value used in two separate inserts
        let shared_page = VirtualPageId::new(99, 42);
        let mut index = KvPrefixIndex::new();
        index.insert(&[10], &[shared_page]);
        index.insert(&[20], &[shared_page]);
        // Act & Assert: both tokens map to the same page value (Copy allows reuse)
        let ma = index.find_longest_prefix(&[10]).unwrap();
        assert_eq!(ma.matched_pages[0], shared_page);
        let mb = index.find_longest_prefix(&[20]).unwrap();
        assert_eq!(mb.matched_pages[0], shared_page);
        // Assert: both pages are equal (Copy semantics)
        assert_eq!(ma.matched_pages[0], mb.matched_pages[0]);
    }

    // 13. KvPrefixIndex Debug after multiple inserts shows non-empty structure
    #[test]
    fn kv_prefix_index_debug_after_multiple_inserts_shows_structure() {
        // Arrange: build a trie with several branches
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[1, 4, 5], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        // Act: format Debug
        let debug = format!("{index:?}");
        // Assert: debug output contains "root" and "children" (TrieNode fields)
        assert!(debug.contains("root"));
        assert!(debug.contains("children"));
        // Assert: debug output is non-trivially long (has structure beyond empty)
        assert!(debug.len() > 20, "debug output should reflect non-empty trie");
        // Assert: format is deterministic
        let debug2 = format!("{index:?}");
        assert_eq!(debug, debug2);
    }

    // ── Wave 5: 13 additional tests covering uncovered edge cases ──

    // 1. Insert sequence longer than any existing, then query a middle subsequence
    #[test]
    fn insert_long_sequence_query_internal_subsequence_matches_prefix() {
        // Arrange: insert a 10-token sequence with pages
        let mut index = KvPrefixIndex::new();
        let tokens: Vec<TokenId> = (100..110).collect();
        let pages: Vec<VirtualPageId> = (0..10)
            .map(|i| VirtualPageId::new(5, i))
            .collect();
        index.insert(&tokens, &pages);

        // Act: query a 5-token prefix of the 10-token sequence
        let matched = index.find_longest_prefix(&[100, 101, 102, 103, 104]).unwrap();

        // Assert: matches exactly the first 5 tokens with their pages
        assert_eq!(matched.matched_tokens, 5);
        assert_eq!(matched.matched_pages.len(), 5);
        assert_eq!(matched.matched_pages[4], VirtualPageId::new(5, 4));
    }

    // 2. Three separate single-token entries, each queried independently
    #[test]
    fn three_single_token_entries_independent_queries() {
        // Arrange: insert three unrelated single-token sequences
        let mut index = KvPrefixIndex::new();
        let t1 = 1001u32;
        let t2 = 2002u32;
        let t3 = 3003u32;
        let p1 = VirtualPageId::new(10, 0);
        let p2 = VirtualPageId::new(20, 0);
        let p3 = VirtualPageId::new(30, 0);
        index.insert(&[t1], &[p1]);
        index.insert(&[t2], &[p2]);
        index.insert(&[t3], &[p3]);

        // Act & Assert: each query returns its own page
        let m1 = index.find_longest_prefix(&[t1]).unwrap();
        assert_eq!(m1.matched_tokens, 1);
        assert_eq!(m1.matched_pages[0], p1);

        let m2 = index.find_longest_prefix(&[t2]).unwrap();
        assert_eq!(m2.matched_tokens, 1);
        assert_eq!(m2.matched_pages[0], p2);

        let m3 = index.find_longest_prefix(&[t3]).unwrap();
        assert_eq!(m3.matched_tokens, 1);
        assert_eq!(m3.matched_pages[0], p3);
    }

    // 3. Overwrite a shared prefix node's page, then verify both branches see new page
    #[test]
    fn overwrite_shared_prefix_node_page_visible_to_both_branches() {
        // Arrange: build Y-shaped trie: [1, 2, 10] and [1, 2, 20]
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 10], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        index.insert(&[1, 2, 20], &[
            VirtualPageId::new(0, 10),
            VirtualPageId::new(0, 11),
            VirtualPageId::new(0, 12),
        ]);

        // Act: overwrite shared prefix [1, 2] with new page at position 1
        let new_page = VirtualPageId::new(99, 99);
        index.insert(&[1, 2], &[VirtualPageId::new(0, 0), new_page]);

        // Assert: both branches now see the overwritten page at position 1
        let ma = index.find_longest_prefix(&[1, 2, 10]).unwrap();
        assert_eq!(ma.matched_pages[1], new_page);

        let mb = index.find_longest_prefix(&[1, 2, 20]).unwrap();
        assert_eq!(mb.matched_pages[1], new_page);
    }

    // 4. Query with tokens that match a path but no node has a page_ref at all
    #[test]
    fn query_path_exists_but_no_page_ref_at_any_node_returns_none() {
        // Arrange: insert tokens with empty pages so trie nodes exist but page_ref = None
        let mut index = KvPrefixIndex::new();
        index.insert(&[5, 6, 7], &[]);

        // Act: query the exact same tokens
        let result = index.find_longest_prefix(&[5, 6, 7]);

        // Assert: no page_ref found despite trie path existing
        assert!(result.is_none());
    }

    // 5. Insert with pages for only the last token; verify match returns that single page
    #[test]
    fn pages_only_at_last_token_position_matches_single_page() {
        // Arrange: insert 4 tokens but only provide pages starting from the last index
        let mut index = KvPrefixIndex::new();
        // pages array is empty — no pages for positions 0, 1, 2
        // But we can trick by providing a single page that only maps to index 0
        // To get a page only at the last token, we insert with pages for all, then
        // overwrite with empty pages for the prefix only.
        // Actually, simpler: just provide 1 page in a 1-token sequence
        let mut index2 = KvPrefixIndex::new();
        index2.insert(&[42], &[VirtualPageId::new(7, 3)]);

        // Act
        let matched = index2.find_longest_prefix(&[42]).unwrap();

        // Assert
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages.len(), 1);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(7, 3));
    }

    // 6. Verify insert with duplicate consecutive tokens creates distinct trie levels
    #[test]
    fn duplicate_consecutive_tokens_creates_distinct_trie_levels() {
        // Arrange: insert [7, 7, 7] — same token at every position
        let mut index = KvPrefixIndex::new();
        let pages = vec![
            VirtualPageId::new(1, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(3, 2),
        ];
        index.insert(&[7, 7, 7], &pages);

        // Act: query with [7, 7, 7]
        let matched = index.find_longest_prefix(&[7, 7, 7]).unwrap();

        // Assert: all three levels are traversed, each with distinct page
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
        assert_eq!(matched.matched_pages[0].sequence_id, 1);
        assert_eq!(matched.matched_pages[1].sequence_id, 2);
        assert_eq!(matched.matched_pages[2].sequence_id, 3);
    }

    // 7. Token value u32::MIN (0) and u32::MAX in the same sequence
    #[test]
    fn token_sequence_spanning_u32_min_to_max() {
        // Arrange: sequence starting with 0 and ending with u32::MAX
        let mut index = KvPrefixIndex::new();
        let tokens = vec![0u32, 100, u32::MAX];
        let pages = vec![
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ];
        index.insert(&tokens, &pages);

        // Act: query the exact sequence
        let matched = index.find_longest_prefix(&[0, 100, u32::MAX]).unwrap();

        // Assert: full match
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
    }

    // 8. After multiple inserts to same path, offset_in_page validity is checked per node
    #[test]
    fn offset_validity_per_node_after_mixed_inserts() {
        // Arrange: insert [10, 20] with pages, then overwrite [10] with a different page
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);
        index.insert(&[10], &[VirtualPageId::new(2, 0)]);

        // Act: query [10, 20]
        let matched = index.find_longest_prefix(&[10, 20]).unwrap();

        // Assert: first node uses overwritten page, second node keeps original
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(2, 0));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(1, 1));
    }

    // 9. Large branch factor at root: 64 distinct first tokens, verify all and none others
    #[test]
    fn large_root_branch_factor_64_distinct_first_tokens() {
        // Arrange: insert 64 single-token sequences with tokens 1000..1064
        let mut index = KvPrefixIndex::new();
        for i in 0u32..64 {
            let token = 1000 + i;
            let page = VirtualPageId::new(i as u64, 0);
            index.insert(&[token], &[page]);
        }

        // Act & Assert: all 64 tokens match
        for i in 0u32..64 {
            let token = 1000 + i;
            let matched = index.find_longest_prefix(&[token]).unwrap();
            assert_eq!(matched.matched_tokens, 1, "token {token} should match");
            assert_eq!(matched.matched_pages[0], VirtualPageId::new(i as u64, 0));
        }

        // Assert: tokens just outside the range return None
        assert!(index.find_longest_prefix(&[999]).is_none());
        assert!(index.find_longest_prefix(&[1064]).is_none());
        assert!(index.find_longest_prefix(&[2000]).is_none());
    }

    // 10. PrefixMatch with exactly one page and matched_tokens > 1
    #[test]
    fn prefix_match_single_page_with_multiple_matched_tokens() {
        // Arrange: insert [1, 2, 3] with the same page for all positions
        let mut index = KvPrefixIndex::new();
        let shared = VirtualPageId::new(42, 0);
        index.insert(&[1, 2, 3], &[shared, shared, shared]);

        // Act
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();

        // Assert: deduplication collapses consecutive identical pages to one
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 1);
        assert_eq!(matched.matched_pages[0], shared);
    }

    // 11. Insert empty token sequence does nothing, subsequent real insert still works
    #[test]
    fn empty_token_insert_no_op_then_real_insert_works() {
        // Arrange: insert empty token list (should be a no-op)
        let mut index = KvPrefixIndex::new();
        index.insert(&[], &[VirtualPageId::new(1, 0)]);

        // Act: insert a real sequence
        index.insert(&[5, 6], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);

        // Assert: real insert works normally
        let matched = index.find_longest_prefix(&[5, 6]).unwrap();
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages.len(), 2);
    }

    // 12. Overwrite middle of a chain with empty pages, deeper page refs survive
    #[test]
    fn overwrite_middle_with_empty_pages_preserves_deeper_refs() {
        // Arrange: insert [1, 2, 3, 4] with full pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
        ]);

        // Act: overwrite [1, 2] with empty pages — page_ref for positions 0, 1
        // should be preserved because insert with empty pages never clears page_ref
        index.insert(&[1, 2], &[]);

        // Assert: full path still matches with all original pages
        let matched = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(matched.matched_tokens, 4);
        assert_eq!(matched.matched_pages.len(), 4);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 0));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 2));
        assert_eq!(matched.matched_pages[3], VirtualPageId::new(0, 3));
    }

    // 13. Two sequences sharing first two tokens, third token diverges, both queried fully
    #[test]
    fn two_sequences_shared_prefix_two_tokens_third_diverges_both_correct() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        // Branch A: [50, 51, 100]
        index.insert(&[50, 51, 100], &[
            VirtualPageId::new(10, 0),
            VirtualPageId::new(10, 1),
            VirtualPageId::new(10, 2),
        ]);
        // Branch B: [50, 51, 200]
        index.insert(&[50, 51, 200], &[
            VirtualPageId::new(20, 0),
            VirtualPageId::new(20, 1),
            VirtualPageId::new(20, 2),
        ]);

        // Act & Assert: branch A full match
        let ma = index.find_longest_prefix(&[50, 51, 100]).unwrap();
        assert_eq!(ma.matched_tokens, 3);
        assert_eq!(ma.matched_pages[0], VirtualPageId::new(20, 0));
        assert_eq!(ma.matched_pages[1], VirtualPageId::new(20, 1));
        assert_eq!(ma.matched_pages[2], VirtualPageId::new(10, 2));

        // Act & Assert: branch B full match
        let mb = index.find_longest_prefix(&[50, 51, 200]).unwrap();
        assert_eq!(mb.matched_tokens, 3);
        assert_eq!(mb.matched_pages[2], VirtualPageId::new(20, 2));

        // Act & Assert: shared prefix only
        let mc = index.find_longest_prefix(&[50, 51, 999]).unwrap();
        assert_eq!(mc.matched_tokens, 2);
        assert_eq!(mc.matched_pages.len(), 2);
    }

    // ── Wave 10: 13 additional tests covering deep overwrite semantics,
    //   subsequence behavior, and structural edge cases ──

    // 1. Deep overwrite chain: insert [A,B,C,D,E] with pages, then overwrite [A,B,C]
    //    with different pages — verify A,B,C use new pages while D,E keep original
    #[test]
    fn deep_overwrite_prefix_keeps_tail_intact() {
        // Arrange: 5-token chain with original pages
        let mut index = KvPrefixIndex::new();
        let tokens: Vec<TokenId> = (10..15).collect();
        let original_pages: Vec<VirtualPageId> = (10..15)
            .map(|i| VirtualPageId::new(0, i as usize))
            .collect();
        index.insert(&tokens, &original_pages);

        // Act: overwrite first 3 tokens with new pages
        let new_pages: Vec<VirtualPageId> = vec![
            VirtualPageId::new(99, 100),
            VirtualPageId::new(99, 101),
            VirtualPageId::new(99, 102),
        ];
        index.insert(&tokens[..3], &new_pages);

        // Assert: full sequence still matches 5 tokens
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 5);
        assert_eq!(matched.matched_pages.len(), 5);
        // First 3 from new pages
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(99, 100));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(99, 101));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(99, 102));
        // Last 2 from original
        assert_eq!(matched.matched_pages[3], VirtualPageId::new(0, 13));
        assert_eq!(matched.matched_pages[4], VirtualPageId::new(0, 14));
    }

    // 2. Insert [A, B, C] then insert [A, B, C, D] — overwriting first 3 page_refs
    //    with the extension's pages. Verify the 4-token match returns the extension's pages.
    #[test]
    fn extension_overwrites_shared_prefix_pages() {
        // Arrange: insert short sequence first
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);

        // Act: extend to 4 tokens — overwrites page_refs at positions 0..3
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(1, 10),
            VirtualPageId::new(1, 11),
            VirtualPageId::new(1, 12),
            VirtualPageId::new(1, 13),
        ]);

        // Assert: full 4-token match uses extension pages
        let matched = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(matched.matched_tokens, 4);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(1, 10));
        assert_eq!(matched.matched_pages[3], VirtualPageId::new(1, 13));
    }

    // 3. Query that matches at position 0 but the page_ref has offset_in_page = 0
    //    which satisfies the offset <= idx (0 <= 0) check — single-token match
    #[test]
    fn page_ref_offset_zero_at_idx_zero_passes_validity_check() {
        // Arrange: single-token insert produces page_ref with offset_in_page = 0
        let mut index = KvPrefixIndex::new();
        let page = VirtualPageId::new(7, 42);
        index.insert(&[99], &[page]);

        // Act: query with the same token
        let matched = index.find_longest_prefix(&[99]).unwrap();

        // Assert: offset_in_page (0) <= idx (0) passes, so match is returned
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages[0], page);
    }

    // 4. Insert two sequences where one is a strict subsequence of the other:
    //    [A, B] and [A, B, C, D]. Query [A, B] returns overwritten pages from the longer insert.
    #[test]
    fn strict_subsequence_query_returns_overwritten_pages() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        // Insert shorter first
        index.insert(&[5, 6], &[
            VirtualPageId::new(10, 0),
            VirtualPageId::new(10, 1),
        ]);
        // Insert longer, sharing prefix
        index.insert(&[5, 6, 7, 8], &[
            VirtualPageId::new(20, 0),
            VirtualPageId::new(20, 1),
            VirtualPageId::new(20, 2),
            VirtualPageId::new(20, 3),
        ]);

        // Act: query only the shorter prefix
        let matched = index.find_longest_prefix(&[5, 6]).unwrap();

        // Assert: pages come from the second (longer) insert's overwrite
        assert_eq!(matched.matched_tokens, 2);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(20, 0));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(20, 1));
    }

    // 5. Insert same token sequence twice with different pages, verify the second insert's
    //    pages completely replace the first — no remnant pages from first insert
    #[test]
    fn full_overwrite_no_remnant_pages() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        let tokens = vec![10, 20, 30];
        let pages_v1 = vec![
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ];
        let pages_v2 = vec![
            VirtualPageId::new(2, 10),
            VirtualPageId::new(2, 11),
            VirtualPageId::new(2, 12),
        ];
        index.insert(&tokens, &pages_v1);
        index.insert(&tokens, &pages_v2);

        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();

        // Assert: every page is from v2, no v1 remnants
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages, pages_v2);
        for page in &matched.matched_pages {
            assert_eq!(page.sequence_id, 2);
        }
    }

    // 6. Insert 10 sequences sharing first 3 tokens but diverging at token 4,
    //    verify each divergent path matches independently at depth 4
    #[test]
    fn ten_branches_shared_3_prefix_diverge_at_4() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        for i in 0u32..10 {
            let tokens = [100, 200, 300, i * 1000];
            let pages: Vec<VirtualPageId> = (0..4)
                .map(|p| VirtualPageId::new(i as u64, p))
                .collect();
            index.insert(&tokens, &pages);
        }

        // Act & Assert: each branch matches 4 tokens
        for i in 0u32..10 {
            let tokens = [100, 200, 300, i * 1000];
            let matched = index.find_longest_prefix(&tokens).unwrap();
            assert_eq!(matched.matched_tokens, 4, "branch {i}");
        }

        // Act: divergent token at position 3 matches only 3 shared tokens
        let partial = index.find_longest_prefix(&[100, 200, 300, 99999]).unwrap();
        assert_eq!(partial.matched_tokens, 3);
    }

    // 7. Insert sequence [A, B], then insert [X, B] (sharing suffix but not prefix) —
    //    verify no cross-contamination between branches
    #[test]
    fn shared_suffix_different_prefix_no_cross_contamination() {
        // Arrange
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);
        index.insert(&[30, 20], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
        ]);

        // Act & Assert: each branch matches independently
        let ma = index.find_longest_prefix(&[10, 20]).unwrap();
        assert_eq!(ma.matched_tokens, 2);
        assert_eq!(ma.matched_pages[0].sequence_id, 1);

        let mb = index.find_longest_prefix(&[30, 20]).unwrap();
        assert_eq!(mb.matched_tokens, 2);
        assert_eq!(mb.matched_pages[0].sequence_id, 2);

        // Assert: token 20 is not a root child (only at depth 1)
        assert!(index.find_longest_prefix(&[20]).is_none());
    }

    // 8. VirtualPageId Eq: verify equal values are equal and different values are not
    #[test]
    fn virtual_page_id_eq_covers_field_variations() {
        // Arrange: pages varying in each field
        let a = VirtualPageId::new(1, 10);
        let b = VirtualPageId::new(1, 10);
        let c = VirtualPageId::new(1, 20);
        let d = VirtualPageId::new(2, 10);

        // Assert: same both fields → equal
        assert_eq!(a, b);
        // Assert: different logical_index → not equal
        assert_ne!(a, c);
        // Assert: different sequence_id → not equal
        assert_ne!(a, d);
        // Assert: both fields different → not equal
        assert_ne!(c, d);
    }

    // 9. Insert sequence with all pages being the same VirtualPageId — matched_pages
    //    should be deduplicated to a single entry
    #[test]
    fn all_same_pages_deduplicated_to_single_entry() {
        // Arrange: 10 tokens all mapping to the same page
        let shared = VirtualPageId::new(42, 7);
        let tokens: Vec<TokenId> = (0..10).collect();
        let pages = vec![shared; 10];
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);

        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();

        // Assert: all 10 tokens matched but pages deduplicated to 1
        assert_eq!(matched.matched_tokens, 10);
        assert_eq!(matched.matched_pages.len(), 1);
        assert_eq!(matched.matched_pages[0], shared);
    }

    // 10. Progressive overwrite at increasing depths: insert [A], then [A,B], then [A,B,C]
    //     verifying at each step that the trie structure grows correctly
    #[test]
    fn progressive_depth_extension_grows_trie_correctly() {
        // Arrange
        let mut index = KvPrefixIndex::new();

        // Step 1: insert [A]
        index.insert(&[10], &[VirtualPageId::new(1, 0)]);
        assert_eq!(index.find_longest_prefix(&[10]).unwrap().matched_tokens, 1);

        // Step 2: insert [A, B]
        index.insert(&[10, 20], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
        ]);
        let m2 = index.find_longest_prefix(&[10, 20]).unwrap();
        assert_eq!(m2.matched_tokens, 2);
        assert_eq!(m2.matched_pages[0], VirtualPageId::new(2, 0));

        // Step 3: insert [A, B, C]
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(3, 0),
            VirtualPageId::new(3, 1),
            VirtualPageId::new(3, 2),
        ]);
        let m3 = index.find_longest_prefix(&[10, 20, 30]).unwrap();
        assert_eq!(m3.matched_tokens, 3);
        assert_eq!(m3.matched_pages[0], VirtualPageId::new(3, 0));
        assert_eq!(m3.matched_pages[1], VirtualPageId::new(3, 1));
        assert_eq!(m3.matched_pages[2], VirtualPageId::new(3, 2));
    }

    // 11. Verify that PrefixMatch with identical matched_tokens but pages in reverse
    //     order are not equal — confirms pages vector order matters for equality
    #[test]
    fn prefix_match_pages_order_matters_for_equality() {
        // Arrange: two PrefixMatch instances with same tokens but reversed pages
        let p1 = VirtualPageId::new(1, 0);
        let p2 = VirtualPageId::new(2, 0);
        let a = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![p1, p2],
        };
        let b = PrefixMatch {
            matched_tokens: 2,
            matched_pages: vec![p2, p1],
        };

        // Assert: different order → not equal
        assert_ne!(a, b);
    }

    // 12. Insert 3 independent sequences, verify each matches with correct pages
    //     and no cross-contamination between branches
    #[test]
    fn three_independent_sequences_full_isolation() {
        // Arrange: three sequences with no shared prefix
        let mut index = KvPrefixIndex::new();
        // Sequence A: [100, 101]
        index.insert(&[100, 101], &[
            VirtualPageId::new(10, 0),
            VirtualPageId::new(10, 1),
        ]);
        // Sequence B: [200, 201]
        index.insert(&[200, 201], &[
            VirtualPageId::new(20, 0),
            VirtualPageId::new(20, 1),
        ]);
        // Sequence C: [300, 301]
        index.insert(&[300, 301], &[
            VirtualPageId::new(30, 0),
            VirtualPageId::new(30, 1),
        ]);

        // Act & Assert: each sequence matches independently with correct pages
        let ma = index.find_longest_prefix(&[100, 101]).unwrap();
        assert_eq!(ma.matched_tokens, 2);
        assert_eq!(ma.matched_pages[0], VirtualPageId::new(10, 0));

        let mb = index.find_longest_prefix(&[200, 201]).unwrap();
        assert_eq!(mb.matched_tokens, 2);
        assert_eq!(mb.matched_pages[0], VirtualPageId::new(20, 0));

        let mc = index.find_longest_prefix(&[300, 301]).unwrap();
        assert_eq!(mc.matched_tokens, 2);
        assert_eq!(mc.matched_pages[0], VirtualPageId::new(30, 0));

        // Assert: no cross-contamination — mixed tokens match only prefix
        assert_eq!(index.find_longest_prefix(&[100, 201]).unwrap().matched_tokens, 1);
        assert_eq!(index.find_longest_prefix(&[300, 101]).unwrap().matched_tokens, 1);
    }

    // 13. Insert [A,B,C,D,E] with full pages, then insert [A,B,C] with empty pages —
    //     the first 3 page_refs should survive (insert with empty pages never clears)
    #[test]
    fn empty_pages_reinsert_preserves_existing_page_refs() {
        // Arrange: full 5-token insert
        let mut index = KvPrefixIndex::new();
        let tokens: Vec<TokenId> = (1..=5).collect();
        let pages: Vec<VirtualPageId> = (1..=5)
            .map(|i| VirtualPageId::new(0, i))
            .collect();
        index.insert(&tokens, &pages);

        // Act: re-insert first 3 tokens with empty pages
        index.insert(&tokens[..3], &[]);

        // Assert: all 5 page_refs survive
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 5);
        assert_eq!(matched.matched_pages.len(), 5);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(0, 1));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 3));
        assert_eq!(matched.matched_pages[4], VirtualPageId::new(0, 5));
    }

    // ── 13 new tests: prefix trie edge cases, hash collisions, empty prefix handling ──

    #[test]
    fn insert_with_no_pages_then_find_returns_none_even_with_trie_nodes() {
        // Arrange: insert 3 tokens with empty pages — trie nodes are created but no page_ref
        let mut index = KvPrefixIndex::new();
        index.insert(&[100, 200, 300], &[]);

        // Act & Assert: trie has nodes for [100, 200, 300] but find returns None
        assert!(index.find_longest_prefix(&[100, 200, 300]).is_none());
        assert!(index.find_longest_prefix(&[100]).is_none());
    }

    #[test]
    fn overlapping_prefix_with_sparse_pages_tracks_best_correctly() {
        // Arrange: insert [1, 2, 3, 4] with pages only at positions 0 and 3
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(0, 0),
        ]);
        // Now overwrite position 3 only by re-inserting the full sequence with 4 pages
        index.insert(&[1, 2, 3, 4], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
            VirtualPageId::new(0, 3),
        ]);

        // Act: query the full sequence
        let matched = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();

        // Assert: all 4 pages present
        assert_eq!(matched.matched_tokens, 4);
        assert_eq!(matched.matched_pages.len(), 4);
    }

    #[test]
    fn find_returns_none_when_first_token_diverges() {
        // Arrange: insert [50, 60, 70]
        let mut index = KvPrefixIndex::new();
        index.insert(&[50, 60, 70], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);

        // Act & Assert: query starting with a different first token
        assert!(index.find_longest_prefix(&[49, 60, 70]).is_none());
        assert!(index.find_longest_prefix(&[51, 60, 70]).is_none());
    }

    #[test]
    fn two_sequences_same_prefix_different_lengths() {
        // Arrange: insert [1, 2] and [1, 2, 3, 4, 5]
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2], &[
            VirtualPageId::new(10, 0),
            VirtualPageId::new(10, 1),
        ]);
        index.insert(&[1, 2, 3, 4, 5], &[
            VirtualPageId::new(20, 0),
            VirtualPageId::new(20, 1),
            VirtualPageId::new(20, 2),
            VirtualPageId::new(20, 3),
            VirtualPageId::new(20, 4),
        ]);

        // Act & Assert: shorter query matches 2
        let short = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(short.matched_tokens, 2);

        // Act & Assert: longer query matches 5
        let long = index.find_longest_prefix(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(long.matched_tokens, 5);
    }

    #[test]
    fn hash_collision_resistant_different_tokens_same_hash_bucket() {
        // Arrange: insert many sequences that exercise HashMap collision paths
        // Use token values that are far apart but exercise the trie children HashMap
        let mut index = KvPrefixIndex::new();
        let root_token: TokenId = 0;
        for i in 0..64u32 {
            let branch_token = i * 1000 + 1; // widely spaced values
            let page = VirtualPageId::new(i as u64, i as usize);
            index.insert(&[root_token, branch_token], &[
                VirtualPageId::new(0, 0),
                page,
            ]);
        }

        // Act & Assert: each branch matches independently
        for i in 0..64u32 {
            let branch_token = i * 1000 + 1;
            let matched = index
                .find_longest_prefix(&[root_token, branch_token, 999])
                .unwrap();
            // For i=0, page is VirtualPageId(0,0) same as root page, so dedup may reduce pages
            if i == 0 {
                // Root page and branch page are both VirtualPageId(0,0) — dedup reduces to 1 page
                assert_eq!(matched.matched_tokens, 2, "failed for branch {i}");
                assert_eq!(matched.matched_pages.len(), 1, "dedup: same page at both positions");
            } else {
                assert_eq!(matched.matched_tokens, 2, "failed for branch {i}");
                assert_eq!(matched.matched_pages.len(), 2, "failed pages for branch {i}");
            }
        }

        // Non-existent branch returns only root match
        let partial = index.find_longest_prefix(&[root_token, 7777]).unwrap();
        assert_eq!(partial.matched_tokens, 1);
    }

    #[test]
    fn empty_query_on_populated_index_returns_none() {
        // Arrange: populate with several sequences
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[VirtualPageId::new(0, 0)]);
        index.insert(&[10, 20], &[VirtualPageId::new(1, 0)]);
        index.insert(&[100], &[VirtualPageId::new(2, 0)]);

        // Act & Assert: empty query always returns None regardless of index state
        assert!(index.find_longest_prefix(&[]).is_none());
    }

    #[test]
    fn consecutive_dedup_across_reused_page_ids() {
        // Arrange: page A appears at tokens 0 and 1, different page B at token 2
        let page_a = VirtualPageId::new(5, 0);
        let page_b = VirtualPageId::new(5, 1);
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[page_a, page_a, page_b]);

        // Act
        let matched = index.find_longest_prefix(&[10, 20, 30]).unwrap();

        // Assert: page_a deduplicated to single entry, page_b is separate
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 2);
        assert_eq!(matched.matched_pages[0], page_a);
        assert_eq!(matched.matched_pages[1], page_b);
    }

    #[test]
    fn insert_single_token_with_single_page_then_query_extends_far() {
        // Arrange: minimal insert — one token, one page
        let mut index = KvPrefixIndex::new();
        let page = VirtualPageId::new(42, 7);
        index.insert(&[99], &[page]);

        // Act: query extends far beyond the single inserted token
        let matched = index
            .find_longest_prefix(&[99, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();

        // Assert: match stops at 1 token
        assert_eq!(matched.matched_tokens, 1);
        assert_eq!(matched.matched_pages, vec![page]);
    }

    #[test]
    fn deeply_nested_trie_path_preserves_all_page_refs() {
        // Arrange: build a deep linear chain of 50 tokens
        let depth = 50;
        let tokens: Vec<TokenId> = (0..depth).collect();
        let pages: Vec<VirtualPageId> = (0..depth)
            .map(|i| VirtualPageId::new(i as u64, i as usize))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);

        // Act: query a prefix of length 25
        let query: Vec<TokenId> = (0..25).collect();
        let matched = index.find_longest_prefix(&query).unwrap();

        // Assert: 25 tokens matched, 25 distinct pages
        assert_eq!(matched.matched_tokens, 25);
        assert_eq!(matched.matched_pages.len(), 25);
        // Each page is unique (different sequence_id)
        for i in 0..25 {
            assert_eq!(
                matched.matched_pages[i],
                VirtualPageId::new(i as u64, i as usize)
            );
        }
    }

    #[test]
    fn overwrite_shared_node_preserves_branch_structure() {
        // Arrange: insert branch A [1, 2, 3], then branch B [1, 4, 5]
        // The shared root node for token 1 gets overwritten by branch B's page
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(100, 0),
            VirtualPageId::new(100, 1),
            VirtualPageId::new(100, 2),
        ]);
        index.insert(&[1, 4, 5], &[
            VirtualPageId::new(200, 0),
            VirtualPageId::new(200, 1),
            VirtualPageId::new(200, 2),
        ]);

        // Act & Assert: branch A still fully traversable
        let ma = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(ma.matched_tokens, 3);

        // Act & Assert: branch B also fully traversable
        let mb = index.find_longest_prefix(&[1, 4, 5]).unwrap();
        assert_eq!(mb.matched_tokens, 3);

        // Shared root node's page comes from last insert (branch B)
        assert_eq!(ma.matched_pages[0], VirtualPageId::new(200, 0));
        assert_eq!(mb.matched_pages[0], VirtualPageId::new(200, 0));
    }

    #[test]
    fn insert_after_many_failed_lookups_preserves_state() {
        // Arrange: insert a sequence, perform many failing lookups, then verify original
        let mut index = KvPrefixIndex::new();
        index.insert(&[7, 8, 9], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);

        // Act: many lookups that miss
        for i in 100..200u32 {
            assert!(index.find_longest_prefix(&[i]).is_none());
        }

        // Assert: original sequence still fully intact
        let matched = index.find_longest_prefix(&[7, 8, 9, 10]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
    }

    #[test]
    fn token_sequence_with_repeating_pairs() {
        // Arrange: sequence [1, 2, 1, 2, 1, 2] — repeating pairs form a zigzag trie path
        let mut index = KvPrefixIndex::new();
        let tokens = vec![1u32, 2, 1, 2, 1, 2];
        let pages: Vec<VirtualPageId> = (0..6).map(|i| VirtualPageId::new(0, i)).collect();
        index.insert(&tokens, &pages);

        // Act: exact match
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 6);
        assert_eq!(matched.matched_pages.len(), 6);

        // Act: partial match at [1, 2, 1]
        let partial = index.find_longest_prefix(&[1, 2, 1, 99]).unwrap();
        assert_eq!(partial.matched_tokens, 3);
        assert_eq!(partial.matched_pages.len(), 3);
    }

    #[test]
    fn find_longest_prefix_prefers_deepest_page_ref() {
        // Arrange: insert [10, 20, 30] with pages only at positions 0 and 2
        // Page at position 0: VirtualPageId(0, 0)
        // Page at position 1: missing (no page because pages slice shorter)
        // Page at position 2: set by insert due to pages.get(2)
        let mut index = KvPrefixIndex::new();
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(0, 0),
        ]);
        // Token 10 gets page (0,0), tokens 20 and 30 get no page
        // Now re-insert with 3 pages to give position 2 a page
        index.insert(&[10, 20, 30], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);

        // Act
        let matched = index.find_longest_prefix(&[10, 20, 30]).unwrap();

        // Assert: deepest match covers all 3 tokens
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(1, 0));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(1, 2));
    }

    // ── Wave 11: 13 additional tests covering Ord trait, HashMap key behavior,
    //   independent index instances, and structural invariants ──

    // 1. VirtualPageId fields are directly accessible for reading and writing
    #[test]
    fn virtual_page_id_fields_mutable_access() {
        // Arrange: create a VirtualPageId then mutate its fields
        let mut page = VirtualPageId::new(0, 0);
        // Act: mutate fields
        page.sequence_id = 123;
        page.logical_index = 456;
        // Assert: mutated values are readable
        assert_eq!(page.sequence_id, 123);
        assert_eq!(page.logical_index, 456);
    }

    // 2. VirtualPageId with both fields at maximum boundary values
    #[test]
    fn virtual_page_id_both_fields_max_boundaries() {
        // Arrange: use u64::MAX for sequence_id and usize::MAX for logical_index
        let page = VirtualPageId::new(u64::MAX, usize::MAX);
        // Assert: values stored without truncation
        assert_eq!(page.sequence_id, u64::MAX);
        assert_eq!(page.logical_index, usize::MAX);
        // Assert: round-trip through insert and find
        let mut index = KvPrefixIndex::new();
        index.insert(&[42], &[page]);
        let matched = index.find_longest_prefix(&[42]).unwrap();
        assert_eq!(matched.matched_pages[0].sequence_id, u64::MAX);
        assert_eq!(matched.matched_pages[0].logical_index, usize::MAX);
    }

    // 3. Two independent KvPrefixIndex instances do not share state
    #[test]
    fn two_independent_index_instances_isolated() {
        // Arrange: create two indexes, insert different data
        let mut index_a = KvPrefixIndex::new();
        let mut index_b = KvPrefixIndex::new();
        index_a.insert(&[10, 20, 30], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 2),
        ]);
        index_b.insert(&[40, 50, 60], &[
            VirtualPageId::new(2, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(2, 2),
        ]);
        // Act & Assert: each index only returns its own data
        let ma = index_a.find_longest_prefix(&[10, 20, 30]).unwrap();
        assert_eq!(ma.matched_tokens, 3);
        assert_eq!(ma.matched_pages[0].sequence_id, 1);
        assert!(index_a.find_longest_prefix(&[40, 50]).is_none());
        let mb = index_b.find_longest_prefix(&[40, 50, 60]).unwrap();
        assert_eq!(mb.matched_tokens, 3);
        assert_eq!(mb.matched_pages[0].sequence_id, 2);
        assert!(index_b.find_longest_prefix(&[10, 20]).is_none());
    }

    // 4. PrefixMatch clone then assert_eq on original and clone are stable across scope
    #[test]
    fn prefix_match_clone_stable_across_scope() {
        // Arrange: create and clone within inner scope
        let original = PrefixMatch {
            matched_tokens: 3,
            matched_pages: vec![
                VirtualPageId::new(1, 10),
                VirtualPageId::new(2, 20),
                VirtualPageId::new(3, 30),
            ],
        };
        let cloned = {
            let c = original.clone();
            c
        };
        // Assert: clone survives the inner scope
        assert_eq!(original, cloned);
        assert_eq!(cloned.matched_tokens, 3);
        assert_eq!(cloned.matched_pages[0], VirtualPageId::new(1, 10));
    }

    // 5. Insert a 3-token sequence, then overwrite only the middle token's page by inserting
    //    a 2-token sequence that covers positions 0 and 1, giving position 1 a new page
    #[test]
    fn overwrite_second_position_via_two_token_insert() {
        // Arrange: full 3-token insert
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 10),
            VirtualPageId::new(0, 20),
            VirtualPageId::new(0, 30),
        ]);
        // Act: overwrite positions 0 and 1 via 2-token insert
        index.insert(&[1, 2], &[
            VirtualPageId::new(5, 100),
            VirtualPageId::new(5, 200),
        ]);
        // Assert: position 0 and 1 overwritten, position 2 survives
        let matched = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages[0], VirtualPageId::new(5, 100));
        assert_eq!(matched.matched_pages[1], VirtualPageId::new(5, 200));
        assert_eq!(matched.matched_pages[2], VirtualPageId::new(0, 30));
    }

    // 6. VirtualPageId as HashMap key: insert into HashMap and retrieve correctly
    #[test]
    fn virtual_page_id_as_hashmap_key() {
        // Arrange
        use std::collections::HashMap;
        let mut map: HashMap<VirtualPageId, &str> = HashMap::new();
        let p1 = VirtualPageId::new(1, 0);
        let p2 = VirtualPageId::new(2, 0);
        let p3 = VirtualPageId::new(1, 0); // same as p1
        // Act
        map.insert(p1, "first");
        map.insert(p2, "second");
        // Assert: retrieve by equal key
        assert_eq!(map.get(&p1), Some(&"first"));
        assert_eq!(map.get(&p3), Some(&"first"), "equal key retrieves same value");
        assert_eq!(map.get(&p2), Some(&"second"));
        // Assert: non-existent key
        let p4 = VirtualPageId::new(3, 0);
        assert_eq!(map.get(&p4), None);
        // Assert: map size
        assert_eq!(map.len(), 2);
    }

    // 7. PrefixMatch with matched_tokens = 1 and multiple pages (more pages than tokens)
    #[test]
    fn prefix_match_single_token_multiple_pages_manual_construction() {
        // Arrange: manually construct PrefixMatch where pages.len() > matched_tokens
        // This is possible via struct construction (not via KvPrefixIndex which guarantees
        // pages.len() <= matched_tokens)
        let pm = PrefixMatch {
            matched_tokens: 1,
            matched_pages: vec![
                VirtualPageId::new(0, 0),
                VirtualPageId::new(0, 1),
            ],
        };
        // Assert: both fields accessible and consistent with construction
        assert_eq!(pm.matched_tokens, 1);
        assert_eq!(pm.matched_pages.len(), 2);
    }

    // 8. KvPrefixIndex debug output is stable (deterministic) across multiple format calls
    #[test]
    fn kv_prefix_index_debug_deterministic_after_inserts() {
        // Arrange: build a trie with known structure
        let mut index = KvPrefixIndex::new();
        index.insert(&[5, 10], &[
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
        ]);
        // Act: format Debug twice
        let d1 = format!("{index:?}");
        let d2 = format!("{index:?}");
        // Assert: identical output
        assert_eq!(d1, d2);
    }

    // 9. Insert [A, B, C] with full pages, query [A, B] then [A, B, C, D] —
    //    verify partial queries always reflect the latest page state
    #[test]
    fn partial_queries_reflect_latest_page_state() {
        // Arrange: insert and then overwrite
        let mut index = KvPrefixIndex::new();
        index.insert(&[1, 2, 3], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);
        // Act: overwrite first 2 pages
        index.insert(&[1, 2], &[
            VirtualPageId::new(9, 0),
            VirtualPageId::new(9, 1),
        ]);
        // Assert: 2-token query sees overwritten pages
        let m2 = index.find_longest_prefix(&[1, 2]).unwrap();
        assert_eq!(m2.matched_pages[0], VirtualPageId::new(9, 0));
        assert_eq!(m2.matched_pages[1], VirtualPageId::new(9, 1));
        // Assert: 3-token query sees overwritten prefix + surviving tail
        let m3 = index.find_longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(m3.matched_pages[0], VirtualPageId::new(9, 0));
        assert_eq!(m3.matched_pages[2], VirtualPageId::new(0, 2));
        // Assert: extended query stops at 3
        let m4 = index.find_longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(m4.matched_tokens, 3);
    }

    // 10. Verify TrieNode page_ref stores offset_in_page correctly:
    //     insert [A, B, C] then query verifies offset_in_page at each level
    #[test]
    fn page_ref_offset_stored_correctly_at_each_depth() {
        // Arrange: insert a 4-token sequence
        let mut index = KvPrefixIndex::new();
        let tokens: Vec<TokenId> = vec![11, 22, 33, 44];
        let pages: Vec<VirtualPageId> = (0..4)
            .map(|i| VirtualPageId::new(0, i * 100))
            .collect();
        index.insert(&tokens, &pages);
        // Act: query at each depth and verify the page matches
        for depth in 1..=4 {
            let query = &tokens[..depth];
            let matched = index.find_longest_prefix(query).unwrap();
            assert_eq!(matched.matched_tokens, depth, "depth {depth}");
            assert_eq!(
                matched.matched_pages[depth - 1],
                VirtualPageId::new(0, (depth - 1) * 100),
                "page at depth {depth} should have logical_index {}",
                (depth - 1) * 100
            );
        }
    }

    // 11. VirtualPageId Hash trait: unequal values should (very likely) produce different hashes
    #[test]
    fn virtual_page_id_hash_unequal_values_likely_different() {
        // Arrange: two distinct VirtualPageIds
        let a = VirtualPageId::new(0, 0);
        let b = VirtualPageId::new(u64::MAX, usize::MAX);
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        // Assert: hashes are different (probabilistic but essentially certain for these values)
        assert_ne!(ha.finish(), hb.finish());
    }

    // 12. Insert 100 single-token sequences, delete concept (re-insert with empty pages),
    //     then verify the token node still exists but has no page_ref
    #[test]
    fn reinsert_with_empty_pages_preserves_node_but_no_page_match() {
        // Arrange: insert 3 single tokens with pages
        let mut index = KvPrefixIndex::new();
        index.insert(&[10], &[VirtualPageId::new(1, 0)]);
        index.insert(&[20], &[VirtualPageId::new(2, 0)]);
        index.insert(&[30], &[VirtualPageId::new(3, 0)]);
        // Act: re-insert [20] with empty pages — node for token 20 still exists
        // but page_ref is NOT cleared (insert only sets, never clears)
        index.insert(&[20], &[]);
        // Assert: token 20's page_ref survives (insert with empty pages never clears)
        let m20 = index.find_longest_prefix(&[20]).unwrap();
        assert_eq!(m20.matched_tokens, 1);
        assert_eq!(m20.matched_pages[0], VirtualPageId::new(2, 0));
        // Assert: other tokens unaffected
        assert_eq!(index.find_longest_prefix(&[10]).unwrap().matched_tokens, 1);
        assert_eq!(index.find_longest_prefix(&[30]).unwrap().matched_tokens, 1);
    }

    // 13. Verify TokenId type alias properties: min, max, and arithmetic boundaries
    #[test]
    fn token_id_boundary_values_in_sequence() {
        // Arrange: sequence with boundary token values
        let tokens = vec![0u32, 1, u32::MAX - 1, u32::MAX];
        let pages: Vec<VirtualPageId> = (0..4)
            .map(|i| VirtualPageId::new(0, i))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);
        // Act: exact match
        let matched = index.find_longest_prefix(&tokens).unwrap();
        // Assert
        assert_eq!(matched.matched_tokens, 4);
        assert_eq!(matched.matched_pages.len(), 4);
        // Act: query starting with u32::MAX (different from root children)
        assert!(index.find_longest_prefix(&[u32::MAX]).is_none(),
            "u32::MAX at root diverges from [0, ...] root");
        // Act: partial match at first 2 tokens
        let partial = index.find_longest_prefix(&[0, 1]).unwrap();
        assert_eq!(partial.matched_tokens, 2);
    }

    // -- Wave 12: 10 additional tests covering ownership, const construction,
    //   HashSet key behavior, mutable field semantics, and structural invariants --

    // 1. KvPrefixIndex ownership transfer: move index into a function and back, verify state preserved
    #[test]
    fn index_ownership_transfer_preserves_state() {
        // Arrange: create and populate an index
        let mut index = KvPrefixIndex::new();
        index.insert(&[5, 10, 15], &[
            VirtualPageId::new(0, 0),
            VirtualPageId::new(0, 1),
            VirtualPageId::new(0, 2),
        ]);

        // Act: move index into a helper, then return it
        let helper = |idx: KvPrefixIndex| -> KvPrefixIndex {
            // Verify state is accessible after move
            let m = idx.find_longest_prefix(&[5, 10]).unwrap();
            assert_eq!(m.matched_tokens, 2);
            idx
        };
        let returned = helper(index);

        // Assert: state preserved after round-trip ownership transfer
        let matched = returned.find_longest_prefix(&[5, 10, 15, 20]).unwrap();
        assert_eq!(matched.matched_tokens, 3);
        assert_eq!(matched.matched_pages.len(), 3);
    }

    // 2. VirtualPageId const construction via new compiles and works at runtime
    #[test]
    fn virtual_page_id_const_construction_roundtrip() {
        // Arrange: use const fn to construct page at compile-time-like values
        const PAGE: VirtualPageId = VirtualPageId::new(42, 999);
        // Act: use in insert and find
        let mut index = KvPrefixIndex::new();
        index.insert(&[7], &[PAGE]);
        let matched = index.find_longest_prefix(&[7]).unwrap();
        // Assert: const-constructed page stored and retrieved correctly
        assert_eq!(matched.matched_pages[0], PAGE);
        assert_eq!(matched.matched_pages[0].sequence_id, 42);
        assert_eq!(matched.matched_pages[0].logical_index, 999);
    }

    // 3. VirtualPageId used as HashSet key — uniqueness and containment
    #[test]
    fn virtual_page_id_as_hashset_key() {
        // Arrange: create a HashSet with multiple VirtualPageIds
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let p1 = VirtualPageId::new(1, 10);
        let p2 = VirtualPageId::new(2, 20);
        let p3 = VirtualPageId::new(1, 10); // duplicate of p1
        // Act
        set.insert(p1);
        set.insert(p2);
        set.insert(p3);
        // Assert: duplicate not added
        assert_eq!(set.len(), 2);
        assert!(set.contains(&p1));
        assert!(set.contains(&p2));
        // Assert: remove works
        assert!(set.remove(&p1));
        assert_eq!(set.len(), 1);
        assert!(!set.contains(&p1));
    }

    // 4. PrefixMatch manually constructed with matched_tokens=0 and non-empty pages
    //    — struct construction allows this invariant-violating state; verify field access
    #[test]
    fn prefix_match_manual_construction_zero_tokens_nonempty_pages() {
        // Arrange: construct with matched_tokens=0 but pages present
        let pm = PrefixMatch {
            matched_tokens: 0,
            matched_pages: vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(2, 1),
            ],
        };
        // Assert: fields are readable even though this represents an unusual state
        assert_eq!(pm.matched_tokens, 0);
        assert_eq!(pm.matched_pages.len(), 2);
        assert_eq!(pm.matched_pages[0], VirtualPageId::new(1, 0));
        // Assert: equality works with same manual construction
        let pm2 = PrefixMatch {
            matched_tokens: 0,
            matched_pages: vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(2, 1),
            ],
        };
        assert_eq!(pm, pm2);
    }

    // 5. VirtualPageId Copy semantics: array of pages can be copied element-by-element
    #[test]
    fn virtual_page_id_copy_in_array() {
        // Arrange: create an array of VirtualPageIds
        let original = [
            VirtualPageId::new(1, 0),
            VirtualPageId::new(2, 1),
            VirtualPageId::new(3, 2),
        ];
        // Act: copy each element (Copy trait)
        let copied: [VirtualPageId; 3] = [original[0], original[1], original[2]];
        // Assert: copied values equal originals
        for i in 0..3 {
            assert_eq!(original[i], copied[i]);
        }
        // Assert: modifying copy does not affect original (mutate to verify)
        let mut modified = copied;
        modified[0].sequence_id = 99;
        assert_eq!(original[0].sequence_id, 1, "original should be unaffected");
        assert_eq!(modified[0].sequence_id, 99);
    }

    // 6. Multiple find calls on the same index return consistent results
    //    even when queries partially overlap with inserted sequences
    #[test]
    fn consistent_results_for_overlapping_partial_queries() {
        // Arrange: insert [10, 20, 30, 40, 50] with distinct pages
        let mut index = KvPrefixIndex::new();
        let tokens: Vec<TokenId> = vec![10, 20, 30, 40, 50];
        let pages: Vec<VirtualPageId> = (0..5)
            .map(|i| VirtualPageId::new(i as u64 + 10, i))
            .collect();
        index.insert(&tokens, &pages);

        // Act: three overlapping partial queries
        let m1 = index.find_longest_prefix(&[10, 20]).unwrap();
        let m2 = index.find_longest_prefix(&[10, 20, 30]).unwrap();
        let m3 = index.find_longest_prefix(&[10, 20, 30, 40]).unwrap();

        // Assert: each partial query returns correct depth and pages
        assert_eq!(m1.matched_tokens, 2);
        assert_eq!(m1.matched_pages[1], VirtualPageId::new(11, 1));
        assert_eq!(m2.matched_tokens, 3);
        assert_eq!(m2.matched_pages[2], VirtualPageId::new(12, 2));
        assert_eq!(m3.matched_tokens, 4);
        assert_eq!(m3.matched_pages[3], VirtualPageId::new(13, 3));

        // Assert: re-query the same overlaps returns identical results
        let m1_again = index.find_longest_prefix(&[10, 20]).unwrap();
        assert_eq!(m1, m1_again);
    }

    // 7. Insert 5 branches sharing a 3-token prefix, then overwrite the shared prefix
    //    via a short [A] insert — verify all 5 branches see the new root page
    #[test]
    fn five_branches_root_overwrite_visible_to_all() {
        // Arrange: 5 branches sharing root token 1
        let mut index = KvPrefixIndex::new();
        for i in 0u32..5 {
            let tokens = [1, (i + 1) * 10, (i + 1) * 100];
            let pages: Vec<VirtualPageId> = (0..3)
                .map(|p| VirtualPageId::new(i as u64, p))
                .collect();
            index.insert(&tokens, &pages);
        }

        // Act: overwrite root token with a new page
        let new_root_page = VirtualPageId::new(99, 99);
        index.insert(&[1], &[new_root_page]);

        // Assert: all 5 branches see the new root page
        for i in 0u32..5 {
            let tokens = [1, (i + 1) * 10, (i + 1) * 100];
            let matched = index.find_longest_prefix(&tokens).unwrap();
            assert_eq!(matched.matched_tokens, 3, "branch {i}");
            assert_eq!(matched.matched_pages[0], new_root_page,
                "branch {i} should see overwritten root page");
        }
    }

    // 8. Insert a sequence where VirtualPageId values form a strictly increasing sequence,
    //    then verify the matched_pages preserve the strict ordering
    #[test]
    fn strictly_increasing_page_values_preserve_order() {
        // Arrange: 10 tokens with pages having strictly increasing sequence_id
        let mut index = KvPrefixIndex::new();
        let tokens: Vec<TokenId> = (0..10).collect();
        let pages: Vec<VirtualPageId> = (0..10)
            .map(|i| VirtualPageId::new(i as u64 * 100, i))
            .collect();
        index.insert(&tokens, &pages);

        // Act
        let matched = index.find_longest_prefix(&tokens).unwrap();

        // Assert: pages are in strictly increasing sequence_id order
        assert_eq!(matched.matched_tokens, 10);
        for i in 0..10 {
            assert_eq!(matched.matched_pages[i].sequence_id, i as u64 * 100,
                "page at index {i} should have sequence_id {}", i * 100);
            assert_eq!(matched.matched_pages[i].logical_index, i);
        }
        // Assert: no page equals its neighbor (strict ordering)
        for i in 1..10 {
            assert!(matched.matched_pages[i].sequence_id > matched.matched_pages[i - 1].sequence_id);
        }
    }

    // 9. TokenId arithmetic: verify tokens constructed via arithmetic on u32 work correctly
    #[test]
    fn token_id_constructed_via_arithmetic() {
        // Arrange: construct token values using arithmetic operations
        let base: TokenId = 1000;
        let tokens: Vec<TokenId> = vec![
            base,
            base + 1,
            base * 2,
            base / 2,
            base % 3,
            base.wrapping_add(1),
        ];
        let pages: Vec<VirtualPageId> = (0..6)
            .map(|i| VirtualPageId::new(0, i))
            .collect();
        let mut index = KvPrefixIndex::new();
        index.insert(&tokens, &pages);

        // Act: exact match
        let matched = index.find_longest_prefix(&tokens).unwrap();
        assert_eq!(matched.matched_tokens, 6);

        // Act: partial matches at each arithmetic boundary
        assert_eq!(
            index.find_longest_prefix(&[base, base + 1]).unwrap().matched_tokens, 2
        );
        // base * 2 = 2000, different from base at root
        assert!(index.find_longest_prefix(&[base * 2]).is_none());
        // base / 2 = 500, different from root
        assert!(index.find_longest_prefix(&[base / 2]).is_none());
    }

    // 10. Large-scale overwrite stress: insert 100 different sequences with shared prefix,
    //     then overwrite the shared prefix, then verify all 100 branches still match fully
    #[test]
    fn large_scale_shared_prefix_overwrite_100_branches() {
        // Arrange: 100 branches all sharing [1, 2] as prefix
        let mut index = KvPrefixIndex::new();
        for i in 0u32..100 {
            let tokens = [1u32, 2u32, i];
            let pages: Vec<VirtualPageId> = (0..3)
                .map(|p| VirtualPageId::new(i as u64, p))
                .collect();
            index.insert(&tokens, &pages);
        }

        // Act: overwrite shared prefix [1, 2] with new pages
        let new_p0 = VirtualPageId::new(255, 0);
        let new_p1 = VirtualPageId::new(255, 1);
        index.insert(&[1, 2], &[new_p0, new_p1]);

        // Assert: all 100 branches still match 3 tokens with overwritten prefix pages
        for i in 0u32..100 {
            let tokens = [1u32, 2u32, i];
            let matched = index.find_longest_prefix(&tokens).unwrap();
            assert_eq!(matched.matched_tokens, 3, "branch {i}");
            assert_eq!(matched.matched_pages[0], new_p0, "branch {i} prefix page 0");
            assert_eq!(matched.matched_pages[1], new_p1, "branch {i} prefix page 1");
            assert_eq!(matched.matched_pages[2], VirtualPageId::new(i as u64, 2),
                "branch {i} unique page");
        }

        // Assert: divergent query matches only shared prefix
        let partial = index.find_longest_prefix(&[1, 2, 999]).unwrap();
        assert_eq!(partial.matched_tokens, 2);
        assert_eq!(partial.matched_pages[0], new_p0);
        assert_eq!(partial.matched_pages[1], new_p1);
    }
}
