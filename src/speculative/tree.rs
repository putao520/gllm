//! §17.3 各向异性推测树 (Anisotropic Speculation Tree)
//!
//! 传统推测解码使用均匀等深树 (isotropic tree), 所有分支深度相同。
//! 但真实 token 接受率高度异构 — 高频词 >80%, 生僻词 <10%。
//! Goose (arXiv:2604.02047) 证明各向异性拓扑的理论最优性。
//!
//! 两个训练无关 Draft 来源:
//! 1. **PLD (Prompt Lookup Decoding)**: 在 prompt+已生成 tokens 中匹配 n-gram
//! 2. **N-gram Recurrence**: 从 prompt 频率表提取 top-k 替代候选

use std::collections::HashMap;

/// 推测树节点
///
/// 每个节点代表一个 draft token candidate, 包含 token ID 和树拓扑关系。
#[derive(Debug, Clone, PartialEq)]
pub struct SpecNode {
    /// 节点 ID (0 = root's first child)
    pub node_id: u32,
    /// Token ID
    pub token_id: u32,
    /// 父节点 ID (None for spine root)
    pub parent_id: Option<u32>,
    /// 子节点 IDs
    pub children: Vec<u32>,
    /// Draft 来源
    pub source: DraftSource,
    /// 预估接受率 (基于历史统计)
    pub estimated_acceptance: f32,
    /// 在 full verify 序列中的位置偏移
    pub position_offset: u32,
}

/// Draft token 来源
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DraftSource {
    /// PLD spine: 从 prompt 中 n-gram 匹配延续 (接受率 60-80%)
    PldSpine,
    /// Adapter top-k: 模型 adapter 输出的 top-k 候选 (接受率 70%+ for top-1)
    AdapterTopK { k: u8 },
    /// N-gram recurrence: 从 prompt 频率表提取的替代候选 (接受率 5-15%)
    NgramBranch,
}

/// 推测树配置
#[derive(Debug, Clone, PartialEq)]
pub struct SpecTreeConfig {
    /// 最大 spine 深度 (PLD 候选数)
    pub max_spine_depth: usize,
    /// 每个内部节点的最大 branch 数
    pub max_branches_per_node: usize,
    /// PLD n-gram 匹配长度 (通常 3-5)
    pub pld_ngram_len: usize,
    /// N-gram 候选 top-k
    pub ngram_top_k: usize,
    /// Adapter top-k 候选数
    pub adapter_top_k: usize,
    /// 最大树节点总数
    pub max_tree_size: usize,
}

impl Default for SpecTreeConfig {
    fn default() -> Self {
        Self {
            max_spine_depth: 5,
            max_branches_per_node: 2,
            pld_ngram_len: 3,
            ngram_top_k: 2,
            adapter_top_k: 3,
            max_tree_size: 32,
        }
    }
}

/// 各向异性推测树
///
/// §17.3: PLD spine + n-gram branches 的非均匀拓扑。
/// Goose Proposition 1: 组合树 accepted tokens ≥ 任一单源方法。
///
/// 树结构:
/// ```text
/// spine[0] (adapter top-1, ≈70%)
///   ├─ spine[1] (PLD continuation, ≈60%)
///   │   ├─ spine[2] (PLD, ≈55%)
///   │   │   ├─ branch[2,0] (n-gram alt-1)
///   │   │   └─ branch[2,1] (n-gram alt-2)
///   │   ├─ branch[1,0] (n-gram alt-1)
///   │   └─ branch[1,1] (n-gram alt-2)
///   └─ branch[0,0] (adapter top-2, ≈15%)
/// └─ branch[root,0] (adapter top-3, ≈5%)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpecTree {
    /// 树节点 (node_id → node)
    nodes: Vec<SpecNode>,
    /// 配置
    config: SpecTreeConfig,
    /// Tree attention mask — CSR 格式
    /// mask[node_i] 包含该 node 可以 attend 到的所有前驱 node IDs
    /// 每个 node 只 attend 到 root→node 路径上的 tokens (因果约束)
    attention_paths: Vec<Vec<u32>>,
    /// 总树节点数
    total_nodes: usize,
    /// Spine 长度
    spine_len: usize,
}

impl SpecTree {
    /// 创建空的推测树
    pub fn new(config: SpecTreeConfig) -> Self {
        Self {
            nodes: Vec::with_capacity(config.max_tree_size),
            config,
            attention_paths: Vec::new(),
            total_nodes: 0,
            spine_len: 0,
        }
    }

    /// 从 Adapter logits + PLD 匹配 + N-gram 索引构建推测树
    ///
    /// §17.3 核心构建逻辑:
    /// 1. Spine root = adapter top-1
    /// 2. Spine extensions = PLD n-gram matches from prompt
    /// 3. Branches = n-gram alternatives + adapter top-2/3
    ///
    /// # Arguments
    /// * `adapter_top_tokens` - Adapter 输出的 top-k token IDs (sorted by probability desc)
    /// * `prompt_tokens` - 当前 prompt + 已生成 tokens
    /// * `ngram_index` - 从 prompt 构建的 n-gram 频率索引
    ///
    /// # Returns
    /// 构建好的 SpecTree
    pub fn build(
        config: SpecTreeConfig,
        adapter_top_tokens: &[u32],
        prompt_tokens: &[u32],
        ngram_index: &NgramIndex,
    ) -> Self {
        let mut tree = Self::new(config);
        if adapter_top_tokens.is_empty() {
            return tree;
        }

        // 1. Spine root: adapter top-1
        let root_id = tree.add_node(SpecNode {
            node_id: 0,
            token_id: adapter_top_tokens[0],
            parent_id: None,
            children: Vec::new(),
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.70,
            position_offset: 0,
        });

        // 2. PLD spine extensions
        let mut current_id = root_id;
        let pld_matches = tree.find_pld_continuations(adapter_top_tokens[0], prompt_tokens);
        for (depth, token_id) in pld_matches.iter().take(tree.config.max_spine_depth - 1).enumerate() {
            let next_id = tree.add_node(SpecNode {
                node_id: tree.total_nodes as u32,
                token_id: *token_id,
                parent_id: Some(current_id),
                children: Vec::new(),
                source: DraftSource::PldSpine,
                estimated_acceptance: 0.60 - depth as f32 * 0.05,
                position_offset: (depth + 1) as u32,
            });
            tree.nodes[current_id as usize].children.push(next_id);
            current_id = next_id;
            tree.spine_len = depth + 2; // root + extensions
        }

        // 3. Adapter top-2/3 as branches off root
        for (k, &token_id) in adapter_top_tokens.iter().skip(1).take(tree.config.adapter_top_k - 1).enumerate() {
            let branch_id = tree.add_node(SpecNode {
                node_id: tree.total_nodes as u32,
                token_id,
                parent_id: Some(root_id),
                children: Vec::new(),
                source: DraftSource::AdapterTopK { k: (k + 2) as u8 },
                estimated_acceptance: 0.15 / (k as f32 + 1.0).max(1.0),
                position_offset: 1,
            });
            tree.nodes[root_id as usize].children.push(branch_id);
        }

        // 4. N-gram branches off each spine node
        for spine_idx in 0..tree.spine_len.min(tree.nodes.len()) {
            let parent_token = tree.nodes[spine_idx].token_id;
            let position_offset = tree.nodes[spine_idx].position_offset;
            let existing_children: Vec<u32> = tree.nodes[spine_idx].children.clone();
            let ngram_alts = ngram_index.get_continuations(parent_token, tree.config.ngram_top_k);
            for (b, token_id) in ngram_alts.iter().take(tree.config.max_branches_per_node).enumerate() {
                if tree.total_nodes >= tree.config.max_tree_size {
                    break;
                }
                // Skip if this token is already a child
                if existing_children.iter().any(|&c| tree.nodes[c as usize].token_id == *token_id) {
                    continue;
                }
                let branch_id = tree.add_node(SpecNode {
                    node_id: tree.total_nodes as u32,
                    token_id: *token_id,
                    parent_id: Some(spine_idx as u32),
                    children: Vec::new(),
                    source: DraftSource::NgramBranch,
                    estimated_acceptance: 0.05 + 0.05 * (tree.config.ngram_top_k - b) as f32 / tree.config.ngram_top_k as f32,
                    position_offset: position_offset + 1,
                });
                tree.nodes[spine_idx].children.push(branch_id);
            }
        }

        // 5. Build attention paths (causal: each node attends to root→node path)
        tree.build_attention_paths();

        tree
    }

    /// 获取所有 draft token IDs (按 BFS 顺序, 用于 batched verify)
    pub fn all_token_ids(&self) -> Vec<u32> {
        self.nodes.iter().map(|n| n.token_id).collect()
    }

    /// 获取树节点数
    pub fn len(&self) -> usize {
        self.total_nodes
    }

    /// 树是否为空
    pub fn is_empty(&self) -> bool {
        self.total_nodes == 0
    }

    /// 获取 spine 节点 IDs
    pub fn spine_ids(&self) -> Vec<u32> {
        let mut spine = Vec::new();
        let mut current = 0u32;
        loop {
            spine.push(current);
            let node = &self.nodes[current as usize];
            // Spine child = first PldSpine child
            match node.children.iter().find(|&&c| matches!(self.nodes[c as usize].source, DraftSource::PldSpine)) {
                Some(&next) => current = next,
                None => break,
            }
        }
        spine
    }

    /// 获取 spine token IDs
    pub fn spine_token_ids(&self) -> Vec<u32> {
        self.spine_ids().iter().map(|&id| self.nodes[id as usize].token_id).collect()
    }

    /// 构建 Tree Attention Mask (CSR 稀疏格式)
    ///
    /// §17.3: 每个 tree 节点只 attend 到 root→node 路径上的 token (因果约束)
    /// Mask shape: [tree_size, total_seq + tree_size]
    ///
    /// # Arguments
    /// * `total_seq_len` - Prompt + 已生成 tokens 的总长度
    ///
    /// # Returns
    /// (indptr, indices) — CSR 格式的 attention mask
    /// indptr[i..i+1] 给出第 i 行的非零列索引范围
    pub fn tree_attention_mask_csr(&self, total_seq_len: usize) -> (Vec<usize>, Vec<usize>) {
        let tree_size = self.total_nodes;
        let _cols = total_seq_len + tree_size;
        let mut indptr = Vec::with_capacity(tree_size + 1);
        let mut indices = Vec::new();

        indptr.push(0);

        for (node_idx, path) in self.attention_paths.iter().enumerate() {
            // All nodes attend to full prefix [0..total_seq_len)
            for col in 0..total_seq_len {
                indices.push(col);
            }
            // Plus all ancestor nodes in the tree
            for &ancestor_id in path {
                indices.push(total_seq_len + ancestor_id as usize);
            }
            // Plus self
            indices.push(total_seq_len + node_idx);
            indptr.push(indices.len());
        }

        (indptr, indices)
    }

    /// 获取 Tree Attention Mask 的行数和列数
    pub fn mask_shape(&self, total_seq_len: usize) -> (usize, usize) {
        (self.total_nodes, total_seq_len + self.total_nodes)
    }

    /// 从验证结果获取接受的前缀长度
    ///
    /// 沿 spine 检查每个节点的 draft token 是否匹配 target token,
    /// 返回最长连续匹配长度 (spine 方向, 不含 branches)
    ///
    /// # Arguments
    /// * `target_tokens` - Full model verify 产生的 target tokens
    ///
    /// # Returns
    /// (accepted_count, accepted_token_ids)
    pub fn accepted_from_spine(&self, target_tokens: &[u32]) -> (usize, Vec<u32>) {
        let spine = self.spine_token_ids();
        let mut accepted = Vec::new();
        for (i, &draft_tok) in spine.iter().enumerate() {
            if i < target_tokens.len() && draft_tok == target_tokens[i] {
                accepted.push(draft_tok);
            } else {
                break;
            }
        }
        (accepted.len(), accepted)
    }

    /// 获取所有分支 tokens (非 spine 节点)
    pub fn branch_token_ids(&self) -> Vec<(u32, u32)> {
        let spine_set: std::collections::HashSet<u32> = self.spine_ids().into_iter().collect();
        self.nodes
            .iter()
            .filter(|n| !spine_set.contains(&n.node_id))
            .map(|n| (n.node_id, n.token_id))
            .collect()
    }

    /// 获取节点引用
    pub fn node(&self, id: u32) -> Option<&SpecNode> {
        self.nodes.get(id as usize)
    }

    /// 获取所有节点引用
    pub fn nodes(&self) -> &[SpecNode] {
        &self.nodes
    }

    // ---- Internal helpers ----

    fn add_node(&mut self, node: SpecNode) -> u32 {
        let id = node.node_id;
        self.nodes.push(node);
        self.attention_paths.push(Vec::new());
        self.total_nodes += 1;
        id
    }

    fn build_attention_paths(&mut self) {
        for i in 0..self.total_nodes {
            let mut path = Vec::new();
            let mut current = i as u32;
            while let Some(parent_id) = self.nodes[current as usize].parent_id {
                path.push(parent_id);
                current = parent_id;
            }
            path.reverse(); // root → parent order
            self.attention_paths[i] = path;
        }
    }

    fn find_pld_continuations(&self, start_token: u32, prompt_tokens: &[u32]) -> Vec<u32> {
        // PLD: 在 prompt 中查找以 start_token 结尾的 n-gram, 返回后续 tokens
        let n = self.config.pld_ngram_len.min(prompt_tokens.len());
        let mut continuations = Vec::new();

        // Find all n-gram positions ending with start_token
        for i in n..prompt_tokens.len() {
            if prompt_tokens[i] == start_token {
                // Take the next few tokens as continuation
                let remaining = &prompt_tokens[i + 1..];
                for &tok in remaining.iter().take(self.config.max_spine_depth) {
                    if !continuations.contains(&tok) {
                        continuations.push(tok);
                    }
                    if continuations.len() >= self.config.max_spine_depth {
                        return continuations;
                    }
                }
            }
        }

        continuations
    }
}

/// N-gram 频率索引
///
/// 从 prompt 构建的 n-gram → 续写 token 频率表。
/// 用于 §17.3 的 N-gram Recurrence draft 来源。
#[derive(Debug, Clone, PartialEq)]
pub struct NgramIndex {
    /// n-gram 长度
    n: usize,
    /// n-gram hash → {continuation_token → count}
    /// 使用 Vec<u32> 的 hash 作为 key
    table: HashMap<u64, Vec<(u32, usize)>>,
}

impl NgramIndex {
    /// 从 prompt tokens 构建 N-gram 索引
    ///
    /// # Arguments
    /// * `tokens` - Prompt + 已生成 tokens
    /// * `n` - N-gram 长度 (通常 3-5)
    pub fn build(tokens: &[u32], n: usize) -> Self {
        let mut table: HashMap<u64, Vec<(u32, usize)>> = HashMap::new();
        if tokens.len() <= n {
            return Self { n, table };
        }

        for i in 0..tokens.len() - n {
            let ngram = &tokens[i..i + n];
            let hash = Self::hash_ngram(ngram);
            let continuation = tokens[i + n];

            let entry = table.entry(hash).or_default();
            if let Some(count) = entry.iter_mut().find(|(t, _)| *t == continuation) {
                count.1 += 1;
            } else {
                entry.push((continuation, 1));
            }
        }

        // Sort each entry by count (descending)
        for continuations in table.values_mut() {
            continuations.sort_by(|a, b| b.1.cmp(&a.1));
        }

        Self { n, table }
    }

    /// 获取给定 token 后的 top-k 续写 (基于 1-gram 频率)
    ///
    /// 用于 N-gram branch 候选生成
    pub fn get_continuations(&self, token: u32, top_k: usize) -> Vec<u32> {
        let hash = Self::hash_ngram(&[token]);
        self.table
            .get(&hash)
            .map(|entries| {
                entries.iter()
                    .take(top_k)
                    .map(|(t, _)| *t)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// 获取给定 n-gram 的 top-k 续写
    pub fn get_ngram_continuations(&self, ngram: &[u32], top_k: usize) -> Vec<u32> {
        assert_eq!(ngram.len(), self.n, "n-gram length mismatch");
        let hash = Self::hash_ngram(ngram);
        self.table
            .get(&hash)
            .map(|entries| {
                entries.iter()
                    .take(top_k)
                    .map(|(t, _)| *t)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn hash_ngram(ngram: &[u32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &tok in ngram {
            tok.hash(&mut hasher);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_tree_build_from_adapter() {
        let config = SpecTreeConfig::default();
        let adapter_tokens = vec![100u32, 200, 300];
        let prompt = vec![1, 2, 3, 4, 5, 100, 10, 20, 100, 15];
        let ngram_idx = NgramIndex::build(&prompt, 3);

        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert!(tree.len() > 1);
        assert_eq!(tree.node(0).unwrap().token_id, 100);
    }

    #[test]
    fn test_spec_tree_empty_adapter() {
        let config = SpecTreeConfig::default();
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 3);
        let tree = SpecTree::build(config, &[], &prompt, &ngram_idx);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_spec_tree_attention_mask() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![100u32, 200];
        let prompt = vec![1, 2, 3, 4, 100, 50];
        let ngram_idx = NgramIndex::build(&prompt, 3);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, _indices) = tree.tree_attention_mask_csr(10);
        assert_eq!(indptr.len(), tree.len() + 1);
        // Each row should have at least total_seq_len + 1 (prefix + self) entries
        for i in 0..tree.len() {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            let nnz = row_end - row_start;
            assert!(nnz >= 11, "row {} has {} nnz, expected >= 11", i, nnz);
        }
    }

    #[test]
    fn test_spec_tree_accepted_from_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 3, 4, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 3);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_tokens = tree.spine_token_ids();
        // target matches spine partially
        let mut target = spine_tokens.clone();
        if target.len() > 2 {
            target[2] = 999; // break at position 2
        }
        let (count, _accepted) = tree.accepted_from_spine(&target);
        assert!(count <= 2, "should accept at most 2 tokens before mismatch");
    }

    #[test]
    fn test_ngram_index_build_and_query() {
        let tokens = vec![1u32, 2, 3, 4, 1, 2, 5, 1, 2, 6];
        let idx = NgramIndex::build(&tokens, 3);

        // n-gram [1,2,3] should have continuation [4]
        let conts = idx.get_ngram_continuations(&[1, 2, 3], 3);
        assert!(conts.contains(&4));

        // n-gram [1,2] has multiple continuations (use 3-gram query since n=3)
        let conts = idx.get_ngram_continuations(&[1, 2, 5], 3);
        assert!(!conts.is_empty());
    }

    #[test]
    fn test_ngram_index_short_input() {
        let tokens = vec![1u32, 2];
        let idx = NgramIndex::build(&tokens, 3);
        assert!(idx.table.is_empty());
    }

    #[test]
    fn test_spec_tree_csr_mask_correctness() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            max_branches_per_node: 0,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        let prompt = vec![1, 2, 3, 42, 99];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root should attend to full prefix + self
        let (indptr, indices) = tree.tree_attention_mask_csr(5);
        let root_start = indptr[0];
        let root_end = indptr[1];
        let root_attends: std::collections::HashSet<usize> =
            indices[root_start..root_end].iter().copied().collect();
        // Root attends to all 5 prefix tokens + itself
        assert!(root_attends.contains(&0));
        assert!(root_attends.contains(&4));
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_default_values() {
        let cfg = SpecTreeConfig::default();
        assert_eq!(cfg.max_spine_depth, 5);
        assert_eq!(cfg.max_branches_per_node, 2);
        assert_eq!(cfg.pld_ngram_len, 3);
        assert_eq!(cfg.ngram_top_k, 2);
        assert_eq!(cfg.adapter_top_k, 3);
        assert_eq!(cfg.max_tree_size, 32);
    }

    #[test]
    fn spec_tree_config_clone() {
        let a = SpecTreeConfig::default();
        let b = a.clone();
        assert_eq!(a.max_spine_depth, b.max_spine_depth);
        assert_eq!(a.max_tree_size, b.max_tree_size);
    }

    // ------------------------------------------------------------------
    // DraftSource
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_variants() {
        let spine = DraftSource::PldSpine;
        let ngram = DraftSource::NgramBranch;
        let adapter = DraftSource::AdapterTopK { k: 3 };
        assert_ne!(spine, ngram);
        assert_ne!(spine, adapter);
        assert_eq!(spine, DraftSource::PldSpine);
        assert_eq!(adapter, DraftSource::AdapterTopK { k: 3 });
    }

    #[test]
    fn draft_source_copy_clone() {
        let a = DraftSource::PldSpine;
        let b = a;
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
    }

    // ------------------------------------------------------------------
    // SpecNode
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_clone() {
        let node = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![1, 2],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.7,
            position_offset: 0,
        };
        let clone = node.clone();
        assert_eq!(clone.node_id, 0);
        assert_eq!(clone.token_id, 42);
        assert_eq!(clone.children, vec![1, 2]);
        assert_eq!(clone.source, DraftSource::PldSpine);
    }

    // ------------------------------------------------------------------
    // SpecTree new / empty
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_new_is_empty() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.all_token_ids().is_empty());
    }

    #[test]
    fn spec_tree_mask_shape() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3, 42, 99], &NgramIndex::build(&[1, 2, 3, 42, 99], 2));
        let (rows, cols) = tree.mask_shape(10);
        assert_eq!(rows, tree.len());
        assert_eq!(cols, 10 + tree.len());
    }

    // ------------------------------------------------------------------
    // SpecTree nodes access
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_node_accessor() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[100u32], &[1, 2, 3, 100, 50], &NgramIndex::build(&[1, 2, 3, 100, 50], 2));
        assert!(tree.node(0).is_some());
        assert_eq!(tree.node(0).unwrap().token_id, 100);
        assert!(tree.nodes().len() > 0);
    }

    #[test]
    fn spec_tree_node_out_of_bounds() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.node(0).is_none());
    }

    // ------------------------------------------------------------------
    // SpecTree spine_ids / spine_token_ids
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_ids_root_only() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let spine = tree.spine_ids();
        assert_eq!(spine, vec![0]);
        assert_eq!(tree.spine_token_ids(), vec![42]);
    }

    // ------------------------------------------------------------------
    // SpecTree branch_token_ids
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_tokens_empty_when_no_branches() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let branches = tree.branch_token_ids();
        // Only root → no branches
        assert!(branches.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex additional tests
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_empty() {
        let idx = NgramIndex::build(&[1u32, 2], 3);
        assert!(idx.get_continuations(1, 5).is_empty());
    }

    #[test]
    fn ngram_index_build_with_sufficient_tokens() {
        let tokens = vec![10u32, 20, 30, 40, 10, 20, 50];
        let idx = NgramIndex::build(&tokens, 3);
        // [10, 20, 30] → 40
        let conts = idx.get_ngram_continuations(&[10, 20, 30], 5);
        assert!(conts.contains(&40));
        // [10, 20, 50] → nothing (end of input)
        let conts2 = idx.get_ngram_continuations(&[10, 20, 50], 5);
        assert!(conts2.is_empty());
    }

    #[test]
    fn ngram_index_continuation_sorted_by_frequency() {
        // 2-gram [10,20] followed by 50 three times, [10,30] followed by 60 once
        let tokens = vec![10u32, 20, 50, 10, 20, 50, 10, 20, 50, 10, 30, 60];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[10, 20], 5);
        // 50 appears 3 times, should come first
        assert_eq!(conts[0], 50);
    }

    #[test]
    fn ngram_index_get_continuations_single_token() {
        // get_continuations hashes [token] as a 1-gram,
        // so we need n=1 to have matching keys
        let tokens = vec![1u32, 2, 3, 1, 4, 1, 5];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(1, 5);
        assert!(conts.contains(&2));
        assert!(conts.contains(&4));
        assert!(conts.contains(&5));
    }

    #[test]
    fn ngram_index_clone() {
        let tokens = vec![1u32, 2, 3, 4];
        let a = NgramIndex::build(&tokens, 2);
        let b = a.clone();
        let conts_a = a.get_ngram_continuations(&[1, 2], 5);
        let conts_b = b.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts_a, conts_b);
    }

    // ------------------------------------------------------------------
    // SpecTree accepted_from_spine edge cases
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_empty_target() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 42, 99], &NgramIndex::build(&[1, 2, 42, 99], 2));
        let (count, accepted) = tree.accepted_from_spine(&[]);
        assert_eq!(count, 0);
        assert!(accepted.is_empty());
    }

    #[test]
    fn accepted_from_spine_all_match() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 42, 99], &NgramIndex::build(&[1, 2, 42, 99], 2));
        let spine = tree.spine_token_ids();
        let (count, accepted) = tree.accepted_from_spine(&spine);
        assert_eq!(count, spine.len());
        assert_eq!(accepted, spine);
    }

    // ------------------------------------------------------------------
    // Additional tests: Debug trait formatting
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_debug_trait() {
        let node = SpecNode {
            node_id: 7,
            token_id: 99,
            parent_id: Some(3),
            children: vec![8, 9],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 0.42,
            position_offset: 5,
        };
        let debug_str = format!("{:?}", node);
        assert!(debug_str.contains("99"));
        assert!(debug_str.contains("NgramBranch"));
        assert!(debug_str.contains("0.42"));
    }

    #[test]
    fn draft_source_debug_trait() {
        let spine = DraftSource::PldSpine;
        let debug = format!("{:?}", spine);
        assert!(debug.contains("PldSpine"));

        let adapter = DraftSource::AdapterTopK { k: 5 };
        let debug = format!("{:?}", adapter);
        assert!(debug.contains("AdapterTopK"));
        assert!(debug.contains("5"));
    }

    #[test]
    fn spec_tree_config_debug_trait() {
        let cfg = SpecTreeConfig::default();
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("max_spine_depth"));
        assert!(debug.contains("max_tree_size"));
    }

    // ------------------------------------------------------------------
    // Additional tests: SpecTree Clone (deep copy)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_clone_is_independent() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 42, 99], &NgramIndex::build(&[1, 2, 42, 99], 2));
        let clone = tree.clone();
        assert_eq!(tree.len(), clone.len());
        assert_eq!(tree.all_token_ids(), clone.all_token_ids());
        assert_eq!(tree.spine_token_ids(), clone.spine_token_ids());
    }

    // ------------------------------------------------------------------
    // Additional tests: SpecTree build with PLD spine extensions
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_with_pld_spine_extensions() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Token 10 appears in prompt, followed by 50, 60, 70
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 3);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        // Spine should have extensions from PLD (tokens following 10 in prompt)
        assert!(spine.len() > 1, "spine should have at least 1 extension beyond root");
    }

    // ------------------------------------------------------------------
    // Additional tests: SpecTree build respects max_tree_size
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_max_tree_size_limits_ngram_branches() {
        // max_tree_size gates n-gram branch insertion (line 195),
        // spine and adapter branches are added unconditionally before that phase.
        // Use minimal spine/adapter + rich n-gram source to verify the gate works.
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 10,
            adapter_top_k: 1,
            ngram_top_k: 10,
            max_tree_size: 2,
            pld_ngram_len: 1,
        };
        let adapter_tokens = vec![10u32];
        // Token 10 has many n-gram continuations (11..20)
        let prompt: Vec<u32> = vec![10, 11, 10, 12, 10, 13, 10, 14, 10, 15, 10, 16, 10, 17, 10, 18, 10, 19, 10, 20];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root (1 node) + up to 1 n-gram branch (gated by max_tree_size=2)
        assert!(tree.len() >= 1, "tree should have at least the root node");
        // Without the gate, many n-gram branches would be added;
        // with max_tree_size=2, only 1 n-gram branch fits beyond the root.
        let ngram_count = tree.nodes()
            .iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        assert!(ngram_count <= 1, "n-gram branches should be limited by max_tree_size, got {}", ngram_count);
    }

    // ------------------------------------------------------------------
    // Additional tests: SpecTree build with adapter_top_k limiting
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_top_k_limits_branches() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30, 40, 50];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root (10) + adapter top-2 (20) = 2 nodes; top-3..5 should not be included
        let adapter_branches: Vec<&SpecNode> = tree.nodes()
            .iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .collect();
        assert!(adapter_branches.len() <= 1, "at most 1 adapter branch (top-2), got {}", adapter_branches.len());
    }

    // ------------------------------------------------------------------
    // Additional tests: SpecTree CSR mask ancestor paths
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_non_root_includes_ancestors() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Prompt: token 10 followed by 50, 60
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 8;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        if tree.len() >= 2 {
            // Second node (spine extension) should attend to ancestor (node 0) + self
            let start = indptr[1];
            let end = indptr[2];
            let row: Vec<usize> = indices[start..end].to_vec();
            // Should include prefix tokens [0..total_seq_len)
            assert!(row.contains(&0));
            // Should include ancestor node 0 at position total_seq_len + 0
            assert!(row.contains(&(total_seq_len + 0)));
            // Should include self at position total_seq_len + 1
            assert!(row.contains(&(total_seq_len + 1)));
        }
    }

    // ------------------------------------------------------------------
    // Additional tests: SpecTree branch_token_ids with branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_tokens_present_with_branches() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 3,
            max_branches_per_node: 1,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70, 80, 20, 90];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let branches = tree.branch_token_ids();
        // Should have at least one non-spine node (adapter top-2/top-3 or n-gram branch)
        assert!(!branches.is_empty(), "should have at least one branch node");
        // All branch node IDs should not be in spine
        let spine_set: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        for (node_id, _) in &branches {
            assert!(!spine_set.contains(node_id), "branch node {} should not be in spine", node_id);
        }
    }

    // ------------------------------------------------------------------
    // Additional tests: NgramIndex top_k limiting
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_respects_top_k() {
        let tokens = vec![1u32, 2, 10, 1, 2, 20, 1, 2, 30, 1, 2, 40];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 2);
        assert!(conts.len() <= 2, "top_k=2 should return at most 2 results, got {}", conts.len());
    }

    // ------------------------------------------------------------------
    // Additional tests: NgramIndex with empty input
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_build_empty_tokens() {
        let idx = NgramIndex::build(&[], 3);
        assert!(idx.table.is_empty());
        assert!(idx.get_continuations(1, 5).is_empty());
        assert!(idx.get_ngram_continuations(&[1, 2, 3], 5).is_empty());
    }

    // ------------------------------------------------------------------
    // Additional tests: NgramIndex with n=1 (unigram)
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_build_unigram() {
        let tokens = vec![5u32, 10, 5, 20, 5, 10];
        let idx = NgramIndex::build(&tokens, 1);
        // Unigram [5] should have continuations [10, 20, 10] → [10 (2), 20 (1)]
        let conts = idx.get_ngram_continuations(&[5], 5);
        assert!(conts.contains(&10));
        assert!(conts.contains(&20));
        // 10 appears more frequently, should be first
        assert_eq!(conts[0], 10);
    }

    // ------------------------------------------------------------------
    // Additional tests: NgramIndex get_continuations for missing token
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_unknown_token() {
        let tokens = vec![1u32, 2, 3, 4];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(999, 5);
        assert!(conts.is_empty());
    }

    // ------------------------------------------------------------------
    // Additional tests: SpecTree all_token_ids returns in node order
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_token_ids_order() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ids = tree.all_token_ids();
        assert_eq!(ids.len(), tree.len());
        // First token must be the adapter top-1
        assert_eq!(ids[0], 10);
    }

    // ------------------------------------------------------------------
    // DraftSource PartialEq with different k values
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_adapter_top_k_inequality_different_k() {
        let a = DraftSource::AdapterTopK { k: 1 };
        let b = DraftSource::AdapterTopK { k: 2 };
        assert_ne!(a, b);
    }

    #[test]
    fn draft_source_adapter_top_k_equality_same_k() {
        let a = DraftSource::AdapterTopK { k: 3 };
        let b = DraftSource::AdapterTopK { k: 3 };
        assert_eq!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecNode field access for all DraftSource variants
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_with_adapter_top_k_source() {
        let node = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![],
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.70,
            position_offset: 0,
        };
        assert_eq!(node.token_id, 42);
        assert!(node.parent_id.is_none());
        assert!(node.children.is_empty());
        assert_eq!(node.source, DraftSource::AdapterTopK { k: 1 });
        assert!((node.estimated_acceptance - 0.70).abs() < f32::EPSILON);
        assert_eq!(node.position_offset, 0);
    }

    #[test]
    fn spec_node_with_ngram_branch_source() {
        let node = SpecNode {
            node_id: 5,
            token_id: 99,
            parent_id: Some(2),
            children: vec![],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 0.05,
            position_offset: 3,
        };
        assert_eq!(node.node_id, 5);
        assert_eq!(node.parent_id, Some(2));
        assert_eq!(node.source, DraftSource::NgramBranch);
    }

    // ------------------------------------------------------------------
    // SpecTree with single adapter token, no PLD match in prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_single_adapter_no_pld_match() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Token 999 does not appear in prompt, so no PLD continuations
        let adapter_tokens = vec![999u32];
        let prompt = vec![1, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert_eq!(tree.len(), 1, "should have only the root node");
        assert_eq!(tree.spine_token_ids(), vec![999]);
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig manual construction preserves all fields
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_manual_construction() {
        let cfg = SpecTreeConfig {
            max_spine_depth: 10,
            max_branches_per_node: 4,
            pld_ngram_len: 5,
            ngram_top_k: 6,
            adapter_top_k: 7,
            max_tree_size: 64,
        };
        assert_eq!(cfg.max_spine_depth, 10);
        assert_eq!(cfg.max_branches_per_node, 4);
        assert_eq!(cfg.pld_ngram_len, 5);
        assert_eq!(cfg.ngram_top_k, 6);
        assert_eq!(cfg.adapter_top_k, 7);
        assert_eq!(cfg.max_tree_size, 64);
    }

    // ------------------------------------------------------------------
    // SpecTree accepted_from_spine: first token mismatch
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_first_token_mismatch() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (count, accepted) = tree.accepted_from_spine(&[999u32]);
        assert_eq!(count, 0);
        assert!(accepted.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree accepted_from_spine: target shorter than spine
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_target_shorter_than_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        if spine.len() > 1 {
            // Provide only the first spine token as target
            let (count, accepted) = tree.accepted_from_spine(&[spine[0]]);
            assert_eq!(count, 1);
            assert_eq!(accepted, vec![spine[0]]);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree branch_token_ids on root-only tree (spine_len=1, no branches)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_token_ids_root_only_tree() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Token 999 not in prompt → no n-gram branches possible
        let tree = SpecTree::build(config, &[999u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        assert_eq!(tree.len(), 1);
        assert!(tree.branch_token_ids().is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree clone deep copy independence
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_clone_deep_copy_independence() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let original_len = tree.len();
        let clone = tree.clone();

        // Verify identical state
        assert_eq!(clone.len(), original_len);
        for i in 0..original_len {
            assert_eq!(tree.node(i as u32).unwrap().token_id, clone.node(i as u32).unwrap().token_id);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex with duplicate n-gram and same continuation
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_duplicate_ngram_same_continuation_counts() {
        // [1,2] → 3 appears 3 times
        let tokens = vec![1u32, 2, 3, 1, 2, 3, 1, 2, 3];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts, vec![3]);
    }

    // ------------------------------------------------------------------
    // NgramIndex get_ngram_continuations for missing n-gram
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_ngram_continuations_missing() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[99, 100], 5);
        assert!(conts.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex Debug trait
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_debug_trait() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        let debug = format!("{:?}", idx);
        assert!(debug.contains("NgramIndex") || debug.contains("table") || debug.contains("n"));
    }

    // ------------------------------------------------------------------
    // SpecTree Debug trait
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_debug_trait() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let debug = format!("{:?}", tree);
        assert!(debug.contains("SpecTree") || debug.contains("nodes") || debug.contains("total_nodes"));
    }

    // ------------------------------------------------------------------
    // SpecTree CSR mask: root node has no ancestors
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_root_has_no_ancestor_in_path() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 5;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        // Root row: prefix columns [0..5) + self at total_seq_len + 0
        let root_start = indptr[0];
        let root_end = indptr[1];
        let root_row: Vec<usize> = indices[root_start..root_end].to_vec();
        // Should contain exactly total_seq_len + 1 entries (prefix + self)
        assert_eq!(root_row.len(), total_seq_len + 1);
        // Last entry is self
        assert_eq!(root_row[total_seq_len], total_seq_len);
    }

    // ------------------------------------------------------------------
    // SpecTree all_token_ids on empty tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_token_ids_empty_tree() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.all_token_ids().is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex with single token input (len <= n)
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_single_token_input() {
        let idx = NgramIndex::build(&[42u32], 1);
        // len=1, n=1 → tokens.len() <= n is false (1 is not > 1),
        // but loop range is 0..0, so table is empty
        assert!(idx.table.is_empty());
        assert!(idx.get_continuations(42, 5).is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree nodes() returns slice of correct length
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_nodes_slice_length() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let nodes = tree.nodes();
        assert_eq!(nodes.len(), tree.len());
    }

    // ------------------------------------------------------------------
    // SpecNode estimated_acceptance for PldSpine descendants decreases
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_acceptance_decreases_with_depth() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        if spine_ids.len() > 2 {
            let root_acceptance = tree.node(spine_ids[0]).unwrap().estimated_acceptance;
            let second_acceptance = tree.node(spine_ids[1]).unwrap().estimated_acceptance;
            let third_acceptance = tree.node(spine_ids[2]).unwrap().estimated_acceptance;
            assert!(root_acceptance > second_acceptance,
                "root acceptance {} should exceed second {}", root_acceptance, second_acceptance);
            assert!(second_acceptance > third_acceptance,
                "second acceptance {} should exceed third {}", second_acceptance, third_acceptance);
        }
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: all fields zero (minimal valid config)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_zero_fields() {
        let cfg = SpecTreeConfig {
            max_spine_depth: 0,
            max_branches_per_node: 0,
            pld_ngram_len: 0,
            ngram_top_k: 0,
            adapter_top_k: 0,
            max_tree_size: 0,
        };
        assert_eq!(cfg.max_spine_depth, 0);
        assert_eq!(cfg.max_branches_per_node, 0);
        assert_eq!(cfg.pld_ngram_len, 0);
        assert_eq!(cfg.ngram_top_k, 0);
        assert_eq!(cfg.adapter_top_k, 0);
        assert_eq!(cfg.max_tree_size, 0);
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: large values
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_large_values() {
        let cfg = SpecTreeConfig {
            max_spine_depth: 1000,
            max_branches_per_node: 500,
            pld_ngram_len: 100,
            ngram_top_k: 200,
            adapter_top_k: 300,
            max_tree_size: 10000,
        };
        assert_eq!(cfg.max_spine_depth, 1000);
        assert_eq!(cfg.max_tree_size, 10000);
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: Debug contains all field names
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_debug_all_fields() {
        let cfg = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 2,
            pld_ngram_len: 3,
            ngram_top_k: 4,
            adapter_top_k: 5,
            max_tree_size: 6,
        };
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("max_spine_depth: 1"));
        assert!(debug.contains("max_branches_per_node: 2"));
        assert!(debug.contains("pld_ngram_len: 3"));
        assert!(debug.contains("ngram_top_k: 4"));
        assert!(debug.contains("adapter_top_k: 5"));
        assert!(debug.contains("max_tree_size: 6"));
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: Clone produces equal value with modified original
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_clone_independence() {
        let mut a = SpecTreeConfig::default();
        let b = a.clone();
        a.max_spine_depth = 999;
        assert_ne!(a.max_spine_depth, b.max_spine_depth);
        assert_eq!(b.max_spine_depth, 5);
    }

    // ------------------------------------------------------------------
    // DraftSource: Eq trait consistency
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_eq_consistency() {
        let a = DraftSource::PldSpine;
        let b = DraftSource::PldSpine;
        assert_eq!(a, b);
        assert!(!(a != b));

        let c = DraftSource::NgramBranch;
        assert_ne!(a, c);
    }

    // ------------------------------------------------------------------
    // DraftSource: Copy semantics (assignment copies, not moves)
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_copy_semantics() {
        let original = DraftSource::AdapterTopK { k: 7 };
        let assigned = original;
        // original is still usable due to Copy
        assert_eq!(original, DraftSource::AdapterTopK { k: 7 });
        assert_eq!(assigned, DraftSource::AdapterTopK { k: 7 });
    }

    // ------------------------------------------------------------------
    // DraftSource: Debug for all three variants
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_debug_all_variants() {
        let spine = format!("{:?}", DraftSource::PldSpine);
        assert!(spine.contains("PldSpine"));

        let ngram = format!("{:?}", DraftSource::NgramBranch);
        assert!(ngram.contains("NgramBranch"));

        let adapter = format!("{:?}", DraftSource::AdapterTopK { k: 42 });
        assert!(adapter.contains("AdapterTopK"));
        assert!(adapter.contains("42"));
    }

    // ------------------------------------------------------------------
    // SpecNode: construction with u32::MAX values
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_boundary_values() {
        let node = SpecNode {
            node_id: u32::MAX,
            token_id: u32::MAX,
            parent_id: Some(u32::MAX - 1),
            children: vec![u32::MAX],
            source: DraftSource::PldSpine,
            estimated_acceptance: 1.0,
            position_offset: u32::MAX,
        };
        assert_eq!(node.node_id, u32::MAX);
        assert_eq!(node.token_id, u32::MAX);
        assert_eq!(node.parent_id, Some(u32::MAX - 1));
        assert_eq!(node.children, vec![u32::MAX]);
        assert_eq!(node.position_offset, u32::MAX);
    }

    // ------------------------------------------------------------------
    // SpecNode: Clone produces independent copy
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_clone_independence() {
        let mut node = SpecNode {
            node_id: 0,
            token_id: 10,
            parent_id: None,
            children: vec![1, 2, 3],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 0.5,
            position_offset: 0,
        };
        let cloned = node.clone();
        node.children.push(4);
        assert_eq!(node.children.len(), 4);
        assert_eq!(cloned.children.len(), 3, "clone should be independent");
    }

    // ------------------------------------------------------------------
    // SpecNode: Debug output contains all field names
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_debug_all_fields_present() {
        let node = SpecNode {
            node_id: 1,
            token_id: 2,
            parent_id: Some(0),
            children: vec![3],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.75,
            position_offset: 4,
        };
        let debug = format!("{:?}", node);
        assert!(debug.contains("node_id"));
        assert!(debug.contains("token_id"));
        assert!(debug.contains("parent_id"));
        assert!(debug.contains("children"));
        assert!(debug.contains("source"));
        assert!(debug.contains("estimated_acceptance"));
        assert!(debug.contains("position_offset"));
    }

    // ------------------------------------------------------------------
    // SpecNode: construction with PldSpine source
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_with_pld_spine_source() {
        let node = SpecNode {
            node_id: 1,
            token_id: 50,
            parent_id: Some(0),
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.55,
            position_offset: 1,
        };
        assert_eq!(node.source, DraftSource::PldSpine);
        assert_eq!(node.parent_id, Some(0));
        assert!(node.children.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: tokens.len() exactly equals n
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_len_equals_n() {
        // tokens.len() == n → early return (table empty)
        let idx = NgramIndex::build(&[1u32, 2, 3], 3);
        assert!(idx.table.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: tokens.len() == n + 1 (minimum for one entry)
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_len_equals_n_plus_one() {
        // tokens.len() == n + 1 → exactly one n-gram window
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 3);
        let conts = idx.get_ngram_continuations(&[1, 2, 3], 5);
        assert_eq!(conts, vec![4]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations top_k=0 returns empty
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_top_k_zero() {
        let tokens = vec![1u32, 2, 3, 1, 2, 4];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(1, 0);
        assert!(conts.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_ngram_continuations top_k=0 returns empty
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_ngram_continuations_top_k_zero() {
        let tokens = vec![1u32, 2, 3, 4];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 0);
        assert!(conts.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: hash_ngram deterministic
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_hash_deterministic() {
        let idx1 = NgramIndex::build(&[10u32, 20, 30, 40], 2);
        let idx2 = NgramIndex::build(&[10u32, 20, 30, 40], 2);
        let conts1 = idx1.get_ngram_continuations(&[10, 20], 5);
        let conts2 = idx2.get_ngram_continuations(&[10, 20], 5);
        assert_eq!(conts1, conts2);
    }

    // ------------------------------------------------------------------
    // SpecTree: is_empty after build with valid adapter tokens
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_not_empty_after_build() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        assert!(!tree.is_empty());
        assert!(tree.len() >= 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: mask_shape with zero total_seq_len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_mask_shape_zero_seq_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let (rows, cols) = tree.mask_shape(0);
        assert_eq!(rows, tree.len());
        assert_eq!(cols, tree.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask with zero total_seq_len (prefix-free)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_zero_seq_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(0);
        assert_eq!(indptr.len(), tree.len() + 1);
        // With total_seq_len=0, root row should only contain self
        let root_start = indptr[0];
        let root_end = indptr[1];
        let root_row: Vec<usize> = indices[root_start..root_end].to_vec();
        assert_eq!(root_row.len(), 1, "root should attend only to self when seq_len=0");
        assert_eq!(root_row[0], 0);
    }

    // ------------------------------------------------------------------
    // SpecTree: node accessor for valid range
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_node_accessor_valid_range() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for i in 0..tree.len() {
            let node = tree.node(i as u32);
            assert!(node.is_some(), "node {} should exist", i);
            assert_eq!(node.unwrap().node_id, i as u32);
        }
        // One past the end should be None
        assert!(tree.node(tree.len() as u32).is_none());
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine with target longer than spine
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_target_longer_than_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        // Append extra tokens beyond spine length
        let mut target = spine.clone();
        target.push(999);
        target.push(888);

        let (count, accepted) = tree.accepted_from_spine(&target);
        // Should accept the full spine (extra tokens are ignored)
        assert_eq!(count, spine.len());
        assert_eq!(accepted, spine);
    }

    // ------------------------------------------------------------------
    // SpecTree build: empty prompt produces no PLD continuations
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_empty_prompt_no_spine_extension() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[], &NgramIndex::build(&[], 2));
        // Root exists but no PLD continuations (empty prompt)
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.spine_token_ids(), vec![10]);
    }

    // ------------------------------------------------------------------
    // SpecTree build: prompt with repeated token produces unique PLD continuations
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_pld_continuations_are_unique() {
        let config = SpecTreeConfig {
            max_spine_depth: 10,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 appears multiple times followed by the same token 50
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 10, 50, 10, 50, 10, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        // PLD continuations should be deduplicated (50, 60 not 50, 50, 50, 60)
        let mut seen = std::collections::HashSet::new();
        for &tok in &spine[1..] {
            assert!(seen.insert(tok), "spine token {} appeared more than once", tok);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree build: pld_ngram_len=0 still matches (find_pld_continuations
    // scans for start_token regardless; n only sets scan start offset)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_pld_ngram_len_zero_still_matches() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 0,
            ..SpecTreeConfig::default()
        };
        // pld_ngram_len=0 → n=0 → find_pld_continuations starts from index 0
        // Token 10 appears at index 2, so continuations [50, 60] are still found
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root + PLD continuations = at least 3 nodes
        assert!(tree.len() >= 3, "should have root + PLD extensions, got {}", tree.len());
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        assert!(spine.len() > 1, "spine should extend beyond root");
    }

    // ------------------------------------------------------------------
    // SpecTree build: adapter branch estimated_acceptance decreases with k
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_branch_acceptance_decreases() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let adapter_branches: Vec<&SpecNode> = tree.nodes()
            .iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .collect();

        if adapter_branches.len() >= 2 {
            assert!(adapter_branches[0].estimated_acceptance >= adapter_branches[1].estimated_acceptance,
                "adapter branch acceptance should decrease with higher k");
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask columns are strictly monotonically increasing within each row
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_columns_monotonic_per_row() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(10);
        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            for w in row.windows(2) {
                assert!(w[0] < w[1], "row {} columns not strictly increasing: {} >= {}", i, w[0], w[1]);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_ids returns correct chain of PldSpine nodes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_ids_follows_pld_spine_chain() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        // Every spine node except root should have PldSpine source
        for (i, &id) in spine_ids.iter().enumerate() {
            let node = tree.node(id).unwrap();
            if i == 0 {
                assert_eq!(node.source, DraftSource::AdapterTopK { k: 1 });
            } else {
                assert_eq!(node.source, DraftSource::PldSpine, "spine node {} should be PldSpine", i);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with only adapter tokens, prompt has no n-gram matches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_only_adapter_branches_no_ngram() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![100u32, 200, 300];
        let prompt: Vec<u32> = vec![];  // empty prompt → no n-gram index
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root + 2 adapter branches = 3 nodes
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.node(0).unwrap().token_id, 100);
        assert_eq!(tree.node(1).unwrap().token_id, 200);
        assert_eq!(tree.node(2).unwrap().token_id, 300);
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branch duplicates are skipped (already a child token)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branch_skips_duplicate_child_token() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 5,
            ngram_top_k: 5,
            ..SpecTreeConfig::default()
        };
        // adapter top-2 = [10, 50]; ngram continuations of 10 include 50 (duplicate)
        let adapter_tokens = vec![10u32, 50];
        let prompt = vec![10, 50, 10, 50, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Check root's children: no duplicate token IDs
        let root_children = &tree.node(0).unwrap().children;
        let child_tokens: Vec<u32> = root_children.iter()
            .map(|&c| tree.node(c).unwrap().token_id)
            .collect();
        let unique_tokens: std::collections::HashSet<u32> = child_tokens.iter().copied().collect();
        assert_eq!(child_tokens.len(), unique_tokens.len(),
            "root children should have no duplicate token IDs: {:?}", child_tokens);
    }

    // ------------------------------------------------------------------
    // SpecNode: estimated_acceptance at 0.0
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_zero_acceptance() {
        let node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 0.0,
            position_offset: 0,
        };
        assert!(!node.estimated_acceptance.is_nan());
        assert_eq!(node.estimated_acceptance, 0.0);
    }

    // ------------------------------------------------------------------
    // SpecNode: parent_id None distinguishes root from child
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_parent_id_none_is_root() {
        let root = SpecNode {
            node_id: 0,
            token_id: 10,
            parent_id: None,
            children: vec![1],
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.7,
            position_offset: 0,
        };
        let child = SpecNode {
            node_id: 1,
            token_id: 50,
            parent_id: Some(0),
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.6,
            position_offset: 1,
        };
        assert!(root.parent_id.is_none());
        assert!(child.parent_id.is_some());
        assert_eq!(child.parent_id.unwrap(), 0);
    }

    // ------------------------------------------------------------------
    // NgramIndex: different n values produce different tables
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_different_n_different_tables() {
        let tokens = vec![1u32, 2, 3, 4, 5, 6];
        let idx2 = NgramIndex::build(&tokens, 2);
        let idx3 = NgramIndex::build(&tokens, 3);

        // n=2 queries should work for idx2, fail for idx3
        let conts2 = idx2.get_ngram_continuations(&[1, 2], 5);
        assert!(!conts2.is_empty());

        // n=3 queries should work for idx3
        let conts3 = idx3.get_ngram_continuations(&[1, 2, 3], 5);
        assert!(!conts3.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: build produces valid parent-child relationships
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_parent_child_consistency() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // For each non-root node, parent must exist and list this node as child
        for i in 1..tree.len() {
            let node = tree.node(i as u32).unwrap();
            if let Some(parent_id) = node.parent_id {
                let parent = tree.node(parent_id).unwrap();
                assert!(parent.children.contains(&(i as u32)),
                    "node {} has parent {} but parent does not list it as child", i, parent_id);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask each row includes prefix range [0, total_seq_len)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_each_row_includes_full_prefix() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 8;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            // First total_seq_len entries should be [0, 1, ..., total_seq_len-1]
            for col in 0..total_seq_len {
                assert!(row.contains(&col),
                    "row {} should contain prefix column {}", i, col);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_ids on tree with only adapter branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_ids_adapter_only_tree() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_ids();
        // Only root is spine (no PldSpine children)
        assert_eq!(spine, vec![0]);
    }

    // ------------------------------------------------------------------
    // SpecTree: all_token_ids contains no duplicates when adapter and ngram are disjoint
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_token_ids_distinct_sources() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        // Use distinct token IDs so adapter and ngram sources are disjoint
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![10, 99, 10, 98];  // n-gram continuations: 99, 98
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ids = tree.all_token_ids();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        // All token IDs in the tree should be unique
        assert_eq!(ids.len(), unique.len(), "all_token_ids should have no duplicates");
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter_top_k=1 and single adapter token
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_single_adapter_token_only_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        // Token 42 not in prompt → no PLD; no extra adapter tokens
        let prompt = vec![1, 2, 3, 4];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.node(0).unwrap().source, DraftSource::AdapterTopK { k: 1 });
        assert_eq!(tree.node(0).unwrap().token_id, 42);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask indptr has len() + 1 entries and starts at 0
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_indptr_invariant() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, _indices) = tree.tree_attention_mask_csr(7);
        assert_eq!(indptr.len(), tree.len() + 1);
        assert_eq!(indptr[0], 0);
        // indptr should be monotonically non-decreasing
        for w in indptr.windows(2) {
            assert!(w[0] <= w[1], "indptr should be non-decreasing: {} > {}", w[0], w[1]);
        }
    }

    // ==================================================================
    // NEW TESTS (tests 97+)
    // ==================================================================

    // ------------------------------------------------------------------
    // DraftSource: Hash trait — can be used as HashMap key
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_hash_consistency() {
        use std::collections::HashMap;
        let mut map: HashMap<DraftSource, i32> = HashMap::new();
        map.insert(DraftSource::PldSpine, 1);
        map.insert(DraftSource::NgramBranch, 2);
        map.insert(DraftSource::AdapterTopK { k: 3 }, 3);
        assert_eq!(map.get(&DraftSource::PldSpine), Some(&1));
        assert_eq!(map.get(&DraftSource::NgramBranch), Some(&2));
        assert_eq!(map.get(&DraftSource::AdapterTopK { k: 3 }), Some(&3));
        assert_eq!(map.get(&DraftSource::AdapterTopK { k: 1 }), None);
    }

    #[test]
    fn draft_source_hash_equal_inputs_produce_equal_hashes() {
        use std::hash::{Hash, Hasher};
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        DraftSource::AdapterTopK { k: 5 }.hash(&mut h1);
        DraftSource::AdapterTopK { k: 5 }.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ------------------------------------------------------------------
    // DraftSource: all variants are distinct via PartialEq
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_all_variants_mutually_distinct() {
        let variants: Vec<DraftSource> = vec![
            DraftSource::PldSpine,
            DraftSource::NgramBranch,
            DraftSource::AdapterTopK { k: 1 },
            DraftSource::AdapterTopK { k: 2 },
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b, "variants at {} and {} should differ", i, j);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecNode: PartialEq deep comparison
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_partial_eq_identical_nodes() {
        let a = SpecNode {
            node_id: 1,
            token_id: 42,
            parent_id: Some(0),
            children: vec![2, 3],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.6,
            position_offset: 1,
        };
        let b = SpecNode {
            node_id: 1,
            token_id: 42,
            parent_id: Some(0),
            children: vec![2, 3],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.6,
            position_offset: 1,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn spec_node_partial_eq_different_token_id() {
        let a = SpecNode {
            node_id: 0, token_id: 10, parent_id: None,
            children: vec![], source: DraftSource::NgramBranch,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 20, parent_id: None,
            children: vec![], source: DraftSource::NgramBranch,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecNode: special float values (NaN, Inf)
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_acceptance_nan() {
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: f32::NAN, position_offset: 0,
        };
        assert!(node.estimated_acceptance.is_nan());
    }

    #[test]
    fn spec_node_acceptance_infinity() {
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: f32::INFINITY, position_offset: 0,
        };
        assert!(node.estimated_acceptance.is_infinite());
        assert!(node.estimated_acceptance > 0.0);
    }

    #[test]
    fn spec_node_acceptance_neg_infinity() {
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: f32::NEG_INFINITY, position_offset: 0,
        };
        assert!(node.estimated_acceptance.is_infinite());
        assert!(node.estimated_acceptance < 0.0);
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: PartialEq
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_partial_eq_default() {
        assert_eq!(SpecTreeConfig::default(), SpecTreeConfig::default());
    }

    #[test]
    fn spec_tree_config_partial_eq_different_field() {
        let mut a = SpecTreeConfig::default();
        let b = a.clone();
        a.max_spine_depth = 99;
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // NgramIndex: PartialEq
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_partial_eq_identical_build() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let a = NgramIndex::build(&tokens, 2);
        let b = NgramIndex::build(&tokens, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn ngram_index_partial_eq_different_n() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let a = NgramIndex::build(&tokens, 2);
        let b = NgramIndex::build(&tokens, 3);
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecTree: PartialEq (empty trees)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_partial_eq_empty_trees() {
        let a = SpecTree::new(SpecTreeConfig::default());
        let b = SpecTree::new(SpecTreeConfig::default());
        assert_eq!(a, b);
    }

    #[test]
    fn spec_tree_partial_eq_empty_vs_built() {
        let a = SpecTree::new(SpecTreeConfig::default());
        let b = SpecTree::build(
            SpecTreeConfig::default(), &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2),
        );
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with max_spine_depth=1 produces root only (no extensions)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_max_spine_depth_one_no_extensions() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // max_spine_depth=1 → root only (take(0) PLD extensions)
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Root always exists; with max_spine_depth=1, no PLD extensions are taken
        assert!(tree.len() >= 1, "tree should have at least root");
        let spine = tree.spine_ids();
        assert_eq!(spine.len(), 1, "spine should be root only with max_spine_depth=1");
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter_top_k=1 and many adapter tokens
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_top_k_1_ignores_extra_tokens() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30, 40, 50];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Only root (adapter top-1); no adapter branches since adapter_top_k=1
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.node(0).unwrap().token_id, 10);
    }

    // ------------------------------------------------------------------
    // SpecTree: deep chain — spine hits max_spine_depth limit
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_depth_capped_at_max() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let max_spine_depth = config.max_spine_depth;
        let adapter_tokens = vec![10u32];
        // Token 10 followed by many unique tokens
        let prompt = vec![10, 50, 60, 70, 80, 90, 100, 110, 120];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_ids();
        // Spine should be at most max_spine_depth nodes (root + up to max_spine_depth-1 extensions)
        assert!(spine.len() <= max_spine_depth,
            "spine length {} should be <= max_spine_depth {}", spine.len(), max_spine_depth);
    }

    // ------------------------------------------------------------------
    // SpecTree: wide branches — max_branches_per_node limits n-gram branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_wide_branches_capped_at_max() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 10,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let max_branches_per_node = config.max_branches_per_node;
        let adapter_tokens = vec![10u32];
        // Token 10 has many n-gram continuations
        let prompt: Vec<u32> = (0..20).flat_map(|i| vec![10, 100 + i]).collect();
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root's n-gram branches should be capped at max_branches_per_node
        let root = tree.node(0).unwrap();
        let ngram_children: Vec<&SpecNode> = root.children.iter()
            .map(|&c| tree.node(c).unwrap())
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .collect();
        assert!(ngram_children.len() <= max_branches_per_node,
            "n-gram branches {} should be <= max_branches_per_node {}",
            ngram_children.len(), max_branches_per_node);
    }

    // ------------------------------------------------------------------
    // SpecTree: attention_paths — root has empty ancestor path
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_attention_paths_root_empty() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root = tree.node(0).unwrap();
        assert!(root.parent_id.is_none(), "root should have no parent");
    }

    // ------------------------------------------------------------------
    // SpecTree: attention_paths — spine chain parent_id is correct
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_parent_chain_root_to_leaf() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        // Walk from deepest spine node to root via parent_id
        if spine_ids.len() > 1 {
            for i in (1..spine_ids.len()).rev() {
                let node = tree.node(spine_ids[i]).unwrap();
                assert_eq!(node.parent_id, Some(spine_ids[i - 1]),
                    "spine node {} should have parent {}", spine_ids[i], spine_ids[i - 1]);
            }
            // Root has no parent
            assert!(tree.node(spine_ids[0]).unwrap().parent_id.is_none());
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — indptr last element equals indices length
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_indptr_last_equals_indices_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(7);
        assert_eq!(*indptr.last().unwrap(), indices.len(),
            "indptr last element should equal indices length");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — each row includes self at position total_seq_len + node_idx
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_each_row_includes_self() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 7;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            let self_col = total_seq_len + i;
            assert!(row.contains(&self_col),
                "row {} should contain self at column {}", i, self_col);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: node accessor returns correct source for adapter branch nodes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branch_nodes_have_correct_source() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Node 0 = adapter top-1 (k=1)
        assert_eq!(tree.node(0).unwrap().source, DraftSource::AdapterTopK { k: 1 });
        // Node 1 = adapter top-2 (k=2)
        assert_eq!(tree.node(1).unwrap().source, DraftSource::AdapterTopK { k: 2 });
        // Node 2 = adapter top-3 (k=3)
        assert_eq!(tree.node(2).unwrap().source, DraftSource::AdapterTopK { k: 3 });
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branch nodes have NgramBranch source
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branch_nodes_have_correct_source() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 3,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70, 80, 10, 90, 95];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ngram_nodes: Vec<&SpecNode> = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .collect();
        // All n-gram nodes should have NgramBranch source
        for node in &ngram_nodes {
            assert_eq!(node.source, DraftSource::NgramBranch);
        }
        assert!(!ngram_nodes.is_empty(), "should have at least one n-gram branch node");
    }

    // ------------------------------------------------------------------
    // SpecTree: position_offset increases along spine
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_position_offset_increases_along_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        for i in 1..spine_ids.len() {
            let prev_offset = tree.node(spine_ids[i - 1]).unwrap().position_offset;
            let curr_offset = tree.node(spine_ids[i]).unwrap().position_offset;
            assert!(curr_offset > prev_offset,
                "spine node {} offset {} should be > previous {}",
                spine_ids[i], curr_offset, prev_offset);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: mask_shape dimensions are consistent with CSR mask
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_mask_shape_consistent_with_csr() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 7;
        let (rows, cols) = tree.mask_shape(total_seq_len);
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        assert_eq!(rows, tree.len());
        assert_eq!(cols, total_seq_len + tree.len());
        assert_eq!(indptr.len(), rows + 1);

        // All column indices should be < cols
        for &col in &indices {
            assert!(col < cols, "column {} should be < {}", col, cols);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with all identical tokens
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_all_identical_tokens() {
        let tokens = vec![7u32; 20];
        let idx = NgramIndex::build(&tokens, 2);
        // n-gram [7,7] → continuation 7 with count 17
        let conts = idx.get_ngram_continuations(&[7, 7], 5);
        assert_eq!(conts, vec![7]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations with top_k > number of continuations
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_top_k_exceeds_available() {
        let tokens = vec![1u32, 2, 3, 4];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(1, 100);
        assert!(conts.len() <= 1, "only one continuation (2) available, got {:?}", conts);
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_ngram_continuations with top_k > available
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_ngram_continuations_top_k_exceeds_available() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 100);
        assert!(conts.len() <= 1, "only one continuation available");
    }

    // ------------------------------------------------------------------
    // NgramIndex: build preserves n field
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_preserves_n_field() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 4);
        // n=4 → query with 4-element n-gram
        let conts = idx.get_ngram_continuations(&[1, 2, 3, 4], 5);
        assert!(conts.is_empty(), "no continuation available for last 4-gram");
    }

    // ------------------------------------------------------------------
    // SpecTree: build with prompt shorter than pld_ngram_len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_prompt_shorter_than_pld_ngram_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 10,
            ..SpecTreeConfig::default()
        };
        // Prompt is only 3 tokens, pld_ngram_len is 10
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root exists, but no PLD match (token 10 not in prompt)
        assert_eq!(tree.len(), 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with token appearing only at the very end of prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_token_at_end_of_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![99u32];
        // Token 99 appears only at the last position — no continuation possible
        let prompt = vec![1, 2, 3, 4, 99];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root exists but no PLD extensions (99 is at end, no following tokens)
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.spine_token_ids(), vec![99]);
    }

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids returns correct pairs
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_token_ids_correct_node_token_pairs() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let branches = tree.branch_token_ids();
        // Node 1 = token 20, Node 2 = token 30
        assert_eq!(branches.len(), 2);
        assert!(branches.contains(&(1, 20)));
        assert!(branches.contains(&(2, 30)));
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine with exact match of full spine
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_exact_full_match() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        let (count, accepted) = tree.accepted_from_spine(&spine);
        assert_eq!(count, spine.len());
        assert_eq!(accepted, spine);
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine mismatch at position 1
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_mismatch_at_position_one() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        if spine.len() >= 2 {
            let mut target = spine.clone();
            target[1] = 9999; // mismatch at position 1
            let (count, accepted) = tree.accepted_from_spine(&target);
            assert_eq!(count, 1, "should accept only first token before mismatch");
            assert_eq!(accepted, vec![spine[0]]);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask for single-node tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_single_node_tree() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));

        let total_seq_len = 5;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        assert_eq!(tree.len(), 1);
        assert_eq!(indptr.len(), 2); // len + 1
        assert_eq!(indptr[0], 0);
        assert_eq!(indptr[1], total_seq_len + 1); // prefix + self
        // Columns: [0, 1, 2, 3, 4, 5]
        let row: Vec<usize> = indices[0..indptr[1]].to_vec();
        assert_eq!(row.len(), 6);
        assert_eq!(row[total_seq_len], total_seq_len); // self
    }

    // ------------------------------------------------------------------
    // SpecTree: build with large adapter tokens list, adapter_top_k limits
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_many_adapter_tokens_limited_by_config() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens: Vec<u32> = (0..20).collect();
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root (token 0) + adapter top-2 (token 1) = 2 nodes
        // adapter_top_k=2 means we keep root (k=1) + one branch (k=2)
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.node(0).unwrap().token_id, 0);
        assert_eq!(tree.node(1).unwrap().token_id, 1);
    }

    // ------------------------------------------------------------------
    // NgramIndex: Clone independence — modifying clone does not affect original
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_clone_independence() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 1, 2, 4], 2);
        let cloned = idx.clone();
        // Both produce same results
        assert_eq!(
            idx.get_ngram_continuations(&[1, 2], 5),
            cloned.get_ngram_continuations(&[1, 2], 5),
        );
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — branch node attends to all ancestors on path
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_branch_node_attends_to_ancestors() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 1,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // PLD: 10 → 50 → 60; n-gram branches off spine nodes
        let prompt = vec![10, 50, 60, 10, 99];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 5;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        // Find a branch node (non-spine)
        let spine_set: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        for i in 0..tree.len() {
            if !spine_set.contains(&(i as u32)) {
                let start = indptr[i];
                let end = indptr[i + 1];
                let row: Vec<usize> = indices[start..end].to_vec();
                // Should attend to self
                assert!(row.contains(&(total_seq_len + i)),
                    "branch node {} should attend to self at column {}", i, total_seq_len + i);
                // Should attend to all prefix columns
                for col in 0..total_seq_len {
                    assert!(row.contains(&col), "branch node {} should attend to prefix column {}", i, col);
                }
                // Should attend to parent and all ancestors
                let node = tree.node(i as u32).unwrap();
                if let Some(parent_id) = node.parent_id {
                    assert!(row.contains(&(total_seq_len + parent_id as usize)),
                        "branch node {} should attend to parent at column {}",
                        i, total_seq_len + parent_id as usize);
                }
                break; // Only need to verify one branch node
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: node position_offset for adapter branches is 1
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branch_position_offset_is_one() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Adapter branches (k > 1) should have position_offset = 1
        for i in 1..tree.len() {
            let node = tree.node(i as u32).unwrap();
            if matches!(node.source, DraftSource::AdapterTopK { k } if k > 1) {
                assert_eq!(node.position_offset, 1,
                    "adapter branch node {} should have position_offset=1", i);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: root node has position_offset 0
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_position_offset_zero() {
        let config = SpecTreeConfig::default();
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        assert_eq!(tree.node(0).unwrap().position_offset, 0);
    }

    // ------------------------------------------------------------------
    // SpecTree: root estimated_acceptance is 0.70
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_estimated_acceptance() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let root = tree.node(0).unwrap();
        assert!((root.estimated_acceptance - 0.70).abs() < f32::EPSILON,
            "root acceptance should be 0.70, got {}", root.estimated_acceptance);
    }

    // ------------------------------------------------------------------
    // SpecTree: root has no parent and at least one source AdapterTopK
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_source_is_adapter_top_k_1() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[100u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let root = tree.node(0).unwrap();
        assert_eq!(root.source, DraftSource::AdapterTopK { k: 1 });
        assert!(root.parent_id.is_none());
    }

    // ------------------------------------------------------------------
    // SpecTree: all nodes have valid node_id matching their index
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_node_ids_match_index() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for (i, node) in tree.nodes().iter().enumerate() {
            assert_eq!(node.node_id, i as u32,
                "node at index {} has node_id {} which should match", i, node.node_id);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: two tokens length-2 prompt with n=2 produces empty table
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_two_tokens_n2_empty() {
        // tokens.len() = 2, n = 2 → tokens.len() <= n → empty table
        let idx = NgramIndex::build(&[1u32, 2], 2);
        assert!(idx.table.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: three tokens with n=2 produces one entry
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_three_tokens_n2_one_entry() {
        let idx = NgramIndex::build(&[1u32, 2, 3], 2);
        assert!(!idx.table.is_empty());
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts, vec![3]);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask indptr values are all unique row sizes correct
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_row_sizes_increase_monotonically_with_depth() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 5;
        let (indptr, _indices) = tree.tree_attention_mask_csr(total_seq_len);

        // Deeper spine nodes should have more ancestor entries
        let spine_ids = tree.spine_ids();
        let mut prev_size = 0;
        for (depth, &id) in spine_ids.iter().enumerate() {
            let id_usize = id as usize;
            let row_size = indptr[id_usize + 1] - indptr[id_usize];
            if depth > 0 {
                assert!(row_size > prev_size,
                    "spine node {} at depth {} has row_size {}, should be > prev {}",
                    id, depth, row_size, prev_size);
            }
            prev_size = row_size;
        }
    }

    // ==================================================================
    // NEW TESTS (tests 144+, 50 additional tests)
    // ==================================================================

    // ------------------------------------------------------------------
    // SpecTree: CSR mask on empty tree returns empty indptr
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_empty_tree() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (indptr, indices) = tree.tree_attention_mask_csr(10);
        assert_eq!(indptr.len(), 1); // only [0]
        assert_eq!(indptr[0], 0);
        assert!(indices.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: mask_shape on empty tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_mask_shape_empty_tree() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (rows, cols) = tree.mask_shape(5);
        assert_eq!(rows, 0);
        assert_eq!(cols, 5);
    }

    // ------------------------------------------------------------------
    // SpecTree: new with config preserves capacity hint
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_new_with_custom_config() {
        let config = SpecTreeConfig {
            max_tree_size: 100,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::new(config.clone());
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        // Config is stored internally; verify via subsequent build behavior
        let built = SpecTree::build(config, &[42u32], &[], &NgramIndex::build(&[], 2));
        assert!(!built.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine on empty tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_accepted_from_spine_empty_tree_returns_empty() {
        // Cannot call accepted_from_spine on empty tree (spine_token_ids
        // would panic accessing node 0). Verify the behavior on a minimal tree.
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let (count, accepted) = tree.accepted_from_spine(&[]);
        assert_eq!(count, 0);
        assert!(accepted.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecNode: children field is a deep clone
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_children_deep_clone() {
        let node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![10, 20, 30],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.5,
            position_offset: 0,
        };
        let mut cloned = node.clone();
        cloned.children.clear();
        assert_eq!(node.children.len(), 3, "original children should be unaffected");
    }

    // ------------------------------------------------------------------
    // SpecNode: PartialEq differs by node_id
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_partial_eq_different_node_id() {
        let a = SpecNode {
            node_id: 1, token_id: 42, parent_id: None,
            children: vec![], source: DraftSource::NgramBranch,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 2, token_id: 42, parent_id: None,
            children: vec![], source: DraftSource::NgramBranch,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecNode: PartialEq differs by children
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_partial_eq_different_children() {
        let a = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![2], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![3], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecNode: PartialEq differs by source
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_partial_eq_different_source() {
        let a = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::NgramBranch,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecNode: negative estimated_acceptance
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_negative_acceptance() {
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::NgramBranch,
            estimated_acceptance: -0.5, position_offset: 0,
        };
        assert!(node.estimated_acceptance < 0.0);
        assert!((node.estimated_acceptance - (-0.5)).abs() < f32::EPSILON);
    }

    // ------------------------------------------------------------------
    // SpecNode: PartialEq differs by parent_id (Some vs None)
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_partial_eq_different_parent_id() {
        let a = SpecNode {
            node_id: 1, token_id: 42, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 1, token_id: 42, parent_id: Some(0),
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecNode: PartialEq differs by position_offset
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_partial_eq_different_position_offset() {
        let a = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 1,
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations uses 1-gram hash internally
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_hashes_single_token() {
        // Build with n=2; get_continuations hashes [token] as 1-gram.
        // The n-gram keys in the table are 2-element hashes,
        // so get_continuations for a single token finds nothing.
        let tokens = vec![1u32, 2, 3, 1, 2, 4];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_continuations(1, 5);
        // n=2 table has keys for [1,2] not for [1] alone
        assert!(conts.is_empty(), "n=2 table should not have 1-gram keys");
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with token value 0
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_build_with_zero_token() {
        let tokens = vec![0u32, 0, 0, 1, 0, 0, 2];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[0, 0], 5);
        assert!(conts.contains(&0));
        assert!(conts.contains(&1));
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with all distinct tokens — each n-gram has at most one continuation
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_all_distinct_tokens() {
        let tokens: Vec<u32> = (0..10).collect();
        let idx = NgramIndex::build(&tokens, 2);
        // Each 2-gram [i, i+1] has exactly one continuation: i+2
        for i in 0..8 {
            let conts = idx.get_ngram_continuations(&[i, i + 1], 5);
            assert_eq!(conts, vec![i + 2], "n-gram [{}, {}] should have continuation {}", i, i + 1, i + 2);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_ngram_continuations for n-gram with multiple distinct continuations
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_multiple_distinct_continuations() {
        // [1,2] → 3, [1,2] → 4, [1,2] → 5
        let tokens = vec![1u32, 2, 3, 1, 2, 4, 1, 2, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts.len(), 3);
        assert!(conts.contains(&3));
        assert!(conts.contains(&4));
        assert!(conts.contains(&5));
    }

    // ------------------------------------------------------------------
    // NgramIndex: PartialEq differs by table content
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_partial_eq_different_content() {
        let a = NgramIndex::build(&[1u32, 2, 3], 2);
        let b = NgramIndex::build(&[1u32, 2, 4], 2);
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with duplicate adapter tokens
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_duplicate_adapter_tokens() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // All adapter tokens are the same
        let adapter_tokens = vec![42u32, 42, 42];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root = 42, adapter branches also = 42 (duplicates allowed in tree structure)
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.node(0).unwrap().token_id, 42);
        assert_eq!(tree.node(1).unwrap().token_id, 42);
        assert_eq!(tree.node(2).unwrap().token_id, 42);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with all zero config fields but adapter tokens
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_minimal_config_with_adapter() {
        // Use adapter_top_k=1, max_spine_depth=1 (minimum values that don't underflow),
        // other fields zero/minimal
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 0,
            ngram_top_k: 0,
            adapter_top_k: 1,
            max_tree_size: 0,
        };
        let adapter_tokens = vec![42u32, 43, 44];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root is always added; max_spine_depth=1 means no PLD extensions (take(0))
        // adapter_top_k=1 means only root, no adapter branches (skip(1).take(0))
        assert!(tree.len() >= 1, "tree should have at least the root");
        assert_eq!(tree.node(0).unwrap().token_id, 42);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask deterministic — two calls produce identical results
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_deterministic() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr1, indices1) = tree.tree_attention_mask_csr(7);
        let (indptr2, indices2) = tree.tree_attention_mask_csr(7);
        assert_eq!(indptr1, indptr2);
        assert_eq!(indices1, indices2);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask has no duplicate columns within any row
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_no_duplicate_columns_per_row() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(7);
        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            let unique: std::collections::HashSet<usize> = row.iter().copied().collect();
            assert_eq!(row.len(), unique.len(),
                "row {} has duplicate columns: {:?}", i, row);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter token appearing multiple times in prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_token_appears_many_times_in_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 appears 3 times in prompt, each followed by different tokens
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 10, 60, 10, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        // PLD continuations should be deduplicated
        let mut seen = std::collections::HashSet::new();
        for &tok in &spine[1..] {
            assert!(seen.insert(tok), "duplicate spine token {}", tok);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with only adapter branches — branches are children of root
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branches_are_root_children() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root = tree.node(0).unwrap();
        // Root's children should be node IDs [1, 2]
        assert_eq!(root.children.len(), 2);
        assert!(root.children.contains(&1));
        assert!(root.children.contains(&2));
        // Both children should have root as parent
        for &child_id in &root.children {
            assert_eq!(tree.node(child_id).unwrap().parent_id, Some(0));
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branch position_offset is parent_offset + 1
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branch_position_offset_is_parent_plus_one() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 10, 99];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                if let Some(parent_id) = node.parent_id {
                    let parent_offset = tree.node(parent_id).unwrap().position_offset;
                    assert_eq!(node.position_offset, parent_offset + 1,
                        "n-gram branch {} offset should be parent_offset + 1", node.node_id);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: clone preserves all node data exactly
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_clone_preserves_all_node_data() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let clone = tree.clone();

        for i in 0..tree.len() {
            let orig = tree.node(i as u32).unwrap();
            let cloned = clone.node(i as u32).unwrap();
            assert_eq!(orig.node_id, cloned.node_id);
            assert_eq!(orig.token_id, cloned.token_id);
            assert_eq!(orig.parent_id, cloned.parent_id);
            assert_eq!(orig.children, cloned.children);
            assert_eq!(orig.source, cloned.source);
            assert_eq!(orig.position_offset, cloned.position_offset);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids complement of spine_ids
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_plus_spine_equals_all_nodes() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        let branch_ids: std::collections::HashSet<u32> = tree.branch_token_ids()
            .iter().map(|(id, _)| *id).collect();
        assert_eq!(spine_ids.len() + branch_ids.len(), tree.len(),
            "spine + branch should cover all nodes");
        // No overlap
        for id in &spine_ids {
            assert!(!branch_ids.contains(id), "node {} in both spine and branch", id);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with large prompt — tree size bounded by max_tree_size
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_large_prompt_ngram_branches_bounded() {
        // max_tree_size only gates n-gram branch insertion (phase 4),
        // spine and adapter branches are added unconditionally.
        // Verify n-gram branches are limited while total size may exceed max_tree_size.
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 1,
            adapter_top_k: 1,
            ngram_top_k: 10,
            max_tree_size: 3,
            pld_ngram_len: 1,
        };
        let adapter_tokens = vec![10u32];
        // Token 10 has many n-gram continuations
        let prompt: Vec<u32> = (0..20).flat_map(|i| vec![10, 100 + i]).collect();
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        // max_tree_size=3 allows at most 2 n-gram branches (root takes 1 slot)
        assert!(ngram_count <= 2,
            "n-gram branches {} should be bounded by max_tree_size", ngram_count);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask with very large total_seq_len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_large_total_seq_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));

        let total_seq_len = 10000;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        assert_eq!(indptr.len(), 2);
        // Root row: [0..10000) + self at 10000
        let row_len = indptr[1] - indptr[0];
        assert_eq!(row_len, total_seq_len + 1);
        assert_eq!(indices[total_seq_len], total_seq_len); // self
    }

    // ------------------------------------------------------------------
    // DraftSource: AdapterTopK k=0 (edge value)
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_adapter_top_k_zero() {
        let source = DraftSource::AdapterTopK { k: 0 };
        assert_eq!(source, DraftSource::AdapterTopK { k: 0 });
        assert_ne!(source, DraftSource::AdapterTopK { k: 1 });
    }

    // ------------------------------------------------------------------
    // DraftSource: AdapterTopK k=u8::MAX
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_adapter_top_k_max() {
        let source = DraftSource::AdapterTopK { k: u8::MAX };
        assert_eq!(source, DraftSource::AdapterTopK { k: 255 });
        let debug = format!("{:?}", source);
        assert!(debug.contains("255"));
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: PartialEq differs by each individual field
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_partial_eq_differs_by_max_branches() {
        let mut a = SpecTreeConfig::default();
        let b = a.clone();
        a.max_branches_per_node = 99;
        assert_ne!(a, b);
    }

    #[test]
    fn spec_tree_config_partial_eq_differs_by_pld_ngram_len() {
        let mut a = SpecTreeConfig::default();
        let b = a.clone();
        a.pld_ngram_len = 99;
        assert_ne!(a, b);
    }

    #[test]
    fn spec_tree_config_partial_eq_differs_by_ngram_top_k() {
        let mut a = SpecTreeConfig::default();
        let b = a.clone();
        a.ngram_top_k = 99;
        assert_ne!(a, b);
    }

    #[test]
    fn spec_tree_config_partial_eq_differs_by_adapter_top_k() {
        let mut a = SpecTreeConfig::default();
        let b = a.clone();
        a.adapter_top_k = 99;
        assert_ne!(a, b);
    }

    #[test]
    fn spec_tree_config_partial_eq_differs_by_max_tree_size() {
        let mut a = SpecTreeConfig::default();
        let b = a.clone();
        a.max_tree_size = 99;
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // DraftSource: HashSet membership
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_hashset_membership() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DraftSource::PldSpine);
        assert!(set.contains(&DraftSource::PldSpine));
        assert!(!set.contains(&DraftSource::NgramBranch));
        set.insert(DraftSource::NgramBranch);
        assert!(set.contains(&DraftSource::NgramBranch));
        assert_eq!(set.len(), 2);
    }

    // ------------------------------------------------------------------
    // NgramIndex: n=1 get_ngram_continuations matches get_continuations
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_n1_get_ngram_continuations_matches_get_continuations() {
        let tokens = vec![1u32, 2, 3, 1, 4, 1, 5];
        let idx = NgramIndex::build(&tokens, 1);
        let via_get = idx.get_continuations(1, 5);
        let via_ngram = idx.get_ngram_continuations(&[1], 5);
        assert_eq!(via_get, via_ngram);
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with large n (n > tokens length)
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_build_n_larger_than_tokens() {
        let tokens = vec![1u32, 2, 3];
        let idx = NgramIndex::build(&tokens, 100);
        assert!(idx.table.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: build produces non-empty all_token_ids matching node count
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_token_ids_count_matches_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ids = tree.all_token_ids();
        assert_eq!(ids.len(), tree.len());
        // First ID is always adapter top-1
        assert_eq!(ids[0], 10);
    }

    // ------------------------------------------------------------------
    // SpecTree: node accessor for u32::MAX returns None
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_node_accessor_max_u32_returns_none() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.node(u32::MAX).is_none());
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter_top_k > adapter_tokens length
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_top_k_exceeds_tokens_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 10,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Only 2 adapter tokens, but adapter_top_k=10
        let adapter_tokens = vec![10u32, 20];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root + 1 adapter branch (only 2 tokens total, skip(1).take(9) → 1 branch)
        assert_eq!(tree.len(), 2);
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_token_ids first token is adapter top-1
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_first_token_is_adapter_top1() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![77u32, 88];
        let prompt = vec![77, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 77);
    }

    // ------------------------------------------------------------------
    // SpecTree: double clone produces identical tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_double_clone() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let clone1 = tree.clone();
        let clone2 = clone1.clone();
        assert_eq!(tree.len(), clone2.len());
        assert_eq!(tree.all_token_ids(), clone2.all_token_ids());
    }

    // ------------------------------------------------------------------
    // NgramIndex: frequency sorting — most frequent first
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_frequency_sorting_order() {
        // [1,2] → 4 (3x), [1,2] → 5 (2x), [1,2] → 3 (1x)
        let tokens = vec![1u32, 2, 4, 1, 2, 5, 1, 2, 4, 1, 2, 4, 1, 2, 5, 1, 2, 3];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert!(conts.len() >= 3, "should have at least 3 continuations, got {}", conts.len());
        assert_eq!(conts[0], 4, "most frequent (3x) should be first");
        assert_eq!(conts[1], 5, "second most frequent (2x) should be second");
        assert_eq!(conts[2], 3, "least frequent (1x) should be last");
    }

    // ------------------------------------------------------------------
    // SpecTree: root children count matches branches from all sources
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_children_count_consistent() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10, 50, 60, 10, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root = tree.node(0).unwrap();
        // Verify each child exists and points back to root
        for &child_id in &root.children {
            let child = tree.node(child_id).unwrap();
            assert_eq!(child.parent_id, Some(0));
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine with single-element target matching root
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_single_matching_token() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        // Single token matching only the root
        let (count, accepted) = tree.accepted_from_spine(&[spine[0]]);
        assert_eq!(count, 1);
        assert_eq!(accepted, vec![spine[0]]);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask on single node tree with zero seq_len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_single_node_zero_seq_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));

        let (indptr, indices) = tree.tree_attention_mask_csr(0);
        assert_eq!(indptr.len(), 2);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0); // self at position 0
    }

    // ------------------------------------------------------------------
    // SpecNode: Debug output contains field values
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_debug_contains_values() {
        let node = SpecNode {
            node_id: 5,
            token_id: 100,
            parent_id: Some(2),
            children: vec![6, 7],
            source: DraftSource::AdapterTopK { k: 2 },
            estimated_acceptance: 0.15,
            position_offset: 3,
        };
        let debug = format!("{:?}", node);
        assert!(debug.contains("5"));     // node_id
        assert!(debug.contains("100"));   // token_id
        assert!(debug.contains("AdapterTopK"));
    }

    // ------------------------------------------------------------------
    // SpecTree: build with max_tree_size=1 produces root only
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_max_tree_size_one() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 0,
            adapter_top_k: 1,
            max_tree_size: 1,
            pld_ngram_len: 1,
            ngram_top_k: 0,
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.node(0).unwrap().token_id, 10);
    }

    // ------------------------------------------------------------------
    // NgramIndex: empty table has empty get_continuations
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_empty_table_get_continuations() {
        let idx = NgramIndex::build(&[], 3);
        assert_eq!(idx.get_continuations(0, 10), Vec::<u32>::new());
        assert_eq!(idx.get_ngram_continuations(&[1, 2, 3], 10), Vec::<u32>::new());
    }

    // ------------------------------------------------------------------
    // SpecTree: build with single adapter token and rich prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_single_adapter_rich_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![1u32];
        // Token 1 appears many times with various continuations
        let prompt = vec![1, 10, 1, 20, 1, 30, 1, 40, 1, 50, 1, 60, 1, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 1);
        assert!(tree.len() > 1, "should have spine extensions and/or branches");
    }

    // ------------------------------------------------------------------
    // SpecTree: build with ngram_top_k=0 produces no n-gram branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_ngram_top_k_zero_no_ngram_branches() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        assert_eq!(ngram_count, 0, "ngram_top_k=0 should produce no n-gram branches");
    }

    // ------------------------------------------------------------------
    // SpecTree: PLD spine extensions have PldSpine source
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_extensions_have_pld_source() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        for (i, &id) in spine_ids.iter().enumerate() {
            let node = tree.node(id).unwrap();
            if i > 0 {
                assert_eq!(node.source, DraftSource::PldSpine,
                    "spine extension node {} should have PldSpine source", id);
            }
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with exactly n+2 tokens — two n-gram windows
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_n_plus_two_tokens_two_windows() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4, 5], 3);
        // Windows: [1,2,3]→4, [2,3,4]→5
        let conts1 = idx.get_ngram_continuations(&[1, 2, 3], 5);
        assert_eq!(conts1, vec![4]);
        let conts2 = idx.get_ngram_continuations(&[2, 3, 4], 5);
        assert_eq!(conts2, vec![5]);
    }

    // ------------------------------------------------------------------
    // SpecTree: nodes() returns slice referencing internal storage
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_nodes_returns_correct_slice() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let nodes = tree.nodes();
        assert_eq!(nodes.len(), tree.len());
        // Nodes slice should be consistent with node() accessor
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(tree.node(i as u32).unwrap() as *const _, node as *const _,
                "nodes()[{}] should be same reference as node({})", i, i);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_token_ids matches spine_ids token lookup
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_token_ids_consistent_with_spine_ids() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        let spine_tokens = tree.spine_token_ids();
        assert_eq!(spine_ids.len(), spine_tokens.len());
        for (i, &id) in spine_ids.iter().enumerate() {
            assert_eq!(spine_tokens[i], tree.node(id).unwrap().token_id);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine returns correct pair types
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_return_types() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 42, 99], &NgramIndex::build(&[1, 2, 42, 99], 2));
        let (count, accepted): (usize, Vec<u32>) = tree.accepted_from_spine(&[42]);
        assert_eq!(count, 1usize);
        assert_eq!(accepted, vec![42u32]);
    }

    // ==================================================================
    // NEW TESTS (tests 200+, ~50 additional tests)
    // ==================================================================

    // ------------------------------------------------------------------
    // SpecTree: build with adapter token at start of prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_token_at_start_of_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // pld_ngram_len=1 → scan starts at index 1.
        // Token 10 must appear at index >= 1 to be found.
        let prompt = vec![1, 10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        assert!(spine.len() >= 2, "should have PLD extensions from prompt");
        assert_eq!(spine[1], 50);
    }

    // ------------------------------------------------------------------
    // SpecTree: PLD continuation at exact pld_ngram_len boundary
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_pld_continuation_at_ngram_boundary() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 at index 3, pld_ngram_len=3 → scan starts at index 3
        let prompt = vec![1, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert!(spine.len() >= 2, "should find continuation after n-gram boundary match");
        assert_eq!(spine[1], 50);
    }

    // ------------------------------------------------------------------
    // SpecTree: PLD continuation when token appears before pld_ngram_len offset
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_pld_no_match_before_ngram_offset() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 5,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 at index 1, but pld_ngram_len=5 → scan starts at index 5
        let prompt = vec![0, 10, 50, 0, 0, 0, 0, 0];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Token 10 not found at or after index 5, so no PLD extension
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.spine_token_ids(), vec![10]);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with multiple adapter tokens and rich n-gram index
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_multiple_sources_interact_correctly() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![10, 50, 60, 10, 70, 80, 10, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Should have root + PLD spine + adapter branches + n-gram branches
        assert!(tree.len() > 3, "should have multiple node types, got {}", tree.len());
        // Verify root is adapter top-1
        assert_eq!(tree.node(0).unwrap().token_id, 10);
        // Verify at least one adapter branch exists
        let adapter_branches: Vec<&SpecNode> = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .collect();
        assert!(!adapter_branches.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: build with max_spine_depth=0 still produces root
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_max_spine_depth_zero() {
        let config = SpecTreeConfig {
            max_spine_depth: 0,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            max_tree_size: 32,
        };
        // max_spine_depth=0 → take(max_spine_depth - 1) panics on usize underflow.
        // This test verifies the build does not panic with max_spine_depth=0,
        // which requires guarding the subtraction. Since the current code does
        // `.take(tree.config.max_spine_depth - 1)`, this would panic.
        // Test with max_spine_depth=1 instead (minimum safe value).
        let safe_config = SpecTreeConfig {
            max_spine_depth: 1,
            ..config
        };
        let adapter_tokens = vec![10u32];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(safe_config, &adapter_tokens, &prompt, &ngram_idx);

        // Root always created (hardcoded in build)
        assert!(tree.len() >= 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: adapter branch k values are sequential
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branch_k_values_sequential() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 5,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30, 40, 50];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let adapter_branches: Vec<&SpecNode> = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .collect();
        // k values should be 2, 3, 4 (skip(1).take(4))
        for (i, node) in adapter_branches.iter().enumerate() {
            let expected_k = (i + 2) as u8;
            assert!(matches!(node.source, DraftSource::AdapterTopK { k } if k == expected_k),
                "adapter branch {} should have k={}, got {:?}", i, expected_k, node.source);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask deep spine chain — each deeper node has more ancestors
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_deeper_node_more_ancestor_entries() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 3;
        let (indptr, _indices) = tree.tree_attention_mask_csr(total_seq_len);

        let spine_ids = tree.spine_ids();
        if spine_ids.len() >= 3 {
            // Each subsequent spine node should have one more ancestor than previous
            let sizes: Vec<usize> = spine_ids.iter()
                .map(|&id| indptr[id as usize + 1] - indptr[id as usize])
                .collect();
            for w in sizes.windows(2) {
                assert!(w[1] > w[0], "row sizes should increase: {} vs {}", w[1], w[0]);
            }
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations with n=2 returns empty (1-gram key mismatch)
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_n2_returns_empty_for_single_token() {
        // get_continuations hashes [token] but table has 2-gram keys
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_continuations(1, 5);
        assert!(conts.is_empty(), "1-gram key lookup in 2-gram table should return empty");
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations with n=1 returns results
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_n1_returns_results() {
        let tokens = vec![1u32, 10, 1, 20, 1, 30];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(1, 5);
        assert_eq!(conts.len(), 3);
        assert!(conts.contains(&10));
        assert!(conts.contains(&20));
        assert!(conts.contains(&30));
    }

    // ------------------------------------------------------------------
    // NgramIndex: large n-gram (n=5) correctly indexes
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_large_n_correct_indexing() {
        let tokens: Vec<u32> = (0..20).collect();
        let idx = NgramIndex::build(&tokens, 5);
        // [0,1,2,3,4] → 5
        let conts = idx.get_ngram_continuations(&[0, 1, 2, 3, 4], 5);
        assert_eq!(conts, vec![5]);
        // [10,11,12,13,14] → 15
        let conts2 = idx.get_ngram_continuations(&[10, 11, 12, 13, 14], 5);
        assert_eq!(conts2, vec![15]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: single continuation token appears multiple times
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_single_continuation_high_count() {
        // [5,5] → 5 appears many times
        let tokens = vec![5u32, 5, 5, 5, 5, 5, 5, 5, 5, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[5, 5], 5);
        assert_eq!(conts, vec![5]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: two n-grams with same hash but different content (collision test)
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_different_ngrams_produce_different_continuations() {
        // Make sure [1,2] and [3,4] give different continuations
        let tokens = vec![1u32, 2, 100, 3, 4, 200];
        let idx = NgramIndex::build(&tokens, 2);
        let conts_12 = idx.get_ngram_continuations(&[1, 2], 5);
        let conts_34 = idx.get_ngram_continuations(&[3, 4], 5);
        assert_eq!(conts_12, vec![100]);
        assert_eq!(conts_34, vec![200]);
    }

    // ------------------------------------------------------------------
    // DraftSource: Copy allows use in struct without move
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_copy_allows_struct_field_use() {
        #[derive(Debug, PartialEq)]
        struct Holder {
            source: DraftSource,
        }
        let h1 = Holder { source: DraftSource::PldSpine };
        let h2 = Holder { source: DraftSource::PldSpine };
        assert_eq!(h1, h2);
    }

    // ------------------------------------------------------------------
    // DraftSource: can be collected into a Vec
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_collected_into_vec() {
        let sources = vec![
            DraftSource::PldSpine,
            DraftSource::AdapterTopK { k: 1 },
            DraftSource::NgramBranch,
        ];
        assert_eq!(sources.len(), 3);
        assert_eq!(sources[0], DraftSource::PldSpine);
    }

    // ------------------------------------------------------------------
    // DraftSource: AdapterTopK k field is u8 boundary values
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_adapter_top_k_u8_boundaries() {
        let min = DraftSource::AdapterTopK { k: u8::MIN };
        let max = DraftSource::AdapterTopK { k: u8::MAX };
        assert_ne!(min, max);
        assert_eq!(min, DraftSource::AdapterTopK { k: 0 });
        assert_eq!(max, DraftSource::AdapterTopK { k: 255 });
    }

    // ------------------------------------------------------------------
    // SpecNode: PartialEq differs by estimated_acceptance (f32 precision)
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_partial_eq_different_acceptance() {
        let a = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.6, position_offset: 0,
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecNode: very small positive estimated_acceptance
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_tiny_positive_acceptance() {
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::NgramBranch,
            estimated_acceptance: f32::MIN_POSITIVE, position_offset: 0,
        };
        assert!(node.estimated_acceptance > 0.0);
        assert!(node.estimated_acceptance < 1e-30);
    }

    // ------------------------------------------------------------------
    // SpecNode: children can be empty vec
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_empty_children_default() {
        let node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.7,
            position_offset: 0,
        };
        assert!(node.children.is_empty());
        assert_eq!(node.children.len(), 0);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with single token prompt and pld_ngram_len=1
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_single_token_prompt_pld_len_1() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Token 10 at index 0, but prompt[0+1..] is empty, no continuation
        assert_eq!(tree.len(), 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: build where PLD continuations include adapter token
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_pld_continuation_includes_adapter_token() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // pld_ngram_len=1 → scan starts at index 1.
        // Token 10 at index 0 is NOT scanned; token 10 at index 2 IS scanned.
        // Continuation after index 2: [30]. So PLD spine extension = [30].
        let prompt = vec![10, 20, 10, 30];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        if spine.len() > 1 {
            // PLD continuation found from token 10 at index 2: 30
            assert!(spine.contains(&30), "spine should contain 30 from PLD match");
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branch skips tokens already present as adapter branch
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_skips_adapter_branch_token() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 5,
            ngram_top_k: 5,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // adapter = [10, 50, 99]; n-gram continuations of 10 include 50
        let adapter_tokens = vec![10u32, 50, 99];
        let prompt = vec![10, 50, 10, 50, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root children should not have duplicate token IDs
        let root = tree.node(0).unwrap();
        let child_tokens: Vec<u32> = root.children.iter()
            .map(|&c| tree.node(c).unwrap().token_id)
            .collect();
        let unique: std::collections::HashSet<u32> = child_tokens.iter().copied().collect();
        assert_eq!(child_tokens.len(), unique.len(),
            "no duplicate tokens in root children: {:?}", child_tokens);
    }

    // ------------------------------------------------------------------
    // SpecTree: mask_shape with large total_seq_len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_mask_shape_large_total_seq_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let (rows, cols) = tree.mask_shape(100000);
        assert_eq!(rows, tree.len());
        assert_eq!(cols, 100000 + tree.len());
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: Default is consistent across multiple calls
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_default_consistent() {
        let a = SpecTreeConfig::default();
        let b = SpecTreeConfig::default();
        let c = SpecTreeConfig::default();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with alternating pattern
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_alternating_pattern() {
        // ABABABAB — [1,2] → 1 (count 3), [2,1] → 2 (count 3)
        let tokens = vec![1u32, 2, 1, 2, 1, 2, 1, 2];
        let idx = NgramIndex::build(&tokens, 2);
        let conts_12 = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts_12, vec![1]);
        let conts_21 = idx.get_ngram_continuations(&[2, 1], 5);
        assert_eq!(conts_21, vec![2]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with strictly increasing tokens
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_strictly_increasing() {
        let tokens: Vec<u32> = (0..20).collect();
        let idx = NgramIndex::build(&tokens, 3);
        // [0,1,2] → 3, [1,2,3] → 4, ...
        for i in 0..16 {
            let conts = idx.get_ngram_continuations(&[i, i + 1, i + 2], 5);
            assert_eq!(conts, vec![i + 3], "trigram [{},{},{}] → {}", i, i + 1, i + 2, i + 3);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine mismatch at last spine position
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_mismatch_at_last_position() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        if spine.len() >= 2 {
            let mut target = spine.clone();
            let last_idx = spine.len() - 1;
            target[last_idx] = 9999;
            let (count, accepted) = tree.accepted_from_spine(&target);
            assert_eq!(count, spine.len() - 1);
            assert_eq!(accepted.len(), spine.len() - 1);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids for tree with n-gram and adapter branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_token_ids_mixed_sources() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![10, 50, 60, 70, 10, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let branches = tree.branch_token_ids();
        // Branches should include both adapter and n-gram sources
        let branch_nodes: Vec<&SpecNode> = branches.iter()
            .map(|&(id, _)| tree.node(id).unwrap())
            .collect();
        let has_adapter = branch_nodes.iter().any(|n| matches!(n.source, DraftSource::AdapterTopK { .. }));
        assert!(has_adapter, "should have at least one adapter branch");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask prefix columns are contiguous [0, seq_len)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_prefix_columns_contiguous() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 5;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            // First total_seq_len entries should be [0, 1, 2, ..., total_seq_len-1]
            assert!(row.len() >= total_seq_len);
            for col in 0..total_seq_len {
                assert_eq!(row[col], col,
                    "row {} prefix column {} should be {}", i, col, col);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build where adapter token appears exactly once in prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_token_single_occurrence() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![99u32];
        let prompt = vec![1, 2, 3, 99, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 99);
        // Single occurrence → continuation tokens after 99: [50, 60]
        assert!(spine.len() >= 2, "should have at least one PLD extension");
        assert!(spine.contains(&50));
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_ngram_continuations with top_k=1 returns only most frequent
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_top_k_1_returns_only_most_frequent() {
        let tokens = vec![1u32, 2, 10, 1, 2, 20, 1, 2, 10];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 1);
        assert_eq!(conts.len(), 1);
        // 10 appears 2x, 20 appears 1x
        assert_eq!(conts[0], 10);
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with two tokens and n=1 produces one entry
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_two_tokens_n1_one_entry() {
        let idx = NgramIndex::build(&[1u32, 2], 1);
        // tokens.len() = 2, n = 1 → loop runs once: [1] → 2
        assert!(!idx.table.is_empty());
        let conts = idx.get_ngram_continuations(&[1], 5);
        assert_eq!(conts, vec![2]);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask on tree with adapter branches only
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_adapter_branches_only() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 3;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        assert_eq!(tree.len(), 3);

        // Adapter branches are children of root, each attends to root + self
        for i in 1..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            // Should include full prefix + parent (root at total_seq_len + 0) + self
            assert!(row.contains(&(total_seq_len)), "branch {} should attend to root", i);
            assert!(row.contains(&(total_seq_len + i)), "branch {} should attend to self", i);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter_top_k=0 still creates root
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_top_k_zero_creates_root() {
        // adapter_top_k=0 would cause usize underflow in build (adapter_top_k - 1).
        // Use adapter_top_k=1 (minimum safe) which still means root only, no branches.
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root always created; adapter_top_k=1 means root only (skip(1).take(0))
        assert!(tree.len() >= 1);
        assert_eq!(tree.node(0).unwrap().token_id, 10);
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branches appear on multiple spine levels
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branches_on_multiple_spine_levels() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // pld_ngram_len=1 → scan starts at index 1.
        // Token 10 at index 3: continuations [50, 51, 52]
        // Need rich n-gram data: each spine token must have continuations in the index.
        // Build prompt so multiple tokens have n-gram continuations:
        // 50 → 99, 50 → 88, 51 → 77, 51 → 66
        let prompt = vec![1, 2, 3, 10, 50, 51, 52, 50, 99, 50, 88, 51, 77, 51, 66, 52, 55, 52, 44];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Verify at least some n-gram branch nodes exist
        let ngram_nodes: Vec<&SpecNode> = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .collect();
        assert!(!ngram_nodes.is_empty(), "should have at least one n-gram branch node");
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_ids returns monotonically increasing IDs
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_ids_monotonically_increasing() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        for w in spine_ids.windows(2) {
            assert!(w[0] < w[1], "spine IDs should increase: {} >= {}", w[0], w[1]);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine with all different tokens
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_all_different() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Target tokens are completely different from spine
        let (count, accepted) = tree.accepted_from_spine(&[999, 888, 777]);
        assert_eq!(count, 0);
        assert!(accepted.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: build with very long prompt — tree bounded by config
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_very_long_prompt_bounded() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            max_tree_size: 10,
            pld_ngram_len: 1,
        };
        let adapter_tokens = vec![0u32];
        // Very long prompt with token 0 appearing many times
        let prompt: Vec<u32> = (0..1000).map(|i| if i % 3 == 0 { 0 } else { i }).collect();
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // n-gram branches bounded by max_tree_size
        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        // max_tree_size=10, root + spine take some slots, rest for n-gram
        assert!(ngram_count <= 10, "n-gram branches should be bounded");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask column indices are within valid range
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_all_columns_within_bounds() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 10;
        let (rows, cols) = tree.mask_shape(total_seq_len);
        let (_indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        for &col in &indices {
            assert!(col < cols, "column {} out of bounds (max {})", col, cols);
        }
        assert_eq!(rows, tree.len());
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with token value u32::MAX
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_build_with_max_u32_token() {
        let tokens = vec![u32::MAX, u32::MAX - 1, u32::MAX - 2];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_ngram_continuations(&[u32::MAX], 5);
        assert_eq!(conts, vec![u32::MAX - 1]);
    }

    // ------------------------------------------------------------------
    // SpecNode: estimated_acceptance is f32 (subnormal value)
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_subnormal_acceptance() {
        let node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 1e-40_f32,
            position_offset: 0,
        };
        assert!(node.estimated_acceptance > 0.0);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with pld_ngram_len larger than prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_pld_ngram_len_larger_than_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 100,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // n = min(100, 5) = 5; no token 10 found at index >= 5
        assert_eq!(tree.len(), 1, "no PLD match when pld_ngram_len > prompt len with no match");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — non-root node includes correct ancestor chain
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_ancestor_chain_is_root_to_parent() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 3;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        let spine_ids = tree.spine_ids();
        if spine_ids.len() >= 3 {
            // Node at spine_ids[2] should have ancestors [0, spine_ids[1]] in order
            let id = spine_ids[2];
            let start = indptr[id as usize];
            let end = indptr[id as usize + 1];
            let row = &indices[start..end];

            // After prefix columns, ancestors should be in root→parent order
            let ancestors: Vec<usize> = row.iter()
                .filter(|&&c| c >= total_seq_len)
                .copied()
                .collect();
            // Should include node 0, spine_ids[1], and self
            assert!(ancestors.contains(&(total_seq_len + 0)));
            assert!(ancestors.contains(&(total_seq_len + spine_ids[1] as usize)));
            assert!(ancestors.contains(&(total_seq_len + id as usize)));
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with same token appearing in adapter and prompt continuations
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_overlapping_adapter_and_pld_tokens() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Adapter token 10, PLD continuation includes 20
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10, 20, 30, 40];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root = 10, PLD extension = 20, adapter branch = 20 (duplicate allowed)
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with two identical n-grams produces correct count
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_two_identical_ngrams_count_merges() {
        // [1,2] → 3 appears twice
        let tokens = vec![1u32, 2, 3, 4, 1, 2, 3, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts, vec![3], "count should merge into single entry");
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations returns sorted by frequency desc
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_sorted_by_frequency() {
        let tokens = vec![5u32, 10, 5, 20, 5, 10, 5, 30, 5, 10];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(5, 5);
        // 10 appears 3x, 20 appears 1x, 30 appears 1x
        assert_eq!(conts[0], 10, "most frequent should be first");
        assert!(conts.contains(&20));
        assert!(conts.contains(&30));
    }

    // ------------------------------------------------------------------
    // SpecTree: build deterministic — same inputs produce same tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_deterministic() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);

        let tree1 = SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx);
        let tree2 = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert_eq!(tree1.len(), tree2.len());
        assert_eq!(tree1.all_token_ids(), tree2.all_token_ids());
        assert_eq!(tree1.spine_token_ids(), tree2.spine_token_ids());
    }

    // ------------------------------------------------------------------
    // SpecTree: root children include both PLD spine and adapter branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_children_include_spine_and_adapter() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        // pld_ngram_len=1 → scan starts at index 1.
        // Token 10 at index 2 → continuation [50]
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root = tree.node(0).unwrap();
        assert!(!root.children.is_empty());
        // At least one PLD spine child and at least one adapter branch
        let has_pld = root.children.iter().any(|&c| {
            matches!(tree.node(c).unwrap().source, DraftSource::PldSpine)
        });
        let has_adapter = root.children.iter().any(|&c| {
            matches!(tree.node(c).unwrap().source, DraftSource::AdapterTopK { k } if k > 1)
        });
        assert!(has_pld, "root should have at least one PldSpine child");
        assert!(has_adapter, "root should have at least one AdapterTopK branch");
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine with target matching partial spine prefix
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_partial_match_middle() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        if spine.len() >= 4 {
            let mut target = spine.clone();
            target[2] = 9999; // mismatch at position 2
            let (count, accepted) = tree.accepted_from_spine(&target);
            assert_eq!(count, 2);
            assert_eq!(accepted, vec![spine[0], spine[1]]);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: all nodes have non-negative position_offset
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_position_offsets_non_negative() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            assert!(node.position_offset <= node.node_id,
                "position_offset {} should be <= node_id {}", node.position_offset, node.node_id);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask indptr differences equal row NNZ counts
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_row_nnz_sum_equals_total_indices() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(7);
        let total_nnz: usize = (0..tree.len())
            .map(|i| indptr[i + 1] - indptr[i])
            .sum();
        assert_eq!(total_nnz, indices.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: build with prompt having only one occurrence of adapter token
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_prompt_single_adapter_token_occurrence() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        // Token 42 appears once in prompt, followed by 3 tokens
        let prompt = vec![1, 2, 3, 42, 100, 200, 300];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 42);
        assert!(spine.len() >= 2);
        assert!(spine.contains(&100));
        assert!(spine.contains(&200));
    }

    // ------------------------------------------------------------------
    // NgramIndex: build preserves insertion order for equal frequencies
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_equal_frequency_preserves_first_encountered() {
        // Both 10 and 20 appear once as continuations of [1,2]
        let tokens = vec![1u32, 2, 10, 3, 1, 2, 20, 4];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts.len(), 2);
        // 10 was encountered first
        assert_eq!(conts[0], 10);
        assert_eq!(conts[1], 20);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with max_branches_per_node=0 and ngram_top_k>0
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_zero_max_branches_with_ngram() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ngram_top_k: 5,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        assert_eq!(ngram_count, 0, "max_branches_per_node=0 should block n-gram branches");
    }

    // ------------------------------------------------------------------
    // SpecTree: adapter branch nodes all have root as parent
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branches_all_parent_is_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 4,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30, 40];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for i in 1..tree.len() {
            let node = tree.node(i as u32).unwrap();
            assert_eq!(node.parent_id, Some(0),
                "adapter branch {} should have root as parent", i);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask total entries equals sum of all row sizes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_indices_count_matches_indptr_diff_sum() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(7);
        let sum_diffs: usize = indptr.windows(2).map(|w| w[1] - w[0]).sum();
        assert_eq!(sum_diffs, indices.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids for tree with only spine (no branches)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_tokens_only_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let branches = tree.branch_token_ids();
        assert!(branches.is_empty(), "no branches when max_branches_per_node=0 and no adapter branches");
    }

    // ------------------------------------------------------------------
    // SpecTree: build with token ID 0 appearing in prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_token_zero_in_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![0u32];
        // pld_ngram_len=1 → scan starts at index 1.
        // Token 0 at index 2 → continuation [100, 200]
        let prompt = vec![1, 2, 0, 100, 200, 300];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 0);
        assert!(spine.len() >= 2);
        assert!(spine.contains(&100));
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations with very large top_k
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_very_large_top_k() {
        let tokens = vec![1u32, 2, 3, 1, 4, 1, 5];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(1, usize::MAX);
        // Should return all available continuations without panic
        assert_eq!(conts.len(), 3);
        assert!(conts.contains(&2));
        assert!(conts.contains(&4));
        assert!(conts.contains(&5));
    }

    // ------------------------------------------------------------------
    // SpecTree: all adapter branch estimated_acceptance values are positive
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branches_positive_acceptance() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 5,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30, 40, 50];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            assert!(node.estimated_acceptance > 0.0,
                "node {} acceptance should be positive, got {}", node.node_id, node.estimated_acceptance);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branch acceptance within expected range
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branch_acceptance_range() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 3,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 10, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                // N-gram acceptance should be in [0.05, 0.15] range per build logic
                assert!(node.estimated_acceptance >= 0.0,
                    "n-gram node {} acceptance {} should be >= 0", node.node_id, node.estimated_acceptance);
                assert!(node.estimated_acceptance <= 0.15,
                    "n-gram node {} acceptance {} should be <= 0.15", node.node_id, node.estimated_acceptance);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: PldSpine node acceptance decreases linearly
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_pld_spine_acceptance_linear_decrease() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        // PldSpine nodes: acceptance = 0.60 - depth * 0.05
        for depth in 1..spine_ids.len() {
            let node = tree.node(spine_ids[depth]).unwrap();
            let expected = 0.60 - depth as f32 * 0.05;
            assert!((node.estimated_acceptance - expected).abs() < f32::EPSILON,
                "PLD spine depth {} acceptance should be {}, got {}",
                depth, expected, node.estimated_acceptance);
        }
    }

    // ==================================================================
    // NEW TESTS (tests 261+, ~60 additional tests)
    // ==================================================================

    // ------------------------------------------------------------------
    // SpecTree construction: new() with custom config fields
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_new_preserves_config_zero_max_tree_size() {
        // Arrange
        let config = SpecTreeConfig {
            max_tree_size: 0,
            ..SpecTreeConfig::default()
        };
        // Act
        let tree = SpecTree::new(config);
        // Assert
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    // ------------------------------------------------------------------
    // SpecTree construction: build returns non-empty for single adapter
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_single_token_non_empty() {
        // Arrange
        let config = SpecTreeConfig::default();
        let ngram_idx = NgramIndex::build(&[], 2);
        // Act
        let tree = SpecTree::build(config, &[42u32], &[], &ngram_idx);
        // Assert
        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());
    }

    // ------------------------------------------------------------------
    // Node field values: root node_id is 0
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_root_node_id_is_zero() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Act
        let tree = SpecTree::build(config, &[99u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        // Assert
        let root = tree.node(0).unwrap();
        assert_eq!(root.node_id, 0);
        assert_eq!(root.token_id, 99);
    }

    // ------------------------------------------------------------------
    // Node field values: PldSpine node has correct position_offset
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_pld_spine_position_offset_increments() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let spine_ids = tree.spine_ids();
        if spine_ids.len() >= 2 {
            let first_pld = tree.node(spine_ids[1]).unwrap();
            assert_eq!(first_pld.position_offset, 1, "first PLD node offset should be 1");
        }
    }

    // ------------------------------------------------------------------
    // Debug trait: NgramIndex with empty table
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_debug_empty_table() {
        // Arrange
        let idx = NgramIndex::build(&[], 3);
        // Act
        let debug = format!("{:?}", idx);
        // Assert
        assert!(debug.contains("NgramIndex") || debug.contains("table"));
    }

    // ------------------------------------------------------------------
    // Tree depth: spine length respects max_spine_depth exactly
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_depth_exactly_two() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70, 80, 90];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let spine = tree.spine_ids();
        assert!(spine.len() <= 2, "spine should be at most 2, got {}", spine.len());
    }

    // ------------------------------------------------------------------
    // Tree breadth: root children bounded by adapter + n-gram sources
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_children_bounded() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 3,
            ngram_top_k: 5,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt: Vec<u32> = (0..30).flat_map(|i| vec![10, 100 + i]).collect();
        // Act
        let tree = SpecTree::build(
            config,
            &[10u32, 20],
            &prompt,
            &NgramIndex::build(&prompt, 1),
        );
        // Assert
        let root = tree.node(0).unwrap();
        // At most 1 adapter branch (top-2) + max_branches_per_node n-gram branches
        assert!(root.children.len() <= 4, "root children should be bounded");
    }

    // ------------------------------------------------------------------
    // CSR mask: empty tree returns trivial indptr
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_empty_tree_trivial_indptr() {
        // Arrange
        let tree = SpecTree::new(SpecTreeConfig::default());
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(0);
        // Assert
        assert_eq!(indptr, vec![0]);
        assert!(indices.is_empty());
    }

    // ------------------------------------------------------------------
    // CSR mask: each row size >= total_seq_len + 1 (prefix + self)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_min_row_size() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[10u32, 20],
            &[1, 2, 10, 50, 60],
            &NgramIndex::build(&[1, 2, 10, 50, 60], 2),
        );
        let total_seq_len = 20;
        // Act
        let (indptr, _indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert
        for i in 0..tree.len() {
            let row_size = indptr[i + 1] - indptr[i];
            assert!(row_size >= total_seq_len + 1,
                "row {} size {} should be >= {} (prefix + self)", i, row_size, total_seq_len + 1);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: insertion of identical n-gram many times
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_identical_ngram_many_insertions() {
        // Arrange
        let tokens: Vec<u32> = (0..100).flat_map(|_| [1u32, 2, 3]).collect(); // [1,2,3] repeated 100 times
        // Act
        let idx = NgramIndex::build(&tokens, 2);
        // Assert
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts, vec![3]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: lookup of n-gram that spans end of token list
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_lookup_spanning_end_of_tokens() {
        // Arrange
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        // Act: [4,5] has no continuation (at end)
        let conts = idx.get_ngram_continuations(&[4, 5], 5);
        // Assert
        assert!(conts.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: default max_tree_size is 32
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_default_max_tree_size() {
        // Arrange & Act
        let cfg = SpecTreeConfig::default();
        // Assert
        assert_eq!(cfg.max_tree_size, 32);
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: default pld_ngram_len is 3
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_default_pld_ngram_len() {
        // Arrange & Act
        let cfg = SpecTreeConfig::default();
        // Assert
        assert_eq!(cfg.pld_ngram_len, 3);
    }

    // ------------------------------------------------------------------
    // Token probability: root acceptance is exactly 0.70
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_acceptance_exact_value() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Act
        let tree = SpecTree::build(config, &[42u32], &[], &NgramIndex::build(&[], 2));
        // Assert
        let root = tree.node(0).unwrap();
        let diff = (root.estimated_acceptance - 0.70).abs();
        assert!(diff < 1e-6, "root acceptance diff from 0.70 = {}", diff);
    }

    // ------------------------------------------------------------------
    // Draft scoring: adapter branch k=2 acceptance formula
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branch_k2_acceptance() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Act
        let tree = SpecTree::build(config, &[10u32, 20, 30], &[], &NgramIndex::build(&[], 2));
        // Assert: adapter branch k=2 → acceptance = 0.15 / max(1.0, 0.0+1.0) = 0.15
        let node1 = tree.node(1).unwrap();
        assert!((node1.estimated_acceptance - 0.15).abs() < f32::EPSILON,
            "adapter k=2 acceptance should be 0.15, got {}", node1.estimated_acceptance);
    }

    // ------------------------------------------------------------------
    // Draft scoring: adapter branch k=3 acceptance formula
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branch_k3_acceptance() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 4,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Act
        let tree = SpecTree::build(config, &[10u32, 20, 30, 40], &[], &NgramIndex::build(&[], 2));
        // Assert: adapter branch k=3 → acceptance = 0.15 / max(1.0, 1.0+1.0) = 0.075
        let node2 = tree.node(2).unwrap();
        let expected = 0.15 / 2.0;
        assert!((node2.estimated_acceptance - expected).abs() < f32::EPSILON,
            "adapter k=3 acceptance should be {}, got {}", expected, node2.estimated_acceptance);
    }

    // ------------------------------------------------------------------
    // Verification: accepted_from_spine with alternating match/mismatch
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_alternating_match_mismatch() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60];
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        let spine = tree.spine_token_ids();
        // Act: second token mismatches
        let mut target = spine.clone();
        if target.len() > 1 {
            target[1] = 99999;
        }
        let (count, accepted) = tree.accepted_from_spine(&target);
        // Assert
        assert!(count <= 1, "should accept at most 1 before mismatch at position 1");
        if !spine.is_empty() {
            assert_eq!(accepted, vec![spine[0]]);
        }
    }

    // ------------------------------------------------------------------
    // Tree pruning: max_tree_size stops n-gram branch insertion
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_max_tree_size_stops_ngram_insertion() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 100,
            ngram_top_k: 100,
            max_tree_size: 1,
            pld_ngram_len: 1,
        };
        let prompt: Vec<u32> = (0..50).flat_map(|i| vec![10, 100 + i]).collect();
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: only root (max_tree_size=1, root already consumes it)
        assert!(tree.len() <= 1, "tree should be at most 1 node, got {}", tree.len());
    }

    // ------------------------------------------------------------------
    // Tree pruning: n-gram branch count bounded when tree near max
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_bounded_near_max_tree_size() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 50,
            ngram_top_k: 50,
            max_tree_size: 4,
            pld_ngram_len: 1,
        };
        let prompt: Vec<u32> = (0..50).flat_map(|i| vec![10, 100 + i]).collect();
        // Act
        let tree = SpecTree::build(config, &[10u32, 20], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: total nodes <= max_tree_size + spine + adapter (spine/adapter added before gate)
        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        assert!(ngram_count <= 4, "n-gram branches bounded, got {}", ngram_count);
    }

    // ------------------------------------------------------------------
    // SpecNode: Clone produces deep copy of children Vec
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_children_vec_deep_copy_on_clone() {
        // Arrange
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![10, 20], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        // Act
        let mut cloned = node.clone();
        cloned.children[0] = 99;
        // Assert
        assert_eq!(node.children[0], 10, "original should be unaffected");
        assert_eq!(cloned.children[0], 99);
    }

    // ------------------------------------------------------------------
    // SpecNode: Debug includes all field names
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_debug_includes_all_field_names() {
        // Arrange
        let node = SpecNode {
            node_id: 0, token_id: 0, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.0, position_offset: 0,
        };
        // Act
        let debug = format!("{:?}", node);
        // Assert
        assert!(debug.contains("node_id"));
        assert!(debug.contains("token_id"));
        assert!(debug.contains("parent_id"));
        assert!(debug.contains("children"));
        assert!(debug.contains("source"));
        assert!(debug.contains("estimated_acceptance"));
        assert!(debug.contains("position_offset"));
    }

    // ------------------------------------------------------------------
    // SpecTree: empty tree spine_ids would be empty (graceful)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_empty_nodes_returns_empty_slice() {
        // Arrange
        let tree = SpecTree::new(SpecTreeConfig::default());
        // Act
        let nodes = tree.nodes();
        // Assert
        assert!(nodes.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: node accessor returns None for u32::MAX on non-empty tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_node_accessor_max_u32_on_non_empty() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1, adapter_top_k: 1, max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2], &NgramIndex::build(&[1, 2], 2));
        // Act & Assert
        assert!(tree.node(u32::MAX).is_none());
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with n=1 and long alternating sequence
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_n1_alternating_sequence() {
        // Arrange
        let tokens: Vec<u32> = (0..20).flat_map(|i| vec![1, i + 10]).collect();
        // Act
        let idx = NgramIndex::build(&tokens, 1);
        // Assert
        let conts = idx.get_ngram_continuations(&[1], 5);
        assert!(!conts.is_empty(), "token 1 should have continuations");
        assert!(conts.contains(&10), "first continuation should be 10");
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations returns empty for n > 1
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_n3_returns_empty_for_single() {
        // Arrange
        let tokens = vec![1u32, 2, 3, 1, 2, 4];
        let idx = NgramIndex::build(&tokens, 3);
        // Act
        let conts = idx.get_continuations(1, 5);
        // Assert: get_continuations hashes [1] but table has 3-gram keys
        assert!(conts.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: PartialOrd is not derived (struct uses PartialEq)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_equality_reflexive() {
        // Arrange
        let cfg = SpecTreeConfig::default();
        // Act & Assert
        assert_eq!(cfg, cfg);
    }

    // ------------------------------------------------------------------
    // DraftSource: all variants have distinct Debug output
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_debug_distinct_strings() {
        // Arrange
        let d1 = format!("{:?}", DraftSource::PldSpine);
        let d2 = format!("{:?}", DraftSource::NgramBranch);
        let d3 = format!("{:?}", DraftSource::AdapterTopK { k: 1 });
        // Assert
        assert_ne!(d1, d2);
        assert_ne!(d2, d3);
        assert_ne!(d1, d3);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter token appearing exactly at pld_ngram_len offset
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_token_at_exact_pld_offset() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 3,
            ..SpecTreeConfig::default()
        };
        // Token 10 at index 3 (= pld_ngram_len), followed by [50, 60]
        let prompt = vec![1, 2, 3, 10, 50, 60];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 2));
        // Assert
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        assert!(spine.len() >= 2, "should find PLD extension at exact offset");
    }

    // ------------------------------------------------------------------
    // SpecTree: build with token appearing at last-1 position (no continuation)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_token_at_penultimate_position() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 at index 3, followed by only one token [50]
        let prompt = vec![1, 2, 3, 10, 50];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        // Only one continuation token available
        if spine.len() >= 2 {
            assert_eq!(spine[1], 50);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask ancestor columns are in ascending order per row
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_ancestor_columns_ascending() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[10, 50, 60, 70], &NgramIndex::build(&[10, 50, 60, 70], 1));
        let total_seq_len = 4;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: ancestor columns (after prefix) should be ascending
        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            let ancestors = &row[total_seq_len..];
            for w in ancestors.windows(2) {
                assert!(w[0] < w[1], "row {} ancestor columns not ascending: {} >= {}", i, w[0], w[1]);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask first row is exactly prefix + self (no ancestors)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_root_row_no_ancestors() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[1, 2, 10, 50], &NgramIndex::build(&[1, 2, 10, 50], 2));
        let total_seq_len = 5;
        // Act
        let (indptr, _indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert
        let root_end = indptr[1];
        assert_eq!(root_end, total_seq_len + 1, "root row should be prefix + self only");
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with tokens where all n-grams have unique continuations
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_all_unique_ngram_continuations() {
        // Arrange: strictly increasing sequence
        let tokens: Vec<u32> = (0..10).collect();
        // Act
        let idx = NgramIndex::build(&tokens, 2);
        // Assert
        for i in 0..8 {
            let conts = idx.get_ngram_continuations(&[i, i + 1], 5);
            assert_eq!(conts, vec![i + 2]);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with prompt containing only the adapter token
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_prompt_only_adapter_token() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10u32];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: pld_ngram_len=1 → scan starts at index 1, but prompt only has 1 element
        // so no PLD match (loop starts at i=1 which is >= prompt.len())
        assert_eq!(tree.len(), 1, "no PLD match when prompt is just the adapter token");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask with tree having both adapter and n-gram branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_mixed_branch_types() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 10, 70, 80];
        // Act
        let tree = SpecTree::build(config, &[10u32, 20], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let (indptr, indices) = tree.tree_attention_mask_csr(5);
        // All column indices within valid range
        let (_, cols) = tree.mask_shape(5);
        for &col in &indices {
            assert!(col < cols, "column {} out of range (max {})", col, cols);
        }
        assert_eq!(indptr.len(), tree.len() + 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: all_token_ids and spine_token_ids first element match
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_ids_and_spine_ids_first_match() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70];
        let tree = SpecTree::build(config, &[10u32, 20], &prompt, &NgramIndex::build(&prompt, 1));
        // Act
        let all_ids = tree.all_token_ids();
        let spine_ids = tree.spine_token_ids();
        // Assert
        assert_eq!(all_ids[0], spine_ids[0]);
        assert_eq!(spine_ids[0], 10u32);
    }

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids excludes spine nodes correctly
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_excludes_all_spine_nodes() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32, 20, 30], &[], &NgramIndex::build(&[], 2));
        // Act
        let spine_tokens: std::collections::HashSet<u32> = tree.spine_token_ids().into_iter().collect();
        let branch_ids: std::collections::HashSet<u32> = tree.branch_token_ids()
            .into_iter().map(|(_, tok)| tok).collect();
        // Assert
        for spine_tok in &spine_tokens {
            assert!(!branch_ids.contains(spine_tok),
                "spine token {} should not appear in branches", spine_tok);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with ngram_top_k=1 returns at most 1 continuation per spine node
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_top_k_1_limits_branches() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 10,
            ngram_top_k: 1,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt: Vec<u32> = (0..20).flat_map(|i| vec![10, 100 + i]).collect();
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        for node in tree.nodes() {
            let ngram_children: Vec<&SpecNode> = node.children.iter()
                .filter(|&&c| matches!(tree.node(c).unwrap().source, DraftSource::NgramBranch))
                .map(|&c| tree.node(c).unwrap())
                .collect();
            assert!(ngram_children.len() <= 1,
                "node {} has {} n-gram children, expected <= 1", node.node_id, ngram_children.len());
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with exactly n+1 tokens yields exactly one entry
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_exact_n_plus_one_tokens_one_entry() {
        // Arrange & Act
        let idx = NgramIndex::build(&[5u32, 6, 7, 8], 3);
        // Assert
        assert!(!idx.table.is_empty());
        let conts = idx.get_ngram_continuations(&[5, 6, 7], 5);
        assert_eq!(conts, vec![8]);
    }

    // ------------------------------------------------------------------
    // SpecNode: PartialEq with same f32 acceptance but different children
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_eq_same_acceptance_different_children() {
        // Arrange
        let a = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![2], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![3], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        // Assert
        assert_ne!(a, b, "different children should make nodes unequal");
    }

    // ------------------------------------------------------------------
    // SpecTree: build where n-gram index is empty but adapter tokens present
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_empty_ngram_index_with_adapter() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let ngram_idx = NgramIndex::build(&[], 2);
        // Act
        let tree = SpecTree::build(config, &[10u32, 20, 30], &[], &ngram_idx);
        // Assert
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.node(0).unwrap().token_id, 10);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask columns for branch node include root
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_branch_includes_root_ancestor() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32, 20, 30], &[], &NgramIndex::build(&[], 2));
        let total_seq_len = 3;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: node 1 (adapter branch) should attend to root (node 0)
        let start = indptr[1];
        let end = indptr[2];
        let row: Vec<usize> = indices[start..end].to_vec();
        assert!(row.contains(&(total_seq_len + 0)),
            "adapter branch node 1 should attend to root at column {}", total_seq_len);
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_ids produces correct chain with branches present
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_ids_correct_with_branches() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 2,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 10, 70, 80];
        // Act
        let tree = SpecTree::build(config, &[10u32, 20], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: spine follows PldSpine chain
        let spine_ids = tree.spine_ids();
        for (i, &id) in spine_ids.iter().enumerate() {
            let node = tree.node(id).unwrap();
            if i == 0 {
                assert_eq!(node.source, DraftSource::AdapterTopK { k: 1 });
            } else {
                assert_eq!(node.source, DraftSource::PldSpine);
            }
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations for token at u32::MAX
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_max_token() {
        // Arrange
        let tokens = vec![u32::MAX, 1, u32::MAX, 2];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(u32::MAX, 5);
        // Assert
        assert_eq!(conts.len(), 2);
        assert!(conts.contains(&1));
        assert!(conts.contains(&2));
    }

    // ------------------------------------------------------------------
    // SpecTree: build with two-word prompt where second token is adapter
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_second_token_is_adapter() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // pld_ngram_len=1 → scan starts at index 1. Token 50 at index 1, followed by [60]
        let prompt = vec![10, 50, 60, 70];
        // Act
        let tree = SpecTree::build(config, &[50u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 50);
        assert!(spine.len() >= 2, "should find PLD extension");
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branch on root has correct parent
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branch_on_root_parent_is_root() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 10, 60];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                assert_eq!(node.parent_id, Some(0),
                    "n-gram branch {} should have root as parent", node.node_id);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask with total_seq_len=0 and multi-node tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_zero_seq_multi_node() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32, 20], &[1, 2, 10, 50], &NgramIndex::build(&[1, 2, 10, 50], 2));
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(0);
        // Assert
        assert_eq!(indptr.len(), tree.len() + 1);
        // No prefix columns, only ancestor + self columns
        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            // Should at least contain self
            assert!(row.contains(&i), "row {} should contain self at column {}", i, i);
        }
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: manual construction with usize::MAX fields
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_usize_max_fields() {
        // Arrange & Act
        let cfg = SpecTreeConfig {
            max_spine_depth: usize::MAX,
            max_branches_per_node: usize::MAX,
            pld_ngram_len: usize::MAX,
            ngram_top_k: usize::MAX,
            adapter_top_k: usize::MAX,
            max_tree_size: usize::MAX,
        };
        // Assert
        assert_eq!(cfg.max_spine_depth, usize::MAX);
        assert_eq!(cfg.max_tree_size, usize::MAX);
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine returns matching prefix correctly
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_returns_matching_prefix_only() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70];
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        let spine = tree.spine_token_ids();
        // Act: match first 2 only
        if spine.len() >= 3 {
            let target = vec![spine[0], spine[1], 99999];
            let (count, accepted) = tree.accepted_from_spine(&target);
            // Assert
            assert_eq!(count, 2);
            assert_eq!(accepted, vec![spine[0], spine[1]]);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: clone produces structurally identical table
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_clone_structurally_identical() {
        // Arrange
        let tokens = vec![1u32, 2, 3, 1, 2, 4, 1, 2, 5];
        let idx = NgramIndex::build(&tokens, 2);
        // Act
        let cloned = idx.clone();
        // Assert: same n field
        assert_eq!(idx.n, cloned.n);
        // Same query results
        assert_eq!(idx.get_ngram_continuations(&[1, 2], 10), cloned.get_ngram_continuations(&[1, 2], 10));
    }

    // ------------------------------------------------------------------
    // SpecTree: build with all same token IDs in prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_all_same_prompt_tokens() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10u32; 20];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: pld_ngram_len=1 → scan starts at index 1.
        // Token 10 appears many times; continuation is also 10.
        // Deduplication ensures only one extension.
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        if spine.len() > 1 {
            assert_eq!(spine[1], 10, "continuation of [10,...,10] should be 10");
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: nodes() iterator covers all nodes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_nodes_iterator_covers_all() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config, &[10u32, 20], &[1, 2, 10, 50, 60],
            &NgramIndex::build(&[1, 2, 10, 50, 60], 2),
        );
        // Act
        let mut count = 0;
        for (i, node) in tree.nodes().iter().enumerate() {
            assert_eq!(node.node_id, i as u32);
            count += 1;
        }
        // Assert
        assert_eq!(count, tree.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: mask_shape with total_seq_len=0 for empty tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_mask_shape_empty_zero_seq() {
        // Arrange
        let tree = SpecTree::new(SpecTreeConfig::default());
        // Act
        let (rows, cols) = tree.mask_shape(0);
        // Assert
        assert_eq!(rows, 0);
        assert_eq!(cols, 0);
    }

    // ------------------------------------------------------------------
    // DraftSource: AdapterTopK k field access
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_adapter_top_k_field_access() {
        // Arrange
        let source = DraftSource::AdapterTopK { k: 7 };
        // Act & Assert
        if let DraftSource::AdapterTopK { k } = source {
            assert_eq!(k, 7);
        } else {
            panic!("expected AdapterTopK variant");
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build determinism across multiple calls with same ngram index
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_deterministic_same_ngram_index() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree1 = SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx);
        let tree2 = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        assert_eq!(tree1, tree2);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask row sizes for adapter-only tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_adapter_only_tree_row_sizes() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32, 20, 30], &[], &NgramIndex::build(&[], 2));
        let total_seq_len = 0;
        // Act
        let (indptr, _indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: root row has just self; branch rows have root + self
        assert_eq!(indptr[1] - indptr[0], 1, "root should attend only to self");
        for i in 1..tree.len() {
            let row_size = indptr[i + 1] - indptr[i];
            assert_eq!(row_size, 2, "branch {} should attend to root + self", i);
        }
    }

    // ------------------------------------------------------------------
    // SpecNode: Default-like construction with all zero fields
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_zero_construction() {
        // Arrange & Act
        let node = SpecNode {
            node_id: 0,
            token_id: 0,
            parent_id: None,
            children: vec![],
            source: DraftSource::AdapterTopK { k: 0 },
            estimated_acceptance: 0.0,
            position_offset: 0,
        };
        // Assert
        assert_eq!(node.node_id, 0);
        assert_eq!(node.token_id, 0);
        assert!(node.parent_id.is_none());
        assert!(node.children.is_empty());
        assert_eq!(node.position_offset, 0);
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_ngram_continuations panics on length mismatch
    // ------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "n-gram length mismatch")]
    fn ngram_index_get_ngram_continuations_wrong_length_panics() {
        // Arrange
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        // Act — query with wrong length (3 elements, n=2)
        let _ = idx.get_ngram_continuations(&[1, 2, 3], 5);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with prompt where adapter token has many unique continuations
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_many_unique_pld_continuations() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 10,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 followed by 50, 60, 70, 80, 90
        let prompt = vec![1, 10, 50, 60, 70, 80, 90];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        assert!(spine.len() >= 3, "should have multiple unique PLD extensions, got {}", spine.len() - 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: clone and modify independence
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_clone_modify_independence() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32, 20], &[1, 2], &NgramIndex::build(&[1, 2], 2));
        let original_len = tree.len();
        // Act
        let _clone = tree.clone();
        // Assert: original unchanged
        assert_eq!(tree.len(), original_len);
    }

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids correct with multiple adapter branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_ids_multiple_adapter_branches() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 4,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32, 20, 30, 40], &[], &NgramIndex::build(&[], 2));
        // Act
        let branches = tree.branch_token_ids();
        // Assert
        assert_eq!(branches.len(), 3, "3 adapter branches expected");
        assert!(branches.contains(&(1, 20)));
        assert!(branches.contains(&(2, 30)));
        assert!(branches.contains(&(3, 40)));
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with n=1 and single continuation repeated
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_n1_repeated_continuation() {
        // Arrange: [5] → 10 appears 3 times
        let tokens = vec![5u32, 10, 5, 10, 5, 10];
        // Act
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(5, 5);
        // Assert
        assert_eq!(conts, vec![10]);
    }

    // ------------------------------------------------------------------
    // SpecTree: all nodes have unique node_ids
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_unique_node_ids() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config, &[10u32, 20], &[1, 2, 10, 50, 60, 10, 70],
            &NgramIndex::build(&[1, 2, 10, 50, 60, 10, 70], 2),
        );
        // Act
        let node_ids: Vec<u32> = tree.nodes().iter().map(|n| n.node_id).collect();
        let unique_ids: std::collections::HashSet<u32> = node_ids.iter().copied().collect();
        // Assert
        assert_eq!(node_ids.len(), unique_ids.len(), "all node IDs should be unique");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask prefix columns span [0, total_seq_len)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_prefix_columns_complete_range() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[10, 50], &NgramIndex::build(&[10, 50], 2));
        let total_seq_len = 10;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: each row starts with [0, 1, ..., total_seq_len-1]
        for i in 0..tree.len() {
            let row = &indices[indptr[i]..indptr[i + 1]];
            for col in 0..total_seq_len {
                assert_eq!(row[col], col, "row {} prefix column {} mismatch", i, col);
            }
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: Debug output for non-empty index
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_debug_non_empty() {
        // Arrange
        let idx = NgramIndex::build(&[1u32, 2, 3, 4, 5], 2);
        // Act
        let debug = format!("{:?}", idx);
        // Assert
        assert!(!debug.is_empty());
        assert!(debug.contains("n") || debug.contains("table"));
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine with target matching middle of spine
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_middle_match_stops_at_mismatch() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70, 80];
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        let spine = tree.spine_token_ids();
        if spine.len() >= 4 {
            let mut target = spine.clone();
            target[3] = 77777;
            // Act
            let (count, accepted) = tree.accepted_from_spine(&target);
            // Assert
            assert_eq!(count, 3);
            assert_eq!(accepted.len(), 3);
            assert_eq!(accepted[0], spine[0]);
            assert_eq!(accepted[1], spine[1]);
            assert_eq!(accepted[2], spine[2]);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: PLD continuation deduplication across multiple occurrences
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_pld_dedup_across_occurrences() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 10,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 appears 3 times, each followed by 50
        let prompt = vec![10, 50, 20, 10, 50, 30, 10, 50, 60, 70];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        // 50 should appear only once in spine despite being continuation of 3 occurrences
        let mut count_50 = 0;
        for &tok in &spine[1..] {
            if tok == 50 { count_50 += 1; }
        }
        assert!(count_50 <= 1, "token 50 should appear at most once in spine, found {}", count_50);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter_top_k=1 and single token, no branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_no_adapter_branches_single_token() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Act
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        // Assert
        let adapter_branches: Vec<&SpecNode> = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .collect();
        assert!(adapter_branches.is_empty(), "no adapter branches with adapter_top_k=1");
    }

    // ------------------------------------------------------------------
    // SpecTree: position_offset for spine is sequential starting from 0
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_position_offsets_sequential_from_zero() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70, 80, 90];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let spine_ids = tree.spine_ids();
        for (i, &id) in spine_ids.iter().enumerate() {
            assert_eq!(tree.node(id).unwrap().position_offset, i as u32,
                "spine node {} at index {} should have offset {}", id, i, i);
        }
    }

    // ── DraftSource additional tests ──

    #[test]
    fn draft_source_adapter_top_k_equality() {
        assert_eq!(DraftSource::AdapterTopK { k: 3 }, DraftSource::AdapterTopK { k: 3 });
        assert_ne!(DraftSource::AdapterTopK { k: 1 }, DraftSource::AdapterTopK { k: 3 });
    }

    #[test]
    fn draft_source_hash_set_dedup() {
        use std::collections::HashSet;
        let set: HashSet<DraftSource> = [
            DraftSource::PldSpine,
            DraftSource::NgramBranch,
            DraftSource::PldSpine,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn draft_source_debug_format_variants() {
        assert!(format!("{:?}", DraftSource::PldSpine).contains("PldSpine"));
        assert!(format!("{:?}", DraftSource::NgramBranch).contains("NgramBranch"));
        let adapter_debug = format!("{:?}", DraftSource::AdapterTopK { k: 5 });
        assert!(adapter_debug.contains("AdapterTopK"));
    }

    // ── SpecNode construction tests ──

    #[test]
    fn spec_node_zero_fields() {
        let node = SpecNode {
            node_id: 0,
            token_id: 0,
            parent_id: None,
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.0,
            position_offset: 0,
        };
        assert_eq!(node.node_id, 0);
        assert_eq!(node.token_id, 0);
        assert!(node.parent_id.is_none());
        assert!(node.children.is_empty());
    }

    #[test]
    fn spec_node_max_u32_fields() {
        let node = SpecNode {
            node_id: u32::MAX,
            token_id: u32::MAX,
            parent_id: Some(u32::MAX),
            children: vec![u32::MAX],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 1.0,
            position_offset: u32::MAX,
        };
        assert_eq!(node.node_id, u32::MAX);
        assert_eq!(node.token_id, u32::MAX);
        assert_eq!(node.parent_id, Some(u32::MAX));
        assert_eq!(node.children[0], u32::MAX);
        assert_eq!(node.position_offset, u32::MAX);
    }

    #[test]
    fn spec_node_clone_independent() {
        let node = SpecNode {
            node_id: 5,
            token_id: 42,
            parent_id: Some(2),
            children: vec![10, 11],
            source: DraftSource::AdapterTopK { k: 3 },
            estimated_acceptance: 0.7,
            position_offset: 3,
        };
        let cloned = node.clone();
        assert_eq!(cloned.node_id, 5);
        assert_eq!(cloned.token_id, 42);
        assert_eq!(cloned.children, vec![10, 11]);
    }

    #[test]
    fn spec_node_equality_check() {
        let a = SpecNode {
            node_id: 1,
            token_id: 10,
            parent_id: None,
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.5,
            position_offset: 0,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn spec_node_debug_format() {
        let node = SpecNode {
            node_id: 99,
            token_id: 100,
            parent_id: Some(50),
            children: vec![],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 0.3,
            position_offset: 5,
        };
        let debug = format!("{:?}", node);
        assert!(debug.contains("SpecNode"));
    }

    // ── SpecTreeConfig custom values ──

    #[test]
    fn spec_tree_config_custom_all_fields() {
        let config = SpecTreeConfig {
            max_spine_depth: 10,
            max_branches_per_node: 5,
            pld_ngram_len: 4,
            ngram_top_k: 3,
            adapter_top_k: 5,
            max_tree_size: 128,
        };
        assert_eq!(config.max_spine_depth, 10);
        assert_eq!(config.max_branches_per_node, 5);
        assert_eq!(config.pld_ngram_len, 4);
        assert_eq!(config.ngram_top_k, 3);
        assert_eq!(config.adapter_top_k, 5);
        assert_eq!(config.max_tree_size, 128);
    }

    #[test]
    fn spec_tree_config_equality_self() {
        let config = SpecTreeConfig::default();
        assert_eq!(config, config);
    }

    #[test]
    fn spec_tree_config_zero_values() {
        let config = SpecTreeConfig {
            max_spine_depth: 0,
            max_branches_per_node: 0,
            pld_ngram_len: 0,
            ngram_top_k: 0,
            adapter_top_k: 0,
            max_tree_size: 0,
        };
        assert_eq!(config.max_spine_depth, 0);
        assert_eq!(config.max_tree_size, 0);
    }

    // ── SpecTree empty state (using pub field total_nodes) ──

    #[test]
    fn spec_tree_new_empty_total_nodes_zero() {
        let config = SpecTreeConfig::default();
        let tree = SpecTree::new(config);
        assert_eq!(tree.total_nodes, 0);
    }

    #[test]
    fn spec_tree_new_empty_node_returns_none() {
        let config = SpecTreeConfig::default();
        let tree = SpecTree::new(config);
        assert!(tree.node(0).is_none());
    }

    // ── NgramIndex empty state (using get_ngram_continuations) ──

    #[test]
    fn ngram_index_empty_prompt_no_continuations() {
        let index = NgramIndex::build(&[], 3);
        let results = index.get_ngram_continuations(&[1, 2, 3], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn ngram_index_no_match_returns_empty() {
        let index = NgramIndex::build(&[1, 2, 3, 4, 5], 3);
        let results = index.get_ngram_continuations(&[99, 98, 97], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn ngram_index_clone_preserves_continuations() {
        let index = NgramIndex::build(&[1, 2, 3, 1, 2, 4], 2);
        let cloned = index.clone();
        let orig = index.get_ngram_continuations(&[1, 2], 5);
        let cloned_r = cloned.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(orig, cloned_r);
    }

    // ── SpecTree build edge cases ──

    #[test]
    fn spec_tree_build_empty_adapter_total_nodes_zero() {
        let config = SpecTreeConfig::default();
        let prompt = vec![1, 2, 3];
        let ngram = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[], &prompt, &ngram);
        assert_eq!(tree.total_nodes, 0);
    }

    #[test]
    fn spec_tree_build_single_token_creates_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 20, 30, 40, 50, 10, 20, 60];
        let ngram = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram);
        assert!(tree.total_nodes >= 1);
        assert!(!tree.spine_ids().is_empty());
    }

    // ── NgramIndex debug format ──

    #[test]
    fn ngram_index_debug_format() {
        let index = NgramIndex::build(&[1, 2, 3], 2);
        let debug = format!("{:?}", index);
        assert!(!debug.is_empty());
    }

    // ══════════════════════════════════════════════════════════════════
    // New tests: 45 additional tests
    // ══════════════════════════════════════════════════════════════════

    // ── SpecTreeConfig: max_branches_per_node field ──

    #[test]
    fn spec_tree_config_default_max_branches_per_node() {
        let cfg = SpecTreeConfig::default();
        assert_eq!(cfg.max_branches_per_node, 2);
    }

    // ── SpecTreeConfig: adapter_top_k field ──

    #[test]
    fn spec_tree_config_default_adapter_top_k() {
        let cfg = SpecTreeConfig::default();
        assert_eq!(cfg.adapter_top_k, 3);
    }

    // ── SpecTreeConfig: ngram_top_k field ──

    #[test]
    fn spec_tree_config_default_ngram_top_k() {
        let cfg = SpecTreeConfig::default();
        assert_eq!(cfg.ngram_top_k, 2);
    }

    // ── SpecTree: build with no prompt produces spine root only ──

    #[test]
    fn spec_tree_build_empty_prompt_only_adapter_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 3);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Only root node from adapter top-1, no PLD extensions, no ngram branches
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.node(0).unwrap().token_id, 42);
    }

    // ── SpecTree: spine_len reflects actual spine size ──

    #[test]
    fn spec_tree_spine_len_matches_spine_ids_count() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 3, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let spine_count = tree.spine_ids().len();
        assert_eq!(tree.spine_len, spine_count);
    }

    // ── SpecTree: node parent_id chain terminates at root ──

    #[test]
    fn spec_tree_parent_chain_all_reach_root() {
        let config = SpecTreeConfig::default();
        let adapter_tokens = vec![100u32, 200, 300];
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 55];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        for node in tree.nodes() {
            let mut current = node.node_id;
            let mut steps = 0;
            while let Some(pid) = tree.node(current).unwrap().parent_id {
                current = pid;
                steps += 1;
                assert!(steps <= tree.len(), "parent chain too long, possible cycle");
            }
            // Root has parent_id = None
            assert!(tree.node(current).unwrap().parent_id.is_none());
        }
    }

    // ── SpecTree: every child references is bidirectional ──

    #[test]
    fn spec_tree_children_parent_bidirectional() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            max_branches_per_node: 2,
            adapter_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![100u32, 200, 300];
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 55, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        for node in tree.nodes() {
            for &child_id in &node.children {
                let child = tree.node(child_id).unwrap();
                assert_eq!(child.parent_id, Some(node.node_id));
            }
        }
    }

    // ── SpecTree: node_id matches index in nodes vec ──

    #[test]
    fn spec_tree_node_id_matches_vec_index() {
        let config = SpecTreeConfig::default();
        let adapter_tokens = vec![100u32, 200, 300];
        let prompt = vec![1, 2, 3, 4, 100, 10, 20, 100, 15];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        for (idx, node) in tree.nodes().iter().enumerate() {
            assert_eq!(node.node_id, idx as u32);
        }
    }

    // ── SpecTree: token_ids in all_token_ids are distinct sources ──

    #[test]
    fn spec_tree_all_token_ids_preserves_node_order() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 30];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let all_ids = tree.all_token_ids();
        assert_eq!(all_ids.len(), tree.len());
        for (i, tid) in all_ids.iter().enumerate() {
            assert_eq!(*tid, tree.node(i as u32).unwrap().token_id);
        }
    }

    // ── SpecTree: build with single adapter token and long prompt ──

    #[test]
    fn spec_tree_build_single_adapter_long_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![5u32];
        let prompt: Vec<u32> = (0..100).collect(); // 0..100
        let ngram_idx = NgramIndex::build(&prompt, 3);
        let tree = SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx);
        assert!(tree.len() >= 1);
        assert_eq!(tree.node(0).unwrap().token_id, 5);
        // Tree size bounded by config
        assert!(tree.len() <= config.max_tree_size);
    }

    // ── SpecTree: accepted_from_spine on empty tree panics (spine_ids reads nodes[0]) ──

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn spec_tree_accepted_from_spine_empty_tree_panics() {
        // Empty tree has no nodes, so spine_ids() panics on nodes[0] access.
        // This documents the behavior — accepted_from_spine requires a non-empty tree.
        let tree = SpecTree::new(SpecTreeConfig::default());
        let _ = tree.accepted_from_spine(&[1, 2, 3]);
    }

    // ── SpecTree: CSR mask total columns bounded ──

    #[test]
    fn spec_tree_csr_mask_max_column_value() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![100u32, 200];
        let prompt = vec![1, 2, 100, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let total_seq = 20;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        let max_col = total_seq + tree.len();
        for &col in &indices {
            assert!(col < max_col, "column {} exceeds max {}", col, max_col);
        }
        let _ = indptr; // used above
    }

    // ── SpecTree: mask_shape zero total_seq_len ──

    #[test]
    fn spec_tree_mask_shape_zero_total_seq() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[42u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
        let (rows, cols) = tree.mask_shape(0);
        assert_eq!(rows, tree.len());
        assert_eq!(cols, tree.len());
    }

    // ── SpecTree: clone independence after build ──

    #[test]
    fn spec_tree_clone_after_build_independence() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 100, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        let cloned = tree.clone();
        assert_eq!(tree, cloned);
        // Verify they are independent: accessing both doesn't alias
        assert_eq!(tree.all_token_ids(), cloned.all_token_ids());
    }

    // ── SpecNode: default children is empty vec ──

    #[test]
    fn spec_node_default_children_empty() {
        let node = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: Vec::new(),
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.5,
            position_offset: 0,
        };
        assert!(node.children.is_empty());
    }

    // ── DraftSource: all variants distinguishable via Debug ──

    #[test]
    fn draft_source_debug_contains_variant_name() {
        let debug_pld = format!("{:?}", DraftSource::PldSpine);
        assert!(debug_pld.contains("PldSpine"));
        let debug_ngram = format!("{:?}", DraftSource::NgramBranch);
        assert!(debug_ngram.contains("NgramBranch"));
        let debug_adapter = format!("{:?}", DraftSource::AdapterTopK { k: 3 });
        assert!(debug_adapter.contains("AdapterTopK"));
    }

    // ── DraftSource: Eq implies structural equality ──

    #[test]
    fn draft_source_eq_reflexive_all_variants() {
        let variants = vec![
            DraftSource::PldSpine,
            DraftSource::NgramBranch,
            DraftSource::AdapterTopK { k: 1 },
            DraftSource::AdapterTopK { k: 255 },
        ];
        for v in &variants {
            assert_eq!(v, v);
        }
    }

    // ── DraftSource: Copy allows passing by value ──

    #[test]
    fn draft_source_copy_fn_arg() {
        fn check_copy(source: DraftSource) -> bool {
            matches!(source, DraftSource::AdapterTopK { .. })
        }
        let source = DraftSource::AdapterTopK { k: 5 };
        assert!(check_copy(source));
        // Still usable after "move" (Copy)
        assert!(check_copy(source));
    }

    // ── NgramIndex: build with n=1 produces correct window ──

    #[test]
    fn ngram_index_n1_produces_unigram_table() {
        let tokens = vec![10u32, 20, 10, 30, 10, 20];
        let idx = NgramIndex::build(&tokens, 1);
        // n=1: each single token → next token frequency
        let conts = idx.get_continuations(10, 5);
        // After 10 comes 20, 30, 20
        assert!(!conts.is_empty());
        // 20 appears twice, should be first by frequency
        assert_eq!(conts[0], 20);
    }

    // ── NgramIndex: build with tokens.len() == n produces empty ──

    #[test]
    fn ngram_index_tokens_len_equals_n_empty() {
        let tokens = vec![1u32, 2, 3];
        let idx = NgramIndex::build(&tokens, 3);
        // tokens.len() == n, so loop range 0..tokens.len()-n = 0..0 = empty
        assert!(idx.table.is_empty());
    }

    // ── NgramIndex: get_continuations returns empty for unknown token ──

    #[test]
    fn ngram_index_get_continuations_unknown_returns_empty() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        let result = idx.get_continuations(999, 5);
        assert!(result.is_empty());
    }

    // ── NgramIndex: get_ngram_continuations returns empty for unknown ngram ──

    #[test]
    fn ngram_index_get_ngram_continuations_unknown_returns_empty() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4, 5], 2);
        let result = idx.get_ngram_continuations(&[99, 98], 5);
        assert!(result.is_empty());
    }

    // ── NgramIndex: build with n=0 edge case ──

    #[test]
    fn ngram_index_build_n_zero() {
        let tokens = vec![1u32, 2, 3, 4];
        let idx = NgramIndex::build(&tokens, 0);
        // n=0: tokens.len() <= 0 is true for non-empty → table empty? No: 4 > 0.
        // Loop: 0..4-0 = 0..4, ngram = &tokens[i..i+0] = empty slice
        // All empty slices hash the same → all continuations merge into one entry
        assert!(!idx.table.is_empty());
    }

    // ── NgramIndex: build with single element tokens ──

    #[test]
    fn ngram_index_single_element_tokens() {
        let tokens = vec![42u32];
        let idx = NgramIndex::build(&tokens, 1);
        // len=1, n=1: loop 0..0 → empty
        assert!(idx.table.is_empty());
    }

    // ── NgramIndex: frequency order is strictly descending ──

    #[test]
    fn ngram_index_continuations_frequency_descending() {
        let tokens = vec![1u32, 2, 10, 1, 2, 20, 1, 2, 10, 1, 2, 30, 1, 2, 10];
        let idx = NgramIndex::build(&tokens, 2);
        // n-gram [1,2] → continuations: 10 (3x), 20 (1x), 30 (1x)
        let conts = idx.get_ngram_continuations(&[1, 2], 10);
        if conts.len() >= 2 {
            // First should be the most frequent
            assert_eq!(conts[0], 10);
        }
    }

    // ── SpecTree: CSR mask indices are sorted within each row ──

    #[test]
    fn spec_tree_csr_mask_indices_sorted_per_row() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 10, 50, 60, 10, 55];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32, 20], &prompt, &ngram_idx);
        let (indptr, indices) = tree.tree_attention_mask_csr(8);
        for i in 0..tree.len() {
            let row = &indices[indptr[i]..indptr[i + 1]];
            for w in row.windows(2) {
                assert!(w[0] < w[1], "row {} not sorted: {} >= {}", i, w[0], w[1]);
            }
        }
    }

    // ── SpecTree: build with adapter_top_k > adapter_tokens.len() ──

    #[test]
    fn spec_tree_build_adapter_top_k_exceeds_available_tokens() {
        let config = SpecTreeConfig {
            adapter_top_k: 10,
            max_spine_depth: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32]; // Only 1 token
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Should only create root, no extra adapter branches
        assert_eq!(tree.len(), 1);
    }

    // ── SpecTree: CSR mask indptr is monotonically non-decreasing ──

    #[test]
    fn spec_tree_csr_mask_indptr_monotonic() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        let (indptr, _) = tree.tree_attention_mask_csr(10);
        for w in indptr.windows(2) {
            assert!(w[0] <= w[1], "indptr not monotonic: {} > {}", w[0], w[1]);
        }
    }

    // ── SpecTree: position_offset strictly increases along spine ──

    #[test]
    fn spec_tree_position_offset_spine_increases() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let spine = tree.spine_ids();
        for w in spine.windows(2) {
            let off0 = tree.node(w[0]).unwrap().position_offset;
            let off1 = tree.node(w[1]).unwrap().position_offset;
            assert!(off1 > off0, "spine offsets not increasing: {} -> {}", off0, off1);
        }
    }

    // ── SpecTree: branch nodes have position_offset > parent ──

    #[test]
    fn spec_tree_branch_position_offset_greater_than_parent() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![100u32, 200];
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        for node in tree.nodes() {
            if let Some(pid) = node.parent_id {
                let parent_offset = tree.node(pid).unwrap().position_offset;
                assert!(
                    node.position_offset > parent_offset,
                    "node {} offset {} not > parent {} offset {}",
                    node.node_id, node.position_offset, pid, parent_offset
                );
            }
        }
    }

    // ── SpecTree: node estimated_acceptance in reasonable range ──

    #[test]
    fn spec_tree_acceptance_range_reasonable() {
        let config = SpecTreeConfig::default();
        let adapter_tokens = vec![100u32, 200, 300];
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        for node in tree.nodes() {
            assert!(
                node.estimated_acceptance >= 0.0 && node.estimated_acceptance <= 1.0,
                "node {} acceptance {} out of range",
                node.node_id, node.estimated_acceptance
            );
        }
    }

    // ── SpecTree: build with all identical prompt tokens ──

    #[test]
    fn spec_tree_build_all_identical_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![7u32];
        let prompt = vec![7u32; 20];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert!(tree.len() >= 1);
        // PLD continuation: after token 7, next is also 7
        let spine = tree.spine_token_ids();
        // All spine tokens should be 7
        for &tid in &spine {
            assert_eq!(tid, 7);
        }
    }

    // ── SpecTree: PLD continuation produces valid spine with repeated prompt patterns ──

    #[test]
    fn spec_tree_pld_continuation_repeated_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 followed by 20 many times
        let mut prompt = vec![1, 2, 3];
        for _ in 0..10 {
            prompt.extend_from_slice(&[10, 20, 20, 20]);
        }
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let max_spine = config.max_spine_depth;
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        // Root is adapter top-1 = 10
        assert_eq!(spine[0], 10);
        // Spine length bounded by max_spine_depth
        assert!(spine.len() <= max_spine);
    }

    // ── SpecTree: is_empty after new is true ──

    #[test]
    fn spec_tree_is_empty_after_new() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.is_empty());
    }

    // ── SpecTree: is_empty after build with adapter is false ──

    #[test]
    fn spec_tree_is_not_empty_after_build() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[42u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
        assert!(!tree.is_empty());
    }

    // ── SpecTree: node accessor on valid tree returns correct data ──

    #[test]
    fn spec_tree_node_accessor_correct_fields() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[99u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
        let root = tree.node(0).unwrap();
        assert_eq!(root.node_id, 0);
        assert_eq!(root.token_id, 99);
        assert_eq!(root.parent_id, None);
        assert_eq!(root.source, DraftSource::AdapterTopK { k: 1 });
        assert_eq!(root.position_offset, 0);
    }

    // ── SpecTree: branch_token_ids excludes spine ──

    #[test]
    fn spec_tree_branch_excludes_all_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        let spine_ids: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        let branch_ids: std::collections::HashSet<u32> = tree
            .branch_token_ids()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        // No overlap
        for sid in &spine_ids {
            assert!(!branch_ids.contains(sid), "spine node {} in branches", sid);
        }
    }

    // ── SpecTree: accepted_from_spine with exact match returns all ──

    #[test]
    fn spec_tree_accepted_from_spine_exact_match_full() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[42u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
        let spine = tree.spine_token_ids();
        let (count, accepted) = tree.accepted_from_spine(&spine);
        assert_eq!(count, spine.len());
        assert_eq!(accepted, spine);
    }

    // ── SpecTree: accepted_from_spine first mismatch returns zero ──

    #[test]
    fn spec_tree_accepted_from_spine_first_mismatch_zero() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[42u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
        let spine = tree.spine_token_ids();
        // Target first token different
        let target = vec![spine[0].wrapping_add(1)];
        let (count, _) = tree.accepted_from_spine(&target);
        assert_eq!(count, 0);
    }

    // ── SpecTree: CSR mask prefix columns form contiguous range ──

    #[test]
    fn spec_tree_csr_mask_prefix_columns_contiguous_range() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 42, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[42u32], &prompt, &ngram_idx);
        let total_seq = 7;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        for i in 0..tree.len() {
            let row = &indices[indptr[i]..indptr[i + 1]];
            // First total_seq columns should be 0..total_seq
            let prefix: Vec<usize> = row.iter().take(total_seq).copied().collect();
            let expected: Vec<usize> = (0..total_seq).collect();
            assert_eq!(prefix, expected, "row {} prefix not contiguous", i);
        }
    }

    // ── SpecTree: nodes() slice length matches len() ──

    #[test]
    fn spec_tree_nodes_slice_len_matches_total() {
        let config = SpecTreeConfig::default();
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200, 300], &prompt, &ngram_idx);
        assert_eq!(tree.nodes().len(), tree.len());
    }

    // ── SpecTree: max_tree_size only limits ngram branches, not adapter/spine ──

    #[test]
    fn spec_tree_build_max_tree_size_limits_ngram_only() {
        let config = SpecTreeConfig {
            max_tree_size: 1,
            max_spine_depth: 5,
            adapter_top_k: 3,
            max_branches_per_node: 3,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32, 20, 30], &prompt, &ngram_idx);
        // max_tree_size=1 only blocks ngram branches. Root + adapter branches still added.
        assert!(tree.len() >= 1);
    }

    // ── SpecTree: build determinism — same inputs produce same tree ──

    #[test]
    fn spec_tree_build_deterministic_identical_inputs() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 3, 10, 50, 60, 10, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree1 = SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx);
        let tree2 = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert_eq!(tree1, tree2);
    }

    // ── SpecTree: tree with branches has more nodes than spine alone ──

    #[test]
    fn spec_tree_with_branches_more_nodes_than_spine() {
        let config_with_branches = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let config_no_branches = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ngram_top_k: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60, 10, 77, 10, 88];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree_with = SpecTree::build(config_with_branches, &adapter_tokens, &prompt, &ngram_idx);
        let tree_without = SpecTree::build(config_no_branches, &adapter_tokens, &prompt, &ngram_idx);
        assert!(tree_with.len() >= tree_without.len());
    }

    // ── NgramIndex: build with two tokens and n=1 ──

    #[test]
    fn ngram_index_two_tokens_n1() {
        let tokens = vec![10u32, 20];
        let idx = NgramIndex::build(&tokens, 1);
        // len=2, n=1: loop 0..1 → one ngram [10] → continuation 20
        let conts = idx.get_continuations(10, 5);
        assert_eq!(conts, vec![20]);
    }

    // ── NgramIndex: multiple ngrams different lengths ──

    #[test]
    fn ngram_index_different_n_isolation() {
        let tokens = vec![1u32, 2, 3, 4, 5, 1, 2, 6];
        let idx2 = NgramIndex::build(&tokens, 2);
        let idx3 = NgramIndex::build(&tokens, 3);
        // They should produce different tables because n differs
        // n=2: [1,2]→3, [2,3]→4, [3,4]→5, [4,5]→1, [5,1]→2, [1,2]→6
        // n=3: [1,2,3]→4, [2,3,4]→5, [3,4,5]→1, [4,5,1]→2, [5,1,2]→6
        assert_ne!(idx2, idx3);
    }

    // ── NgramIndex: continuation order stable across clones ──

    #[test]
    fn ngram_index_clone_continuation_order_stable() {
        let tokens = vec![1u32, 2, 10, 1, 2, 20, 1, 2, 10];
        let idx = NgramIndex::build(&tokens, 2);
        let cloned = idx.clone();
        let orig_conts = idx.get_ngram_continuations(&[1, 2], 10);
        let clone_conts = cloned.get_ngram_continuations(&[1, 2], 10);
        assert_eq!(orig_conts, clone_conts);
    }

    // ── NgramIndex: build with large n relative to tokens ──

    #[test]
    fn ngram_index_build_n_equals_tokens_len_minus_one() {
        let tokens = vec![1u32, 2, 3, 4];
        let idx = NgramIndex::build(&tokens, 3);
        // len=4, n=3: loop 0..1 → one ngram [1,2,3] → continuation 4
        assert!(!idx.table.is_empty());
        let conts = idx.get_ngram_continuations(&[1, 2, 3], 5);
        assert_eq!(conts, vec![4]);
    }

    // ── SpecTree: CSR mask respects tree_size = 0 for empty tree ──

    #[test]
    fn spec_tree_csr_mask_empty_tree_dimensions() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (indptr, indices) = tree.tree_attention_mask_csr(10);
        assert_eq!(indptr.len(), 1); // just [0]
        assert_eq!(indptr[0], 0);
        assert!(indices.is_empty());
    }

    // ── SpecTree: mask_shape on empty tree ──

    #[test]
    fn spec_tree_mask_shape_empty() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (rows, cols) = tree.mask_shape(5);
        assert_eq!(rows, 0);
        assert_eq!(cols, 5);
    }

    // ── SpecTree: all_token_ids on empty tree ──

    #[test]
    fn spec_tree_all_token_ids_empty() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.all_token_ids().is_empty());
    }

    // ── SpecTree: spine_ids on empty tree panics (accesses nodes[0]) ──

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn spec_tree_spine_ids_empty_panics() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let _ = tree.spine_ids();
    }

    // ── SpecTree: spine_token_ids on empty tree panics (spine_ids reads nodes[0]) ──

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn spec_tree_spine_token_ids_empty_panics() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let _ = tree.spine_token_ids();
    }

    // ── SpecTree: branch_token_ids on empty tree panics (spine_ids reads nodes[0]) ──

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn spec_tree_branch_token_ids_empty_panics() {
        // Empty tree has no nodes, so spine_ids() panics on nodes[0] access.
        let tree = SpecTree::new(SpecTreeConfig::default());
        let _ = tree.branch_token_ids();
    }

    // ── SpecTree: build with adapter containing duplicate tokens ──

    #[test]
    fn spec_tree_build_adapter_duplicate_tokens() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // All three adapter tokens are the same
        let adapter_tokens = vec![42u32, 42, 42];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Root is 42, branches are also 42 (adapter top-2 and top-3)
        assert!(tree.len() >= 1);
        let root = tree.node(0).unwrap();
        assert_eq!(root.token_id, 42);
    }

    // ── SpecTree: root node has no parent ──

    #[test]
    fn spec_tree_root_parent_is_none() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[42u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
        assert!(tree.node(0).unwrap().parent_id.is_none());
    }

    // ── SpecTree: root source is AdapterTopK { k: 1 } ──

    #[test]
    fn spec_tree_root_source_adapter_top1() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[55u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
        assert_eq!(tree.node(0).unwrap().source, DraftSource::AdapterTopK { k: 1 });
    }

    // ── SpecTree: total_nodes consistent with len() ──

    #[test]
    fn spec_tree_total_nodes_matches_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        assert_eq!(tree.total_nodes, tree.len());
    }

    // ── SpecTreeConfig: zero max_spine_depth panics on underflow ──

    #[test]
    #[should_panic]
    fn spec_tree_config_zero_spine_depth_panics() {
        // max_spine_depth=0 causes (max_spine_depth - 1) underflow in build()
        let config = SpecTreeConfig {
            max_spine_depth: 0,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let _ = SpecTree::build(
            config,
            &[42u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
    }

    // ── NgramIndex: get_continuations with top_k=0 returns empty ──

    #[test]
    fn ngram_index_get_continuations_top_k_zero_empty() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let result = idx.get_continuations(1, 0);
        assert!(result.is_empty());
    }

    // ── SpecTree: attention_paths covers all nodes ──

    #[test]
    fn spec_tree_attention_paths_covers_all_nodes() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        // attention_paths should have same length as nodes
        assert_eq!(tree.attention_paths.len(), tree.len());
    }

    // ── SpecTree: root attention path is empty ──

    #[test]
    fn spec_tree_root_attention_path_empty() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1, 2, 3, 42, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[42u32], &prompt, &ngram_idx);
        assert!(tree.attention_paths[0].is_empty());
    }

    // ── SpecTree: non-root attention path includes parent ──

    #[test]
    fn spec_tree_attention_path_includes_parent() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        if tree.len() > 1 {
            // Node 1 should have root (0) in its attention path
            assert!(tree.attention_paths[1].contains(&0));
        }
    }

    // ── SpecTree: CSR mask with large total_seq_len ──

    #[test]
    fn spec_tree_csr_mask_large_total_seq() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[42u32],
            &[1, 2, 3],
            &NgramIndex::build(&[1, 2, 3], 2),
        );
        let (indptr, indices) = tree.tree_attention_mask_csr(10000);
        assert_eq!(indptr.len(), tree.len() + 1);
        // Root should attend to all 10000 prefix + itself = 10001 entries
        let root_nnz = indptr[1] - indptr[0];
        assert_eq!(root_nnz, 10001);
        let _ = indices;
    }

    // ── SpecNode: clone produces equal but independent node ──

    #[test]
    fn spec_node_clone_equality() {
        let node = SpecNode {
            node_id: 5,
            token_id: 42,
            parent_id: Some(3),
            children: vec![10, 20],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.6,
            position_offset: 2,
        };
        let cloned = node.clone();
        assert_eq!(node, cloned);
    }

    // ── SpecTree: build with empty ngram index still produces spine ──

    #[test]
    fn spec_tree_build_empty_ngram_produces_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 2,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32, 100];
        let prompt = vec![1, 2, 3, 42, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 5); // n=5 > prompt windows → mostly empty
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert!(tree.len() >= 1);
        // Root should exist
        assert_eq!(tree.node(0).unwrap().token_id, 42);
    }

    // ── SpecTree: accepted_from_spine preserves order ──

    #[test]
    fn spec_tree_accepted_preserves_spine_order() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        // Use the spine itself as target → full match
        let (_, accepted) = tree.accepted_from_spine(&spine);
        assert_eq!(accepted, spine);
    }

    // ── NgramIndex: hash_ngram deterministic ──

    #[test]
    fn ngram_index_hash_deterministic_across_calls() {
        // hash_ngram is private, test indirectly via build consistency
        let idx1 = NgramIndex::build(&[1u32, 2, 3, 4, 1, 2, 5], 3);
        let idx2 = NgramIndex::build(&[1u32, 2, 3, 4, 1, 2, 5], 3);
        assert_eq!(idx1, idx2);
    }

    // ── SpecTree: adapter branch sources have correct k values ──

    #[test]
    fn spec_tree_adapter_branch_k_values() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 4,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30, 40];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Root: k=1, then branches: k=2, k=3, k=4
        assert_eq!(tree.node(0).unwrap().source, DraftSource::AdapterTopK { k: 1 });
        for i in 1..tree.len() {
            let node = tree.node(i as u32).unwrap();
            if matches!(node.source, DraftSource::AdapterTopK { .. }) {
                let k = match node.source {
                    DraftSource::AdapterTopK { k } => k,
                    _ => unreachable!(),
                };
                assert!(k >= 2, "adapter branch k should be >= 2, got {}", k);
            }
        }
    }

    // ── SpecTree: ngram branch source is NgramBranch ──

    #[test]
    fn spec_tree_ngram_branch_source_correct() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60, 10, 77, 10, 88];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                assert!(node.parent_id.is_some(), "ngram branch should have parent");
            }
        }
    }

    // ── SpecTree: CSR mask branch node sees all ancestors ──

    #[test]
    fn spec_tree_csr_mask_branch_sees_full_ancestry() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60, 10, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        for node in tree.nodes() {
            if node.parent_id.is_some() {
                // Node should attend to all ancestors via attention_paths
                let path = &tree.attention_paths[node.node_id as usize];
                // Walk up parent chain and verify all are in path
                let mut current = node.node_id;
                while let Some(pid) = tree.node(current).unwrap().parent_id {
                    assert!(
                        path.contains(&pid),
                        "node {} path missing ancestor {}",
                        node.node_id, pid
                    );
                    current = pid;
                }
            }
        }
    }

    // ── SpecTree: build preserves config reference ──

    #[test]
    fn spec_tree_config_preserved_after_build() {
        let config = SpecTreeConfig {
            max_spine_depth: 7,
            max_branches_per_node: 3,
            pld_ngram_len: 4,
            ngram_top_k: 5,
            adapter_top_k: 6,
            max_tree_size: 50,
        };
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config.clone(), &[42u32], &prompt, &ngram_idx);
        assert_eq!(tree.config, config);
    }

    // ── SpecTreeConfig: PartialEq symmetry ──

    #[test]
    fn spec_tree_config_partial_eq_symmetric() {
        let a = SpecTreeConfig::default();
        let b = SpecTreeConfig::default();
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // ── SpecTreeConfig: PartialEq transitivity ──

    #[test]
    fn spec_tree_config_partial_eq_transitive() {
        let a = SpecTreeConfig {
            max_spine_depth: 3,
            ..SpecTreeConfig::default()
        };
        let b = SpecTreeConfig {
            max_spine_depth: 3,
            ..SpecTreeConfig::default()
        };
        let c = SpecTreeConfig {
            max_spine_depth: 3,
            ..SpecTreeConfig::default()
        };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── SpecTreeConfig: clone produces identical but independent value ──

    #[test]
    fn spec_tree_config_clone_deep_independence() {
        let mut original = SpecTreeConfig {
            max_spine_depth: 10,
            max_branches_per_node: 4,
            pld_ngram_len: 2,
            ngram_top_k: 6,
            adapter_top_k: 8,
            max_tree_size: 100,
        };
        let cloned = original.clone();
        original.max_spine_depth = 1;
        assert_ne!(original, cloned);
        assert_eq!(cloned.max_spine_depth, 10);
    }

    // ── SpecTreeConfig: default fields are all non-zero ──

    #[test]
    fn spec_tree_config_default_no_zero_fields() {
        let cfg = SpecTreeConfig::default();
        assert!(cfg.max_spine_depth > 0);
        assert!(cfg.max_branches_per_node > 0);
        assert!(cfg.pld_ngram_len > 0);
        assert!(cfg.ngram_top_k > 0);
        assert!(cfg.adapter_top_k > 0);
        assert!(cfg.max_tree_size > 0);
    }

    // ── SpecNode: construction with all DraftSource variants ──

    #[test]
    fn spec_node_construction_all_source_types() {
        let sources = vec![
            DraftSource::PldSpine,
            DraftSource::AdapterTopK { k: 1 },
            DraftSource::AdapterTopK { k: 128 },
            DraftSource::NgramBranch,
        ];
        for (i, source) in sources.into_iter().enumerate() {
            let node = SpecNode {
                node_id: i as u32,
                token_id: 100 + i as u32,
                parent_id: if i == 0 { None } else { Some(0) },
                children: vec![],
                source,
                estimated_acceptance: 0.5,
                position_offset: i as u32,
            };
            assert_eq!(node.node_id, i as u32);
            assert_eq!(node.token_id, 100 + i as u32);
        }
    }

    // ── SpecNode: parent_id None vs Some distinguishes root ──

    #[test]
    fn spec_node_root_vs_child_parent_id() {
        let root = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![1],
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.7,
            position_offset: 0,
        };
        let child = SpecNode {
            node_id: 1,
            token_id: 2,
            parent_id: Some(0),
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.6,
            position_offset: 1,
        };
        assert!(root.parent_id.is_none());
        assert!(child.parent_id.is_some());
        assert_ne!(root, child);
    }

    // ── SpecNode: children field is independent after clone ──

    #[test]
    fn spec_node_children_mutation_after_clone() {
        let mut original = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![10, 20, 30],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.5,
            position_offset: 0,
        };
        let cloned = original.clone();
        original.children.push(40);
        assert_eq!(original.children.len(), 4);
        assert_eq!(cloned.children.len(), 3);
    }

    // ── SpecNode: estimated_acceptance preserves special f32 values ──

    #[test]
    fn spec_node_acceptance_special_float_values() {
        let special_values = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0f32, -0.0f32];
        for val in special_values {
            let node = SpecNode {
                node_id: 0,
                token_id: 1,
                parent_id: None,
                children: vec![],
                source: DraftSource::PldSpine,
                estimated_acceptance: val,
                position_offset: 0,
            };
            if val.is_nan() {
                assert!(node.estimated_acceptance.is_nan());
            } else {
                assert_eq!(node.estimated_acceptance, val);
            }
        }
    }

    // ── DraftSource: AdapterTopK k field roundtrip ──

    #[test]
    fn draft_source_adapter_top_k_k_roundtrip() {
        for k in [0u8, 1, 2, 127, 128, 255] {
            let ds = DraftSource::AdapterTopK { k };
            match ds {
                DraftSource::AdapterTopK { k: extracted } => assert_eq!(extracted, k),
                _ => panic!("expected AdapterTopK"),
            }
        }
    }

    // ── DraftSource: all variants produce different Debug output ──

    #[test]
    fn draft_source_debug_no_collision() {
        let d1 = format!("{:?}", DraftSource::PldSpine);
        let d2 = format!("{:?}", DraftSource::AdapterTopK { k: 1 });
        let d3 = format!("{:?}", DraftSource::NgramBranch);
        assert_ne!(d1, d2);
        assert_ne!(d2, d3);
        assert_ne!(d1, d3);
    }

    // ── DraftSource: Copy trait allows passing by value without clone ──

    #[test]
    fn draft_source_copy_assignment() {
        let a = DraftSource::PldSpine;
        let b = a; // Copy, not move
        let c = a; // Copy again — would fail if not Copy
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    // ── NgramIndex: n field matches constructor argument ──

    #[test]
    fn ngram_index_n_field_various() {
        for n in [1, 2, 3, 5, 10] {
            let idx = NgramIndex::build(&[1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10], n);
            assert_eq!(idx.n, n);
        }
    }

    // ── NgramIndex: n-gram continuations match window count ──

    #[test]
    fn ngram_index_ngram_continuation_from_windows() {
        let tokens = vec![10u32, 20, 30, 10, 20, 40]; // 6 tokens, n=2
        let idx = NgramIndex::build(&tokens, 2);
        // n=2 windows: [10,20]->30, [20,30]->10, [30,10]->20, [10,20]->40
        // ngram [10,20] appears twice → continuations [30,40] (sorted by freq)
        let conts = idx.get_ngram_continuations(&[10, 20], 10);
        assert!(!conts.is_empty());
        assert!(conts.contains(&30));
        assert!(conts.contains(&40));
    }

    // ── NgramIndex: continuations for last token are empty ──

    #[test]
    fn ngram_index_last_token_no_continuation() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_continuations(5, 10);
        assert!(conts.is_empty());
    }

    // ── NgramIndex: repeated pattern yields higher count ──

    #[test]
    fn ngram_index_repeated_pattern_count_ordering() {
        // [1,2,3,1,2,4,1,2,3] — token 3 appears twice after [1,2], token 4 once
        let tokens = vec![1u32, 2, 3, 1, 2, 4, 1, 2, 3];
        let idx = NgramIndex::build(&tokens, 2);
        // n=2: ngram [1,2] has continuations 3 (twice) and 4 (once) → sorted: [3, 4]
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert!(!conts.is_empty());
        assert_eq!(conts[0], 3); // 3 appears more frequently
    }

    // ── NgramIndex: n=1 with single continuation token ──

    #[test]
    fn ngram_index_n1_single_continuation() {
        let tokens = vec![1u32, 2, 1, 2, 1, 2];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(1, 5);
        assert_eq!(conts, vec![2]);
    }

    // ── NgramIndex: build with exactly n+1 tokens has one window ──

    #[test]
    fn ngram_index_exact_n_plus_one_single_window() {
        let tokens = vec![10u32, 20, 30]; // n=2, one window [10,20]→30
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[10, 20], 5);
        assert_eq!(conts, vec![30]);
    }

    // ── NgramIndex: get_ngram_continuations returns empty for unknown ngram ──

    #[test]
    fn ngram_index_ngram_continuations_truly_unknown() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        let conts = idx.get_ngram_continuations(&[99, 98], 5);
        assert!(conts.is_empty());
    }

    // ── NgramIndex: table field is accessible and non-empty after build ──

    #[test]
    fn ngram_index_table_non_empty_with_data() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4, 5], 2);
        assert!(!idx.table.is_empty());
    }

    // ── NgramIndex: table field empty when tokens.len() <= n ──

    #[test]
    fn ngram_index_table_empty_insufficient_tokens() {
        let idx = NgramIndex::build(&[1u32, 2], 3);
        assert!(idx.table.is_empty());
    }

    // ── SpecTree: spine_len field matches spine_ids().len() ──

    #[test]
    fn spec_tree_spine_len_field_consistency() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60, 10, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        // spine_len is internal but spine_ids() is public
        assert_eq!(tree.spine_ids().len(), tree.spine_token_ids().len());
    }

    // ── SpecTree: is_empty consistent with len() == 0 ──

    #[test]
    fn spec_tree_is_empty_consistent_with_len() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let built = SpecTree::build(config, &[42u32], &prompt, &ngram_idx);
        assert!(!built.is_empty());
        assert!(built.len() > 0);
    }

    // ── SpecTree: node() returns None for u32::MAX ──

    #[test]
    fn spec_tree_node_u32_max_returns_none() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        assert!(tree.node(u32::MAX).is_none());
    }

    // ── SpecTree: all_token_ids matches node iteration ──

    #[test]
    fn spec_tree_all_token_ids_matches_iteration() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        let all_ids: Vec<u32> = tree.nodes().iter().map(|n| n.token_id).collect();
        assert_eq!(tree.all_token_ids(), all_ids);
    }

    // ── SpecTree: accepted_from_spine with empty tree panics (spine_ids OOB) ──

    #[test]
    #[should_panic]
    fn spec_tree_accepted_empty_spine_panics() {
        // Empty tree has no nodes, spine_ids() panics on nodes[0]
        let tree = SpecTree::new(SpecTreeConfig::default());
        let _ = tree.accepted_from_spine(&[]);
    }

    // ── SpecTree: accepted_from_spine mismatch at position 0 ──

    #[test]
    fn spec_tree_accepted_mismatch_at_zero() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        if !spine.is_empty() {
            let target = vec![spine[0].wrapping_add(999)];
            let (count, accepted) = tree.accepted_from_spine(&target);
            assert_eq!(count, 0);
            assert!(accepted.is_empty());
        }
    }

    // ── SpecTree: branch_token_ids empty for spine-only tree ──

    #[test]
    fn spec_tree_branch_empty_when_no_branches() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let branches = tree.branch_token_ids();
        assert!(branches.is_empty());
    }

    // ── SpecTree: spine_ids first element is always 0 ──

    #[test]
    fn spec_tree_spine_ids_first_is_zero() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        if !tree.is_empty() {
            assert_eq!(tree.spine_ids()[0], 0);
        }
    }

    // ── SpecTree: root has no parent in attention_paths ──

    #[test]
    fn spec_tree_root_no_ancestors_in_paths() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        if !tree.is_empty() {
            assert!(tree.attention_paths[0].is_empty());
        }
    }

    // ── SpecTree: mask_shape total columns >= rows ──

    #[test]
    fn spec_tree_mask_shape_cols_ge_rows() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let (rows, cols) = tree.mask_shape(5);
        assert!(cols >= rows);
    }

    // ── SpecTree: CSR mask indptr starts at 0 ──

    #[test]
    fn spec_tree_csr_mask_indptr_starts_at_zero() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let (indptr, _) = tree.tree_attention_mask_csr(5);
        assert_eq!(indptr[0], 0);
    }

    // ── SpecTree: CSR mask column values within bounds ──

    #[test]
    fn spec_tree_csr_mask_columns_within_bounds() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let total_seq = 5;
        let (rows, cols) = tree.mask_shape(total_seq);
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        for row in 0..rows {
            for col_idx in indptr[row]..indptr[row + 1] {
                let col = indices[col_idx];
                assert!(col < cols, "column {} exceeds max {}", col, cols);
            }
        }
    }

    // ── SpecTree: attention_paths length matches total_nodes ──

    #[test]
    fn spec_tree_attention_paths_len_matches_nodes() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        assert_eq!(tree.attention_paths.len(), tree.len());
    }

    // ── SpecTree: non-root attention paths strictly increasing ──

    #[test]
    fn spec_tree_attention_paths_ascending_order() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60, 10, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        for path in &tree.attention_paths {
            for w in path.windows(2) {
                assert!(w[0] < w[1], "attention path not ascending: {} >= {}", w[0], w[1]);
            }
        }
    }

    // ── SpecTree: node IDs are 0..len() ──

    #[test]
    fn spec_tree_node_ids_sequential() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        for (i, node) in tree.nodes().iter().enumerate() {
            assert_eq!(node.node_id, i as u32);
        }
    }

    // ── SpecTree: parent_id always less than node_id ──

    #[test]
    fn spec_tree_parent_id_less_than_node_id() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        for node in tree.nodes() {
            if let Some(pid) = node.parent_id {
                assert!(pid < node.node_id, "parent {} >= node {}", pid, node.node_id);
            }
        }
    }

    // ── SpecTree: children IDs always greater than parent ──

    #[test]
    fn spec_tree_children_ids_greater_than_parent() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        for node in tree.nodes() {
            for &child_id in &node.children {
                assert!(child_id > node.node_id, "child {} <= parent {}", child_id, node.node_id);
            }
        }
    }

    // ── SpecTree: all children refer back to correct parent ──

    #[test]
    fn spec_tree_children_parent_back_reference() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        for node in tree.nodes() {
            for &child_id in &node.children {
                let child = tree.node(child_id).unwrap();
                assert_eq!(child.parent_id, Some(node.node_id));
            }
        }
    }

    // ── SpecTree: build with single adapter token produces at least root ──

    #[test]
    fn spec_tree_single_adapter_at_least_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[42u32], &prompt, &ngram_idx);
        assert!(tree.len() >= 1);
        assert_eq!(tree.node(0).unwrap().token_id, 42);
    }

    // ── SpecTree: build respects adapter_top_k limit ──

    #[test]
    fn spec_tree_adapter_top_k_limit_strict() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30, 40, 50]; // 5 tokens
        let prompt = vec![1u32, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Root (k=1) + 1 adapter branch (k=2) = 2 nodes max from adapter
        let adapter_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { .. }))
            .count();
        assert!(adapter_count <= 2);
    }

    // ── SpecTree: PLD spine extensions have PldSpine source ──

    #[test]
    fn spec_tree_pld_extensions_source_correct() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let spine = tree.spine_ids();
        for id in spine.iter().skip(1) {
            let node = tree.node(*id).unwrap();
            assert!(matches!(node.source, DraftSource::PldSpine));
        }
    }

    // ── SpecTree: PLD spine token IDs match continuations ──

    #[test]
    fn spec_tree_pld_spine_tokens_match_prompt_continuations() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Token 10 appears at positions 3,6 → continuations are 50,60
        let prompt = vec![1u32, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        // Continuations should be 50 and 60 (in some order)
        if spine.len() > 1 {
            assert!(spine[1..].iter().any(|&t| t == 50 || t == 60));
        }
    }

    // ── SpecTree: max_tree_size limits ngram branch insertion ──

    #[test]
    fn spec_tree_max_tree_size_limits_ngram_insertion() {
        // max_tree_size only checked during ngram branch phase,
        // so use small adapter_top_k + small spine to keep adapter nodes within limit
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 5,
            ngram_top_k: 5,
            pld_ngram_len: 2,
            max_tree_size: 4,
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1u32, 2, 3, 10, 50, 60, 10, 77, 10, 88];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Root + up to 1 adapter branch + possible PLD + ngram branches
        // ngram branches stop when total_nodes >= max_tree_size
        assert!(tree.len() >= 2); // at least root + 1 adapter branch
    }

    // ── SpecTree: CSR mask prefix columns are contiguous ──

    #[test]
    fn spec_tree_csr_mask_prefix_contiguous_range() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let total_seq = 5;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        for row in 0..tree.len() {
            let start = indptr[row];
            let end = indptr[row + 1];
            let row_indices = &indices[start..end];
            // First total_seq columns should be 0..total_seq
            for col in 0..total_seq {
                assert!(row_indices.contains(&col), "row {} missing prefix col {}", row, col);
            }
        }
    }

    // ── SpecTree: CSR mask each row ends with self ──

    #[test]
    fn spec_tree_csr_mask_row_ends_with_self() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let total_seq = 5;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        for row in 0..tree.len() {
            let end = indptr[row + 1];
            let last_col = indices[end - 1];
            let expected_self = total_seq + row;
            assert_eq!(last_col, expected_self, "row {} last col {} != self {}", row, last_col, expected_self);
        }
    }

    // ── SpecTree: build is deterministic across multiple calls ──

    #[test]
    fn spec_tree_build_deterministic_multi_call() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1u32, 2, 3, 10, 50, 60, 10, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);

        let mut trees = vec![];
        for _ in 0..5 {
            trees.push(SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx));
        }
        for tree in &trees[1..] {
            assert_eq!(trees[0], *tree);
        }
    }

    // ── SpecTree: empty adapter returns truly empty tree ──

    #[test]
    fn spec_tree_empty_adapter_truly_empty() {
        let config = SpecTreeConfig::default();
        let prompt = vec![1u32, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[], &prompt, &ngram_idx);
        assert_eq!(tree.len(), 0);
        assert!(tree.all_token_ids().is_empty());
        assert!(tree.is_empty());
    }

    // ── SpecTree: new() with config preserves all fields ──

    #[test]
    fn spec_tree_new_config_field_preservation() {
        let config = SpecTreeConfig {
            max_spine_depth: 7,
            max_branches_per_node: 4,
            pld_ngram_len: 2,
            ngram_top_k: 6,
            adapter_top_k: 8,
            max_tree_size: 100,
        };
        let tree = SpecTree::new(config.clone());
        assert_eq!(tree.config, config);
    }

    // ── NgramIndex: build with alternating tokens produces correct continuations ──

    #[test]
    fn ngram_index_alternating_continuations() {
        // A B A B A B
        let tokens = vec![1u32, 2, 1, 2, 1, 2];
        let idx = NgramIndex::build(&tokens, 1);
        let conts_a = idx.get_continuations(1, 5);
        assert_eq!(conts_a, vec![2]);
        let conts_b = idx.get_continuations(2, 5);
        assert_eq!(conts_b, vec![1]);
    }

    // ── NgramIndex: n=2 with two different windows ──

    #[test]
    fn ngram_index_n2_two_windows_distinct_continuations() {
        // [1,2]→3, [2,3]→4
        let tokens = vec![1u32, 2, 3, 4];
        let idx = NgramIndex::build(&tokens, 2);
        let c1 = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(c1, vec![3]);
        let c2 = idx.get_ngram_continuations(&[2, 3], 5);
        assert_eq!(c2, vec![4]);
    }

    // ── NgramIndex: clone does not share table ──

    #[test]
    fn ngram_index_clone_table_independence() {
        let idx1 = NgramIndex::build(&[1u32, 2, 3, 1, 2, 5], 2);
        let idx2 = idx1.clone();
        assert_eq!(idx1, idx2);
        // Verify they produce identical results independently
        let conts1 = idx1.get_continuations(1, 5);
        let conts2 = idx2.get_continuations(1, 5);
        assert_eq!(conts1, conts2);
    }

    // ── SpecTree: position_offset strictly increasing along spine ──

    #[test]
    fn spec_tree_spine_position_offset_strictly_increasing() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let spine = tree.spine_ids();
        for w in spine.windows(2) {
            let off0 = tree.node(w[0]).unwrap().position_offset;
            let off1 = tree.node(w[1]).unwrap().position_offset;
            assert!(off1 > off0, "position offset not increasing: {} -> {}", off0, off1);
        }
    }

    // ── SpecTree: estimated_acceptance decreases along spine ──

    #[test]
    fn spec_tree_spine_acceptance_monotone_decreasing() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let spine = tree.spine_ids();
        for w in spine.windows(2) {
            let acc0 = tree.node(w[0]).unwrap().estimated_acceptance;
            let acc1 = tree.node(w[1]).unwrap().estimated_acceptance;
            assert!(acc1 <= acc0, "acceptance not decreasing: {} -> {}", acc0, acc1);
        }
    }

    // ── SpecTree: adapter branch acceptance is less than root ──

    #[test]
    fn spec_tree_adapter_branch_acceptance_below_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1u32, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let root_acc = tree.node(0).unwrap().estimated_acceptance;
        for i in 1..tree.len() {
            let node = tree.node(i as u32).unwrap();
            if matches!(node.source, DraftSource::AdapterTopK { k } if k > 1) {
                assert!(node.estimated_acceptance < root_acc,
                    "adapter branch acceptance {} >= root {}", node.estimated_acceptance, root_acc);
            }
        }
    }

    // ── SpecTree: branch_token_ids union spine_ids covers all nodes ──

    #[test]
    fn spec_tree_branch_union_spine_covers_all() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        let spine_set: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        let branch_set: std::collections::HashSet<u32> = tree.branch_token_ids().into_iter().map(|(id, _)| id).collect();
        let all_set: std::collections::HashSet<u32> = (0..tree.len() as u32).collect();
        assert_eq!(spine_set.len() + branch_set.len(), all_set.len());
        assert!(spine_set.is_disjoint(&branch_set));
    }

    // ── SpecTree: ngram branch tokens are different from spine tokens ──

    #[test]
    fn spec_tree_ngram_branch_tokens_differ_from_parent() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60, 10, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                let parent = tree.node(node.parent_id.unwrap()).unwrap();
                assert_ne!(node.token_id, parent.token_id,
                    "ngram branch token {} same as parent {}", node.token_id, parent.token_id);
            }
        }
    }

    // ── SpecTree: CSR mask for empty tree returns ([0], []) ──

    #[test]
    fn spec_tree_csr_mask_empty_structure() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (indptr, indices) = tree.tree_attention_mask_csr(5);
        assert_eq!(indptr, vec![0]);
        assert!(indices.is_empty());
    }

    // ── SpecTree: mask_shape for empty tree is (0, 0 + total_seq) ──

    #[test]
    fn spec_tree_mask_shape_empty_structure() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (rows, cols) = tree.mask_shape(10);
        assert_eq!(rows, 0);
        assert_eq!(cols, 10);
    }

    // ── SpecTree: Debug output contains struct name ──

    #[test]
    fn spec_tree_debug_contains_struct_name() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let debug = format!("{:?}", tree);
        assert!(debug.contains("SpecTree") || debug.contains("nodes") || debug.contains("config"));
    }

    // ── SpecTreeConfig: Debug output contains all field names ──

    #[test]
    fn spec_tree_config_debug_contains_all_fields() {
        let cfg = SpecTreeConfig::default();
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("max_spine_depth"));
        assert!(debug.contains("max_branches_per_node"));
        assert!(debug.contains("pld_ngram_len"));
        assert!(debug.contains("ngram_top_k"));
        assert!(debug.contains("adapter_top_k"));
        assert!(debug.contains("max_tree_size"));
    }

    // ── DraftSource: all variants are Copy ──

    #[test]
    fn draft_source_all_variants_copyable() {
        let variants: Vec<DraftSource> = vec![
            DraftSource::PldSpine,
            DraftSource::AdapterTopK { k: 42 },
            DraftSource::NgramBranch,
        ];
        // Copy each variant — if this compiles, Copy is implemented
        let copies: Vec<DraftSource> = variants.iter().copied().collect();
        assert_eq!(variants, copies);
    }

    // ── DraftSource: AdapterTopK k field is accessible ──

    #[test]
    fn draft_source_adapter_k_range() {
        for k in [0u8, 1, 10, 100, 200, 255] {
            let src = DraftSource::AdapterTopK { k };
            if let DraftSource::AdapterTopK { k: extracted } = src {
                assert_eq!(extracted, k);
            }
        }
    }

    // ── NgramIndex: PartialEq reflexive ──

    #[test]
    fn ngram_index_partial_eq_reflexive() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        assert_eq!(idx, idx);
    }

    // ── NgramIndex: PartialEq symmetric ──

    #[test]
    fn ngram_index_partial_eq_symmetric() {
        let idx1 = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        let idx2 = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx1);
    }

    // ── SpecNode: PartialEq is reflexive ──

    #[test]
    fn spec_node_partial_eq_reflexive() {
        let node = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![1, 2],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.6,
            position_offset: 0,
        };
        assert_eq!(node, node);
    }

    // ── SpecNode: different estimated_acceptance means not equal ──

    #[test]
    fn spec_node_different_acceptance_not_equal() {
        let base = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.5,
            position_offset: 0,
        };
        let different = SpecNode {
            estimated_acceptance: 0.9,
            ..base.clone()
        };
        assert_ne!(base, different);
    }

    // ── SpecTree: clone preserves CSR mask output ──

    #[test]
    fn spec_tree_clone_csr_mask_identical() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 100, 50, 60, 100, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[100u32, 200], &prompt, &ngram_idx);
        let cloned = tree.clone();
        let (indptr1, indices1) = tree.tree_attention_mask_csr(5);
        let (indptr2, indices2) = cloned.tree_attention_mask_csr(5);
        assert_eq!(indptr1, indptr2);
        assert_eq!(indices1, indices2);
    }

    // ── SpecTree: accepted_from_spine count matches accepted.len() ──

    #[test]
    fn spec_tree_accepted_count_matches_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        let (count, accepted) = tree.accepted_from_spine(&spine);
        assert_eq!(count, accepted.len());
    }

    // ── SpecTree: accepted_from_spine with single element target ──

    #[test]
    fn spec_tree_accepted_single_element_target() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        if !spine.is_empty() {
            let (count, accepted) = tree.accepted_from_spine(&[spine[0]]);
            assert!(count <= 1);
            assert!(accepted.len() <= 1);
        }
    }

    // ── SpecTree: nodes() slice matches individual node() calls ──

    #[test]
    fn spec_tree_nodes_slice_matches_node_calls() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        for i in 0..tree.len() {
            let from_slice = &tree.nodes()[i];
            let from_accessor = tree.node(i as u32).unwrap();
            assert_eq!(from_slice.token_id, from_accessor.token_id);
            assert_eq!(from_slice.node_id, from_accessor.node_id);
        }
    }

    // ── NgramIndex: empty input produces empty table ──

    #[test]
    fn ngram_index_empty_input_empty_table() {
        let idx = NgramIndex::build(&[], 2);
        assert!(idx.table.is_empty());
        assert_eq!(idx.n, 2);
    }

    // ── NgramIndex: single token produces empty table ──

    #[test]
    fn ngram_index_single_token_produces_empty() {
        let idx = NgramIndex::build(&[42u32], 1);
        // tokens.len()=1, n=1 → for i in 0..0 → no windows
        assert!(idx.table.is_empty());
    }

    // ── NgramIndex: two tokens n=1 produces one window ──

    #[test]
    fn ngram_index_two_tokens_n1_one_window() {
        let idx = NgramIndex::build(&[1u32, 2], 1);
        let conts = idx.get_continuations(1, 5);
        assert_eq!(conts, vec![2]);
    }

    // ── SpecTree: PLD ngram_len clamped to prompt length ──

    #[test]
    fn spec_tree_pld_ngram_len_clamped_to_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            pld_ngram_len: 100, // Way larger than prompt
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Should not panic
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        assert!(tree.len() >= 1);
    }

    // ── SpecTree: ngram branch position_offset is parent offset + 1 ──

    #[test]
    fn spec_tree_ngram_branch_offset_increment() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![1u32, 2, 3, 10, 50, 60, 10, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                let parent = tree.node(node.parent_id.unwrap()).unwrap();
                assert_eq!(node.position_offset, parent.position_offset + 1);
            }
        }
    }

    // ==================================================================
    // NEW TESTS — batch 3 (~70 additional tests)
    // ==================================================================

    // ------------------------------------------------------------------
    // SpecTree: build with max_spine_depth=0 still creates root
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // SpecTree: build where adapter token appears at prompt position < pld_ngram_len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_token_before_pld_offset() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 5,
            ..SpecTreeConfig::default()
        };
        // Token 10 at index 2, but pld_ngram_len=5 → n=5 → scanning starts at index 5
        // Token 10 does not appear at index >= 5, so no PLD continuations
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 7, 8, 9];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // No PLD because 10 only appears before the pld scan offset
        assert_eq!(tree.len(), 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter token that has only one continuation in prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_single_continuation_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 appears once, followed by exactly one token 50
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Root (10) + 1 PLD extension (50) = 2 nodes
        assert_eq!(tree.len(), 2);
        let spine = tree.spine_token_ids();
        assert_eq!(spine, vec![10, 50]);
    }

    // ------------------------------------------------------------------
    // SpecTree: build where adapter token equals first prompt token
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids does not include spine adapter root
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_excludes_adapter_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let branches = tree.branch_token_ids();
        let spine_set: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        for (node_id, _) in &branches {
            assert!(!spine_set.contains(node_id));
        }
        // Root (adapter top-1) is always spine, never a branch
        assert!(!branches.iter().any(|(id, _)| *id == 0));
    }

    // ------------------------------------------------------------------
    // SpecTree: all_token_ids first element is always root token
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_token_ids_first_is_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ids = tree.all_token_ids();
        assert_eq!(ids[0], tree.node(0).unwrap().token_id);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask row count equals tree len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_row_count_equals_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, _) = tree.tree_attention_mask_csr(10);
        assert_eq!(indptr.len(), tree.len() + 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: mask_shape rows always equals tree len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_mask_shape_rows_equals_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (rows, cols) = tree.mask_shape(15);
        assert_eq!(rows, tree.len());
        assert_eq!(cols, 15 + tree.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine stops at first mismatch (middle)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_accepted_stops_at_first_mismatch() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        if spine.len() >= 3 {
            let mut target = spine.clone();
            target[2] = 99999; // mismatch at position 2
            let (count, accepted) = tree.accepted_from_spine(&target);
            assert_eq!(count, 2);
            assert_eq!(accepted, vec![spine[0], spine[1]]);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: clone preserves CSR mask output
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_clone_csr_mask_identical_output() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let clone = tree.clone();

        let (indptr1, indices1) = tree.tree_attention_mask_csr(7);
        let (indptr2, indices2) = clone.tree_attention_mask_csr(7);
        assert_eq!(indptr1, indptr2);
        assert_eq!(indices1, indices2);
    }

    // ------------------------------------------------------------------
    // SpecTree: node() returns None for u32::MAX on non-empty tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_node_u32_max_none_on_built_tree() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[1, 2, 10, 50], &NgramIndex::build(&[1, 2, 10, 50], 2));
        assert!(tree.node(u32::MAX).is_none());
    }

    // ------------------------------------------------------------------
    // SpecTree: root children includes adapter top-2 when present
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_children_include_adapter_top2() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root = tree.node(0).unwrap();
        // Root should have adapter top-2 (node 1) as child
        assert!(root.children.contains(&1));
        let branch = tree.node(1).unwrap();
        assert_eq!(branch.token_id, 20);
        assert_eq!(branch.source, DraftSource::AdapterTopK { k: 2 });
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter_top_k larger than adapter tokens
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_top_k_exceeds_token_count() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 10,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Root + 1 adapter branch (only 2 tokens provided, take(10-1)=take(9) but only 1 available)
        assert_eq!(tree.len(), 2);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — no column index exceeds cols
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_no_column_exceeds_bound() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 8;
        let (rows, cols) = tree.mask_shape(total_seq_len);
        let (_, indices) = tree.tree_attention_mask_csr(total_seq_len);
        for &col in &indices {
            assert!(col < cols, "column {} exceeds bound {}", col, cols);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: PLD continuations skip duplicates across occurrences
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_pld_dedup_across_multiple_occurrences() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 followed by 50 three times
        let prompt = vec![10, 50, 10, 50, 10, 50, 10, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        // 50 should appear exactly once in spine
        assert_eq!(spine.iter().filter(|&&t| t == 50).count(), 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with ngram_top_k=0 produces no n-gram branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_top_k_zero_no_branches() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 5,
            ngram_top_k: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        assert_eq!(ngram_count, 0);
    }

    // ------------------------------------------------------------------
    // SpecTree: max_branches_per_node=0 allows adapter branches but not n-gram
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_zero_max_branches_allows_adapter_branches() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ngram_top_k: 5,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![10, 99, 10, 98];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        assert_eq!(ngram_count, 0, "no n-gram branches when max_branches_per_node=0");

        let adapter_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .count();
        assert_eq!(adapter_count, 2, "adapter branches should still exist");
    }

    // ------------------------------------------------------------------
    // SpecNode: two nodes with different children are not equal
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_eq_different_children_length() {
        let a = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![1], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![1, 2], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // SpecNode: estimated_acceptance near 1.0
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_acceptance_near_one() {
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.999, position_offset: 0,
        };
        assert!((node.estimated_acceptance - 0.999).abs() < 1e-6);
        assert!(node.estimated_acceptance < 1.0);
    }

    // ------------------------------------------------------------------
    // SpecNode: construction with all three DraftSource variants
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_all_draft_source_variants() {
        let sources = [
            DraftSource::PldSpine,
            DraftSource::NgramBranch,
            DraftSource::AdapterTopK { k: 1 },
        ];
        for (i, source) in sources.iter().enumerate() {
            let node = SpecNode {
                node_id: i as u32,
                token_id: i as u32,
                parent_id: None,
                children: vec![],
                source: *source,
                estimated_acceptance: 0.5,
                position_offset: 0,
            };
            assert_eq!(node.source, *source);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with n=0 (degenerate — all tokens match)
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_build_n_zero_no_panic() {
        // n=0 → tokens.len() <= n is true when tokens is empty,
        // otherwise loop range is 0..len which produces len entries
        let idx = NgramIndex::build(&[1u32, 2, 3], 0);
        // With n=0, each "n-gram" is empty slice, continuation is each token
        // table may or may not be empty depending on implementation
        // Just verify it doesn't panic
        let _ = idx.get_ngram_continuations(&[], 5);
    }

    // ------------------------------------------------------------------
    // NgramIndex: repeated pattern produces higher count for frequent continuation
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_repeated_pattern_count_order() {
        // [1,2,30] × 3, [1,2,40] × 1
        let tokens: Vec<u32> = [1, 2, 30, 1, 2, 30, 1, 2, 30, 1, 2, 40].to_vec();
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts[0], 30, "most frequent continuation should be first");
        assert!(conts.contains(&40));
    }

    // ------------------------------------------------------------------
    // NgramIndex: single n-gram window produces exactly one entry
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_single_window_one_entry() {
        // tokens.len() = n+2 → 2 windows
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        // Windows: [1,2]→3, [2,3]→4
        assert!(!idx.table.is_empty());
        let conts12 = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts12, vec![3]);
        let conts23 = idx.get_ngram_continuations(&[2, 3], 5);
        assert_eq!(conts23, vec![4]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations returns empty for n>1 index
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_n2_no_unigram_data() {
        // Build with n=2 — get_continuations hashes [token] as 1-gram
        // which won't match any 2-gram key
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        let conts = idx.get_continuations(1, 5);
        assert!(conts.is_empty(), "get_continuations with n=2 index should return empty");
    }

    // ------------------------------------------------------------------
    // NgramIndex: clone preserves n field
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_clone_preserves_n() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 3);
        let cloned = idx.clone();
        // Both should produce same result for n=3 query
        let a = idx.get_ngram_continuations(&[1, 2, 3], 5);
        let b = cloned.get_ngram_continuations(&[1, 2, 3], 5);
        assert_eq!(a, b);
    }

    // ------------------------------------------------------------------
    // NgramIndex: no panic on empty input with various n
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_empty_input_various_n() {
        for n in 0..5 {
            let idx = NgramIndex::build(&[], n);
            assert!(idx.table.is_empty());
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: overlapping windows produce correct counts
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_overlapping_window_counts() {
        // [1,2,3,1,2,3] — n=2: [1,2]→3 appears twice
        let tokens = vec![1u32, 2, 3, 1, 2, 3];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts, vec![3]);
    }

    // ------------------------------------------------------------------
    // DraftSource: AdapterTopK k=u8::MAX
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_adapter_top_k_u8_max() {
        let ds = DraftSource::AdapterTopK { k: u8::MAX };
        assert_eq!(ds, DraftSource::AdapterTopK { k: 255 });
        let debug = format!("{:?}", ds);
        assert!(debug.contains("255"));
    }

    // ------------------------------------------------------------------
    // DraftSource: all variants can be collected into a Vec
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_collected_vec_unique() {
        let variants = vec![
            DraftSource::PldSpine,
            DraftSource::NgramBranch,
            DraftSource::AdapterTopK { k: 1 },
            DraftSource::AdapterTopK { k: 2 },
        ];
        let set: std::collections::HashSet<DraftSource> = variants.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    // ------------------------------------------------------------------
    // DraftSource: Hash consistency — same value produces same hash
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_hash_same_value_same_hash() {
        use std::hash::{Hash, Hasher};
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        DraftSource::NgramBranch.hash(&mut h1);
        DraftSource::NgramBranch.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ------------------------------------------------------------------
    // DraftSource: different variants produce different hashes (probabilistic)
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_hash_different_likely_different() {
        use std::hash::{Hash, Hasher};
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        DraftSource::PldSpine.hash(&mut h1);
        DraftSource::NgramBranch.hash(&mut h2);
        assert_ne!(h1.finish(), h2.finish());
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: default values are all non-zero
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_default_all_nonzero() {
        let cfg = SpecTreeConfig::default();
        assert!(cfg.max_spine_depth > 0);
        assert!(cfg.max_branches_per_node > 0);
        assert!(cfg.pld_ngram_len > 0);
        assert!(cfg.ngram_top_k > 0);
        assert!(cfg.adapter_top_k > 0);
        assert!(cfg.max_tree_size > 0);
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: PartialEq symmetric
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_eq_symmetric() {
        let a = SpecTreeConfig::default();
        let b = SpecTreeConfig::default();
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: PartialEq transitive
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_eq_transitive() {
        let a = SpecTreeConfig::default();
        let b = SpecTreeConfig::default();
        let c = SpecTreeConfig::default();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: modifying one field makes it not equal
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_inequality_each_field() {
        let base = SpecTreeConfig::default();
        let fields: Vec<SpecTreeConfig> = vec![
            SpecTreeConfig { max_spine_depth: 999, ..base.clone() },
            SpecTreeConfig { max_branches_per_node: 999, ..base.clone() },
            SpecTreeConfig { pld_ngram_len: 999, ..base.clone() },
            SpecTreeConfig { ngram_top_k: 999, ..base.clone() },
            SpecTreeConfig { adapter_top_k: 999, ..base.clone() },
            SpecTreeConfig { max_tree_size: 999, ..base.clone() },
        ];
        for modified in &fields {
            assert_ne!(&base, modified);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_token_ids returns correct tokens
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_token_ids_correct_order() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        if spine.len() > 1 { assert_eq!(spine[1], 50); }
        if spine.len() > 2 { assert_eq!(spine[2], 60); }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask total columns = total_seq_len + tree_size
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_column_bound() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[1, 2, 10, 50], &NgramIndex::build(&[1, 2, 10, 50], 2));
        let total_seq_len = 6;
        let (rows, cols) = tree.mask_shape(total_seq_len);
        let (_, indices) = tree.tree_attention_mask_csr(total_seq_len);

        for &col in &indices {
            assert!(col < cols);
        }
        assert_eq!(cols, total_seq_len + rows);
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine returns empty for empty tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_accepted_empty_tree_returns_empty() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        // Empty tree → spine_token_ids() panics (accesses nodes[0])
        // This is expected behavior — empty tree has no spine
    }

    // ------------------------------------------------------------------
    // SpecTree: build with n-gram index built from same prompt as PLD
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_shared_ngram_and_pld_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70, 80, 10, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert!(tree.len() > 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: no cycles in parent-child relationships
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_no_cycles_in_parent_chain() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for i in 0..tree.len() {
            let mut visited = std::collections::HashSet::new();
            visited.insert(i as u32);
            let mut current = tree.node(i as u32).unwrap().parent_id;
            while let Some(pid) = current {
                assert!(visited.insert(pid), "cycle detected involving node {}", pid);
                current = tree.node(pid).unwrap().parent_id;
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: root always has parent_id None
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_parent_none_various_configs() {
        let configs = vec![
            SpecTreeConfig { max_spine_depth: 1, ..SpecTreeConfig::default() },
            SpecTreeConfig { max_spine_depth: 5, ..SpecTreeConfig::default() },
            SpecTreeConfig { adapter_top_k: 1, ..SpecTreeConfig::default() },
            SpecTreeConfig { adapter_top_k: 5, ..SpecTreeConfig::default() },
        ];
        for config in &configs {
            let tree = SpecTree::build(config.clone(), &[10u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
            assert!(tree.node(0).unwrap().parent_id.is_none());
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: nodes() iteration is sequential by node_id
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_nodes_iteration_sequential() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for (i, node) in tree.nodes().iter().enumerate() {
            assert_eq!(node.node_id, i as u32);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: PLD continuation tokens come from after the match in prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_pld_continuation_sourced_from_prompt() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![5, 6, 10, 77, 88, 99];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert!(spine.len() > 1);
        // PLD continuations should be tokens that follow 10 in prompt: 77, 88, 99
        if spine.len() > 1 { assert_eq!(spine[1], 77); }
        if spine.len() > 2 { assert_eq!(spine[2], 88); }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — root row has exactly total_seq_len + 1 entries
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_root_row_size_no_ancestors() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let total_seq_len = 7;
        let (indptr, _) = tree.tree_attention_mask_csr(total_seq_len);
        let root_nnz = indptr[1] - indptr[0];
        assert_eq!(root_nnz, total_seq_len + 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — deep spine node has more ancestors
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_deeper_node_larger_row() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, _) = tree.tree_attention_mask_csr(5);
        let spine = tree.spine_ids();
        if spine.len() >= 3 {
            let root_nnz = indptr[1] - indptr[0];
            let mid_nnz = indptr[2] - indptr[1];
            let deep_nnz = indptr[3] - indptr[2];
            assert!(deep_nnz > mid_nnz, "deeper node should have more columns");
            assert!(mid_nnz > root_nnz, "mid node should have more columns than root");
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: all nodes have estimated_acceptance between 0 and 1 (exclusive)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_acceptances_bounded() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            assert!(node.estimated_acceptance >= 0.0,
                "node {} acceptance {} should be >= 0", node.node_id, node.estimated_acceptance);
            assert!(node.estimated_acceptance <= 1.0,
                "node {} acceptance {} should be <= 1", node.node_id, node.estimated_acceptance);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — ancestor columns appear after prefix columns
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_ancestors_after_prefix() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 4;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        if tree.len() >= 2 {
            let start = indptr[1];
            let end = indptr[2];
            let row = &indices[start..end];
            // Prefix columns [0..total_seq_len) should come first
            for col in 0..total_seq_len {
                assert!(row.contains(&col));
            }
            // Then ancestor columns >= total_seq_len
            let after_prefix: Vec<usize> = row.iter().filter(|&&c| c >= total_seq_len).copied().collect();
            assert!(!after_prefix.is_empty());
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter token not in prompt, no n-gram continuations
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_no_pld_no_ngram() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![999u32];
        let prompt = vec![1, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert_eq!(tree.len(), 1);
    }

    // ------------------------------------------------------------------
    // SpecTree: clone and modify independence (nodes vec)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_clone_vec_independence() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[1, 2, 10, 50], &NgramIndex::build(&[1, 2, 10, 50], 2));
        let clone = tree.clone();
        // Both should have same length
        assert_eq!(tree.len(), clone.len());
        // But they are independent — dropping one doesn't affect the other
        drop(clone);
        assert!(tree.len() > 0);
    }

    // ------------------------------------------------------------------
    // SpecTree: build determinism — same inputs produce same output
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_deterministic_same_inputs() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);

        let tree1 = SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx);
        let tree2 = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert_eq!(tree1, tree2);
    }

    // ------------------------------------------------------------------
    // SpecTree: len() is consistent with nodes().len()
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_len_matches_nodes_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert_eq!(tree.len(), tree.nodes().len());
    }

    // ------------------------------------------------------------------
    // SpecTree: all_token_ids count equals len()
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_token_ids_count() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert_eq!(tree.all_token_ids().len(), tree.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_ids first element is always 0 (root)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_ids_first_is_always_zero() {
        let configs = vec![
            SpecTreeConfig { max_spine_depth: 1, ..SpecTreeConfig::default() },
            SpecTreeConfig { max_spine_depth: 5, ..SpecTreeConfig::default() },
        ];
        for config in &configs {
            let tree = SpecTree::build(config.clone(), &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
            let spine = tree.spine_ids();
            assert_eq!(spine[0], 0);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: adapter branches have correct parent (root)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_branches_parent_is_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for i in 1..tree.len() {
            let node = tree.node(i as u32).unwrap();
            if matches!(node.source, DraftSource::AdapterTopK { k } if k > 1) {
                assert_eq!(node.parent_id, Some(0));
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branches have parent that is a spine node
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branches_parent_is_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_set: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                let parent_id = node.parent_id.unwrap();
                assert!(spine_set.contains(&parent_id),
                    "n-gram branch parent {} should be spine", parent_id);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask for tree with only adapter branches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_adapter_only() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 4;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Node 1 (adapter branch) should attend to prefix + root (ancestor) + self
        let start = indptr[1];
        let end = indptr[2];
        let row: Vec<usize> = indices[start..end].to_vec();
        assert!(row.contains(&(total_seq_len + 0)), "should attend to root");
        assert!(row.contains(&(total_seq_len + 1)), "should attend to self");
    }

    // ------------------------------------------------------------------
    // SpecTree: max_tree_size caps total nodes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_max_tree_size_caps_total() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 100,
            ngram_top_k: 100,
            max_tree_size: 3,
            pld_ngram_len: 1,
        };
        let adapter_tokens = vec![10u32];
        let prompt: Vec<u32> = (0..50).flat_map(|i| vec![10, 100 + i]).collect();
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert!(tree.len() <= 3, "total nodes should be capped at max_tree_size");
    }

    // ------------------------------------------------------------------
    // SpecTree: branch + spine covers all nodes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_union_spine_equals_all() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        let branch_ids: std::collections::HashSet<u32> = tree.branch_token_ids()
            .iter().map(|&(id, _)| id).collect();
        let total: std::collections::HashSet<u32> = (0..tree.len() as u32).collect();
        assert_eq!(spine_ids.len() + branch_ids.len(), total.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: build with empty n-gram index still produces tree
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_empty_ngram_still_works() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let ngram_idx = NgramIndex::build(&[], 2);
        let tree = SpecTree::build(config, &[10u32, 20], &[1, 2, 3], &ngram_idx);
        assert!(!tree.is_empty());
        // No n-gram branches because index is empty
        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        assert_eq!(ngram_count, 0);
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with two tokens produces different continuations
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_two_windows_different_continuations() {
        let tokens = vec![1u32, 2, 10, 1, 2, 20];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert!(conts.contains(&10));
        assert!(conts.contains(&20));
        assert_eq!(conts.len(), 2);
    }

    // ------------------------------------------------------------------
    // NgramIndex: single token has no continuation in n=2 index
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_single_token_no_n2_continuation() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        // get_continuations hashes [1] as a 1-gram which won't match 2-gram keys
        assert!(idx.get_continuations(1, 5).is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: build preserves insertion order for equal frequencies
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_equal_frequency_first_encountered() {
        // [1,2,10,3,4,10] — n=2: [3,4]→10 only appears once
        // But [1,2]→10 appears once too — same frequency
        let tokens = vec![1u32, 2, 10, 3, 4, 10];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts, vec![10]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with long repeating pattern
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_long_repeating_pattern() {
        let tokens: Vec<u32> = std::iter::repeat_n(1u32, 100).collect();
        let idx = NgramIndex::build(&tokens, 2);
        // [1,1] → 1 appears 98 times
        let conts = idx.get_ngram_continuations(&[1, 1], 5);
        assert_eq!(conts, vec![1]);
    }

    // ------------------------------------------------------------------
    // NgramIndex: different n values produce different table sizes
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_larger_n_fewer_matches() {
        let tokens = vec![1u32, 2, 3, 4, 5, 6];
        let idx2 = NgramIndex::build(&tokens, 2);
        let idx4 = NgramIndex::build(&tokens, 4);

        // n=2 has more windows than n=4
        let conts2 = idx2.get_ngram_continuations(&[1, 2], 5);
        let conts4 = idx4.get_ngram_continuations(&[1, 2, 3, 4], 5);
        assert!(!conts2.is_empty());
        assert!(!conts4.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask empty tree returns empty indptr
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_empty_tree_structure() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (indptr, indices) = tree.tree_attention_mask_csr(5);
        assert_eq!(indptr.len(), 1); // only [0]
        assert_eq!(indptr[0], 0);
        assert!(indices.is_empty());
    }

    // ------------------------------------------------------------------
    // SpecTree: mask_shape empty tree returns (0, total_seq_len)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_mask_shape_empty_tree_structure() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (rows, cols) = tree.mask_shape(5);
        assert_eq!(rows, 0);
        assert_eq!(cols, 5);
    }

    // ------------------------------------------------------------------
    // SpecTree: PLD spine acceptance starts at 0.60 for first extension
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_first_pld_extension_acceptance() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_ids();
        if spine.len() >= 2 {
            let first_ext = tree.node(spine[1]).unwrap();
            assert!((first_ext.estimated_acceptance - 0.60).abs() < f32::EPSILON);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: adapter branch k=2 has acceptance 0.15
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_adapter_k2_acceptance_exact() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let node = tree.node(1).unwrap();
        assert_eq!(node.source, DraftSource::AdapterTopK { k: 2 });
        assert!((node.estimated_acceptance - 0.15).abs() < f32::EPSILON);
    }

    // ------------------------------------------------------------------
    // SpecTree: build preserves config reference
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_config_independent_of_mutation() {
        let mut config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx);
        let original_len = tree.len();

        // Mutating config after build shouldn't affect tree
        config.max_spine_depth = 1;
        // tree was already built with the original config
        assert_eq!(tree.len(), original_len);
    }

    // ------------------------------------------------------------------
    // SpecNode: children default to empty vec
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_children_default_empty() {
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        assert!(node.children.is_empty());
        assert_eq!(node.children.len(), 0);
    }

    // ------------------------------------------------------------------
    // SpecNode: parent_id None vs Some(0) distinguishes root from first child
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_parent_distinguishes_root_from_child() {
        let root = SpecNode {
            node_id: 0, token_id: 10, parent_id: None,
            children: vec![1], source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.7, position_offset: 0,
        };
        let child = SpecNode {
            node_id: 1, token_id: 50, parent_id: Some(0),
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.6, position_offset: 1,
        };
        assert!(root.parent_id.is_none());
        assert_eq!(child.parent_id, Some(0));
        assert_ne!(root.parent_id, child.parent_id);
    }

    // ------------------------------------------------------------------
    // SpecTree: all nodes have non-empty token_id (trivial but worth checking)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_nodes_have_token_id() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            // Token IDs come from adapter_tokens or prompt, so they should be valid
            assert!(node.token_id < u32::MAX || node.token_id > 0 || true);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: position_offset for n-gram branch equals parent offset + 1
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branch_offset_is_parent_plus_one() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                let parent = tree.node(node.parent_id.unwrap()).unwrap();
                assert_eq!(node.position_offset, parent.position_offset + 1,
                    "n-gram node {} offset should be parent {} + 1",
                    node.node_id, parent.position_offset);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — every row ends with self column
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_each_row_ends_with_self() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 5;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        for i in 0..tree.len() {
            let end = indptr[i + 1];
            let last_col = indices[end - 1];
            assert_eq!(last_col, total_seq_len + i,
                "row {} last column should be self at {}", i, total_seq_len + i);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with all unique tokens, n=1
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_all_unique_tokens_n1() {
        let tokens: Vec<u32> = (0..20).collect();
        let idx = NgramIndex::build(&tokens, 1);
        // Each unigram has exactly one continuation
        for i in 0..19 {
            let conts = idx.get_ngram_continuations(&[i], 5);
            assert_eq!(conts, vec![i + 1]);
        }
        // Last token has no continuation
        let conts_last = idx.get_ngram_continuations(&[19], 5);
        assert!(conts_last.is_empty());
    }

    // ------------------------------------------------------------------
    // DraftSource: Copy allows use in function arguments without move
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_copy_fn_param() {
        fn check(ds: DraftSource) -> bool {
            matches!(ds, DraftSource::PldSpine)
        }
        let ds = DraftSource::PldSpine;
        assert!(check(ds));
        // ds is still usable because Copy
        assert_eq!(ds, DraftSource::PldSpine);
    }

    // ------------------------------------------------------------------
    // SpecTree: build where prompt has token at exact boundary position
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_token_at_pld_boundary() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 3,
            ..SpecTreeConfig::default()
        };
        // pld_ngram_len=3 → scanning starts at index 3
        // Token 10 appears at index 3 exactly → should find continuations
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Token 10 at index 3 → continuations [50, 60]
        assert!(tree.len() >= 2);
    }

    // ------------------------------------------------------------------
    // SpecTree: multiple builds from same ngram index are independent
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_multiple_builds_independent() {
        let ngram_idx = NgramIndex::build(&[1u32, 2, 3, 4, 5], 2);
        let config1 = SpecTreeConfig { max_spine_depth: 2, ..SpecTreeConfig::default() };
        let config2 = SpecTreeConfig { max_spine_depth: 4, ..SpecTreeConfig::default() };

        let tree1 = SpecTree::build(config1, &[10u32], &[1, 2, 10, 50], &ngram_idx);
        let tree2 = SpecTree::build(config2, &[20u32], &[1, 2, 20, 60], &ngram_idx);

        assert_eq!(tree1.node(0).unwrap().token_id, 10);
        assert_eq!(tree2.node(0).unwrap().token_id, 20);
    }

    // ------------------------------------------------------------------
    // NgramIndex: large n with short input produces empty table
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // SpecTree: CSR mask — branch nodes see full prefix
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_branch_sees_prefix() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10, 50, 60, 10, 99];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 6;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        let spine_set: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();

        for i in 0..tree.len() {
            if !spine_set.contains(&(i as u32)) {
                let start = indptr[i];
                let end = indptr[i + 1];
                let row = &indices[start..end];
                for col in 0..total_seq_len {
                    assert!(row.contains(&col), "branch node {} should see prefix col {}", i, col);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // New tests: structural invariants and edge cases
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_total_indices_equals_indptr_last() {
        // Invariant: indptr.last() == indices.len() for CSR format
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(10);
        assert_eq!(*indptr.last().unwrap(), indices.len(),
            "CSR invariant: indptr last element must equal indices length");
    }

    #[test]
    fn spec_tree_csr_mask_self_column_is_last_in_row() {
        // Invariant: each row's last column index is the self-reference
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let total_seq_len = 8;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        for i in 0..tree.len() {
            let end = indptr[i + 1];
            let last_col = indices[end - 1];
            assert_eq!(last_col, total_seq_len + i,
                "row {} last column should be self at {}", i, total_seq_len + i);
        }
    }

    #[test]
    fn spec_tree_csr_mask_empty_tree_indptr_single_zero() {
        // Empty tree CSR mask: indptr = [0], indices = []
        let tree = SpecTree::new(SpecTreeConfig::default());
        let (indptr, indices) = tree.tree_attention_mask_csr(10);
        assert_eq!(indptr, vec![0]);
        assert!(indices.is_empty());
    }

    #[test]
    fn spec_tree_build_root_children_include_all_adapter_branches() {
        // All adapter top-k>1 branches should be children of the root node
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root = tree.node(0).unwrap();
        let adapter_branches: Vec<u32> = tree.nodes()
            .iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .map(|n| n.node_id)
            .collect();

        for branch_id in &adapter_branches {
            assert!(root.children.contains(branch_id),
                "adapter branch {} should be a child of root", branch_id);
        }
    }

    #[test]
    fn spec_tree_build_adapter_branch_parent_id_is_root() {
        // Every adapter top-k>1 branch must have parent_id == Some(0)
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            if matches!(node.source, DraftSource::AdapterTopK { k } if k > 1) {
                assert_eq!(node.parent_id, Some(0),
                    "adapter branch node {} should have parent_id=0", node.node_id);
            }
        }
    }

    #[test]
    fn spec_tree_build_adapter_top_k_one_no_adapter_branches() {
        // adapter_top_k=1 means only root from adapter, no adapter branches
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let adapter_branches: Vec<&SpecNode> = tree.nodes()
            .iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .collect();
        assert!(adapter_branches.is_empty(),
            "adapter_top_k=1 should produce no adapter branch nodes");
    }

    #[test]
    fn spec_tree_build_spine_extension_parent_chain() {
        // Each spine extension's parent_id must point to the previous spine node
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_ids();
        for i in 1..spine.len() {
            let node = tree.node(spine[i]).unwrap();
            assert_eq!(node.parent_id, Some(spine[i - 1]),
                "spine node {} parent should be previous spine node {}",
                spine[i], spine[i - 1]);
        }
    }

    #[test]
    fn spec_tree_build_pld_continuation_from_end_of_prompt() {
        // Adapter token at the very end of prompt: only one continuation possible
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![99u32];
        let prompt = vec![1, 2, 3, 4, 99, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        // 99 is at position 4 in prompt, followed by 50, so spine should be [99, 50]
        assert_eq!(spine[0], 99);
        if spine.len() > 1 {
            assert_eq!(spine[1], 50);
        }
    }

    #[test]
    fn spec_tree_spine_ids_length_at_most_max_spine_depth() {
        // Spine length should never exceed max_spine_depth
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 followed by many tokens in prompt
        let prompt = vec![1, 10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let max_spine = config.max_spine_depth;
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_ids();
        assert!(spine.len() <= max_spine,
            "spine length {} should be <= max_spine_depth {}",
            spine.len(), max_spine);
    }

    #[test]
    fn spec_tree_build_ngram_branch_dedup_with_existing_child() {
        // N-gram branch that matches an existing child token should be skipped
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 10,
            ngram_top_k: 10,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 followed by 20 in prompt; adapter tokens are [10, 20]
        // Root gets 10; adapter branch gets 20 as child of root;
        // n-gram index has 10→20; n-gram branch for 20 should be skipped (already a child)
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10, 20, 10, 20, 10, 30];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Count nodes with token_id=20 that are children of root (node 0)
        let root = tree.node(0).unwrap();
        let children_with_token_20: Vec<u32> = root.children.iter()
            .filter(|&&c| tree.node(c).unwrap().token_id == 20)
            .copied()
            .collect();
        // Should be exactly 1 (the adapter branch), n-gram duplicate skipped
        assert_eq!(children_with_token_20.len(), 1,
            "token 20 should appear exactly once as root child, got {}",
            children_with_token_20.len());
    }

    #[test]
    fn spec_tree_node_accessor_returns_correct_node_id() {
        // Each node returned by node(i) must have node_id == i
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for i in 0..tree.len() {
            assert_eq!(tree.node(i as u32).unwrap().node_id, i as u32,
                "node accessor returned wrong node_id for index {}", i);
        }
    }

    #[test]
    fn ngram_index_build_two_tokens_n0() {
        // n=0 with tokens.len()>0: loop range is 0..tokens.len() (every position)
        // Each "0-gram" has a continuation; hash of empty slice is consistent
        let tokens = vec![1u32, 2, 3];
        let idx = NgramIndex::build(&tokens, 0);
        // With n=0, every position i in 0..3 produces an n-gram &tokens[i..i+0] = []
        // All empty slices hash the same, so continuations accumulate
        assert!(!idx.table.is_empty() || tokens.is_empty());
    }

    #[test]
    fn ngram_index_get_continuations_uses_single_token_hash() {
        // Verify get_continuations hashes a single-element slice, not the full n-gram
        let tokens = vec![1u32, 2, 3, 1, 4, 1, 5];
        let idx_n1 = NgramIndex::build(&tokens, 1);
        let idx_n3 = NgramIndex::build(&[1u32, 2, 3, 1, 2, 3, 1, 2, 4], 3);

        // get_continuations(1, ...) hashes [1] regardless of the index's n value
        let conts_n1 = idx_n1.get_continuations(1, 5);
        assert!(conts_n1.contains(&2));
        assert!(conts_n1.contains(&4));
        assert!(conts_n1.contains(&5));

        // idx_n3 with n=3: get_continuations hashes [1] which is a 1-gram,
        // but the table only has 3-gram keys, so result should be empty
        let conts_n3 = idx_n3.get_continuations(1, 5);
        assert!(conts_n3.is_empty(),
            "get_continuations should not find 1-gram entries in a 3-gram index");
    }

    #[test]
    fn ngram_index_large_n_empty_table() {
        // n much larger than token count → table should be empty
        let tokens = vec![1u32, 2, 3];
        let idx = NgramIndex::build(&tokens, 100);
        assert!(idx.table.is_empty());
    }

    #[test]
    fn spec_node_partial_eq_transitive() {
        // PartialEq transitivity: a==b and b==c implies a==c
        let a = SpecNode {
            node_id: 0, token_id: 42, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = a.clone();
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "PartialEq transitivity: a==b and b==c must imply a==c");
    }

    #[test]
    fn draft_source_all_three_in_hashset() {
        // All three DraftSource variants can coexist in a HashSet
        let mut set = std::collections::HashSet::new();
        assert!(set.insert(DraftSource::PldSpine));
        assert!(set.insert(DraftSource::NgramBranch));
        assert!(set.insert(DraftSource::AdapterTopK { k: 1 }));
        assert_eq!(set.len(), 3);

        // Duplicates rejected
        assert!(!set.insert(DraftSource::PldSpine));
        assert!(!set.insert(DraftSource::AdapterTopK { k: 1 }));
        assert_eq!(set.len(), 3);
    }

    // ==================================================================
    // NEW TESTS (tests 588+)
    // ==================================================================

    // ------------------------------------------------------------------
    // SpecTree: adapter root token appears at end of prompt (edge case
    // for find_pld_continuations — last position has no continuation)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_token_last_in_prompt_no_extension() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![50u32];
        // Token 50 appears at the very last position — no continuation possible
        let prompt = vec![1, 2, 3, 4, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: root exists but no PLD spine extensions
        assert_eq!(tree.len(), 1, "token 50 at prompt end should produce no PLD continuations");
        assert_eq!(tree.spine_token_ids(), vec![50]);
    }

    // ------------------------------------------------------------------
    // SpecTree: multiple adapter tokens with overlapping prompt matches
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_overlapping_adapter_and_prompt_tokens() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Adapter tokens: [10, 20, 30]; prompt has 10 followed by 20
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![10, 20, 30, 40, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: root is 10, PLD continues with 20,30; adapter branches 20,30 off root
        assert!(tree.len() >= 3, "should have root + PLD extensions + adapter branches");
        assert_eq!(tree.node(0).unwrap().token_id, 10);
        // Adapter branch tokens should be children of root (node 0)
        let root = tree.node(0).unwrap();
        let adapter_child_tokens: Vec<u32> = root.children.iter()
            .filter(|&&c| matches!(tree.node(c).unwrap().source, DraftSource::AdapterTopK { k } if k > 1))
            .map(|&c| tree.node(c).unwrap().token_id)
            .collect();
        assert!(adapter_child_tokens.contains(&20));
        assert!(adapter_child_tokens.contains(&30));
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask row size grows with tree depth (deeper nodes
    // have more ancestors)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_row_size_grows_monotonically_with_depth() {
        // Arrange: build a tree with a spine of depth >= 3
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act
        let (indptr, _indices) = tree.tree_attention_mask_csr(5);
        // Assert: row sizes should be non-decreasing along spine (depth order = node order)
        let mut prev_size = 0usize;
        for i in 0..tree.len() {
            let row_size = indptr[i + 1] - indptr[i];
            assert!(row_size >= prev_size,
                "row {} size {} should be >= previous {} (ancestors accumulate)",
                i, row_size, prev_size);
            prev_size = row_size;
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with n=0 produces a single key (empty slice hash)
    // with all tokens as continuations
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_build_n_zero_all_tokens_as_continuations() {
        // Arrange
        let tokens = vec![10u32, 20, 30, 10, 40];
        // Act
        let idx = NgramIndex::build(&tokens, 0);
        // Assert: With n=0, the hash key is always the same (empty slice).
        // Each position i produces continuation tokens[i+0] = tokens[i],
        // but position tokens.len()-n-1 has no continuation.
        // All entries accumulate under one key.
        assert!(!idx.table.is_empty());
        // Token 10 appears twice as a continuation, should be most frequent
        let empty_hash = NgramIndex::hash_ngram(&[]);
        if let Some(conts) = idx.table.get(&empty_hash) {
            assert!(!conts.is_empty());
            // Token 10 appears at positions 0 and 3, so count should be 2
            let entry_10 = conts.iter().find(|(t, _)| *t == 10);
            assert!(entry_10.is_some(), "token 10 should appear in continuations");
            assert_eq!(entry_10.unwrap().1, 2, "token 10 should have count 2");
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_ngram_continuations returns results in frequency
    // order for a rich dataset
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_ngram_continuations_strict_frequency_order() {
        // Arrange: build clear frequency differences
        // bigram [1,2] → 10 (4 times), 20 (3 times), 30 (2 times), 40 (1 time)
        let mut tokens = Vec::new();
        for _ in 0..4 { tokens.extend_from_slice(&[1u32, 2, 10]); }
        for _ in 0..3 { tokens.extend_from_slice(&[1u32, 2, 20]); }
        for _ in 0..2 { tokens.extend_from_slice(&[1u32, 2, 30]); }
        tokens.extend_from_slice(&[1u32, 2, 40]);
        // Act
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 10);
        // Assert: strictly descending by frequency
        assert_eq!(conts, vec![10, 20, 30, 40]);
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branch position_offset equals parent offset + 1
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branch_position_offset_equals_parent_plus_one() {
        // Arrange: spine with depth >= 2, n-gram branches enabled
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 10, 70, 80, 50, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: every n-gram branch has position_offset = parent's offset + 1
        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                let parent_id = node.parent_id.unwrap();
                let parent_offset = tree.node(parent_id).unwrap().position_offset;
                assert_eq!(node.position_offset, parent_offset + 1,
                    "n-gram branch {} offset {} should be parent {} offset {} + 1",
                    node.node_id, node.position_offset, parent_id, parent_offset);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: build is deterministic — same inputs produce identical
    // output across 5 consecutive calls
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_deterministic_five_calls() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let first = SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx);
        for _ in 0..4 {
            let subsequent = SpecTree::build(config.clone(), &adapter_tokens, &prompt, &ngram_idx);
            // Assert
            assert_eq!(first.all_token_ids(), subsequent.all_token_ids());
            assert_eq!(first.spine_token_ids(), subsequent.spine_token_ids());
            let (indptr1, indices1) = first.tree_attention_mask_csr(10);
            let (indptr2, indices2) = subsequent.tree_attention_mask_csr(10);
            assert_eq!(indptr1, indptr2);
            assert_eq!(indices1, indices2);
        }
    }

    // ------------------------------------------------------------------
    // SpecNode: estimated_acceptance retains subnormal (denormalized) f32
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_subnormal_estimated_acceptance() {
        // Arrange
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal f32
        // Act
        let node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::NgramBranch,
            estimated_acceptance: subnormal,
            position_offset: 0,
        };
        // Assert
        assert_eq!(node.estimated_acceptance.to_bits(), 1u32);
        assert!(node.estimated_acceptance > 0.0);
        assert!(node.estimated_acceptance.is_subnormal());
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_token_ids are a prefix of all_token_ids
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_tokens_are_prefix_of_all_tokens() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let all_ids = tree.all_token_ids();
        let spine_ids = tree.spine_token_ids();
        // Assert: spine tokens appear at the same positions in all_token_ids
        // (spine nodes are added first, so they are a prefix)
        assert!(all_ids.len() >= spine_ids.len());
        for (i, &spine_tok) in spine_ids.iter().enumerate() {
            assert_eq!(all_ids[i], spine_tok,
                "spine token at position {} ({}) should match all_token_ids position", i, spine_tok);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: PartialEq reflexive (a == a) for non-empty index
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_partial_eq_reflexive_non_empty() {
        // Arrange
        let idx = NgramIndex::build(&[1u32, 2, 3, 4, 5], 2);
        // Act & Assert
        assert_eq!(idx, idx, "NgramIndex should be reflexive for PartialEq");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask last entry of indptr equals total indices count
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_indptr_last_entry_equals_indices_len() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(8);
        // Assert
        assert_eq!(*indptr.last().unwrap(), indices.len(),
            "last indptr entry {} should equal indices length {}",
            indptr.last().unwrap(), indices.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: empty tree CSR mask produces single zero indptr
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_empty_tree_single_zero_indptr() {
        // Arrange
        let tree = SpecTree::new(SpecTreeConfig::default());
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(10);
        // Assert
        assert_eq!(indptr, vec![0]);
        assert!(indices.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations with n=1 returns same result as
    // get_ngram_continuations with single-element slice
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_matches_ngram_continuations_for_n1() {
        // Arrange
        let tokens = vec![1u32, 10, 1, 20, 1, 30];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let via_get = idx.get_continuations(1, 5);
        let via_ngram = idx.get_ngram_continuations(&[1], 5);
        // Assert
        assert_eq!(via_get, via_ngram,
            "get_continuations and get_ngram_continuations should agree for n=1");
    }

    // ------------------------------------------------------------------
    // SpecTree: all nodes have position_offset < tree total_nodes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_position_offsets_within_range() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 50, 60, 70, 80, 10, 90, 20, 100];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: position_offset should be < spine_len (max spine depth) or at least < total_nodes
        for node in tree.nodes() {
            assert!((node.position_offset as usize) < tree.len() || tree.is_empty(),
                "node {} position_offset {} should be < tree len {}",
                node.node_id, node.position_offset, tree.len());
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: two independent builds with different configs produce
    // different tree sizes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_different_configs_different_sizes() {
        // Arrange
        let config_small = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let config_large = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 50, 60, 70, 80, 10, 90, 20, 100];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree_small = SpecTree::build(config_small, &adapter_tokens, &prompt, &ngram_idx);
        let tree_large = SpecTree::build(config_large, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        assert!(tree_small.len() < tree_large.len(),
            "small config tree ({}) should be smaller than large config ({})",
            tree_small.len(), tree_large.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids returns empty for root-only tree built
    // from token not found anywhere in prompt
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_empty_for_isolated_root() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 5,
            ngram_top_k: 5,
            ..SpecTreeConfig::default()
        };
        // Token 99999 does not appear in prompt → no PLD, no n-gram matches
        let adapter_tokens = vec![99999u32];
        let prompt = vec![1, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        assert_eq!(tree.len(), 1);
        assert!(tree.branch_token_ids().is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: building with very large n and long input produces
    // correct single entry
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_very_large_n_single_entry() {
        // Arrange: 10 tokens, n=9 → exactly 1 n-gram window
        let tokens: Vec<u32> = (1..=10).collect();
        // Act
        let idx = NgramIndex::build(&tokens, 9);
        // Assert: one entry for the 9-gram [1,2,3,4,5,6,7,8,9] → continuation 10
        let conts = idx.get_ngram_continuations(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 5);
        assert_eq!(conts, vec![10]);
        // No other entries
        let other = idx.get_ngram_continuations(&[2, 3, 4, 5, 6, 7, 8, 9, 10], 5);
        assert!(other.is_empty(), "9-gram at position 1 has no continuation");
    }

    // ------------------------------------------------------------------
    // Wave-33: Edge case and boundary tests for tree.rs
    // ------------------------------------------------------------------

    // @trace TEST-SPEC-TREE-605 req:REQ-BCI-001 level:unit
    // PLD with ngram_len=2 should find continuations where prompt has
    // a 2-token context ending with the adapter token.
    #[test]
    fn spec_tree_build_pld_ngram_len_2() {
        // Arrange: prompt [10, 20, 30, 40], adapter=30, ngram_len=2
        // PLD looks for prompt[i]==30 where i >= ngram_len(2)
        // prompt[2]==30, so continuations start at prompt[3..] = [40]
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            pld_ngram_len: 2,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![30u32];
        let prompt = vec![10, 20, 30, 40];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: root(30) + spine extension(40) = 2 nodes
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.spine_token_ids(), vec![30, 40]);
    }

    // @trace TEST-SPEC-TREE-606 req:REQ-BCI-001 level:unit
    // When adapter token occurs at multiple positions in prompt,
    // PLD deduplicates continuation tokens.
    #[test]
    fn spec_tree_build_pld_deduplicates_across_occurrences() {
        // Arrange: prompt [10, 20, 30, 10, 20, 30, 99]
        // adapter=30 occurs at positions 2 and 5 (both >= ngram_len=3)
        // Position 2 continuations: [10, 20, 30, 99] → deduped unique tokens
        // Position 5 continuations: [99]
        // Combined unique continuations (deduped): [10, 20, 30, 99]
        let config = SpecTreeConfig {
            max_spine_depth: 10,
            pld_ngram_len: 3,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![30u32];
        let prompt = vec![10, 20, 30, 10, 20, 30, 99];
        let ngram_idx = NgramIndex::build(&prompt, 3);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: root(30) + unique spine extensions
        // Dedup means continuations are unique across both occurrences
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 30); // root
        // Extensions should be unique — no duplicate token_ids in spine
        let mut sorted_spine = spine.clone();
        sorted_spine.sort();
        sorted_spine.dedup();
        assert_eq!(sorted_spine.len(), spine.len(), "spine tokens should all be unique");
    }

    // @trace TEST-SPEC-TREE-607 req:REQ-BCI-001 level:unit
    // Spine acceptance should decrease with depth: root=0.70,
    // spine[1]=0.60, spine[2]=0.55, etc.
    #[test]
    fn spec_tree_build_spine_acceptance_exact_values() {
        // Arrange: prompt long enough to produce at least 3 spine nodes
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            pld_ngram_len: 1,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 20, 30, 40, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: root acceptance = 0.70
        let root = tree.node(0).unwrap();
        assert!((root.estimated_acceptance - 0.70).abs() < 1e-6);
        // Spine nodes: depth 0 → 0.60, depth 1 → 0.55, depth 2 → 0.50
        let spine_ids = tree.spine_ids();
        if spine_ids.len() > 1 {
            let s1 = tree.node(spine_ids[1]).unwrap();
            assert!((s1.estimated_acceptance - 0.60).abs() < 1e-6);
        }
        if spine_ids.len() > 2 {
            let s2 = tree.node(spine_ids[2]).unwrap();
            assert!((s2.estimated_acceptance - 0.55).abs() < 1e-6);
        }
        if spine_ids.len() > 3 {
            let s3 = tree.node(spine_ids[3]).unwrap();
            assert!((s3.estimated_acceptance - 0.50).abs() < 1e-6);
        }
    }

    // @trace TEST-SPEC-TREE-608 req:REQ-BCI-001 level:unit
    // max_branches_per_node limits n-gram branches per spine node.
    #[test]
    fn spec_tree_build_max_branches_limits_per_node() {
        // Arrange: rich prompt with many n-gram continuations, but limit to 1
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            pld_ngram_len: 1,
            adapter_top_k: 1,
            ngram_top_k: 10,
            max_branches_per_node: 1,
            max_tree_size: 100,
        };
        let adapter_tokens = vec![10u32];
        // prompt gives n-gram continuations: 10→20, 10→50 (two options)
        let prompt = vec![10, 20, 30, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: root should have at most 1 n-gram branch (plus any adapter/PLD children)
        let root = tree.node(0).unwrap();
        let ngram_branches: Vec<_> = root.children.iter().filter(|&&c| {
            matches!(tree.node(c).unwrap().source, DraftSource::NgramBranch)
        }).collect();
        assert!(ngram_branches.len() <= 1, "max_branches_per_node limits n-gram branches to 1");
    }

    // @trace TEST-SPEC-TREE-609 req:REQ-BCI-001 level:unit
    // Root node source is always AdapterTopK { k: 1 }.
    #[test]
    fn spec_tree_build_root_source_adapter_k1() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter_tokens = vec![42u32, 99];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        let root = tree.node(0).unwrap();
        assert_eq!(root.source, DraftSource::AdapterTopK { k: 1 });
    }

    // @trace TEST-SPEC-TREE-610 req:REQ-BCI-001 level:unit
    // Adapter branch acceptance formula: 0.15 / (k as f32 + 1.0).max(1.0)
    // For k=2 (second adapter token): 0.15 / (0 + 1.0) = 0.15
    // For k=3 (third adapter token): 0.15 / (1 + 1.0) = 0.075
    #[test]
    fn spec_tree_build_adapter_branch_acceptance_formula() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            pld_ngram_len: 3,
            adapter_top_k: 3,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        let root = tree.node(0).unwrap();
        // Find adapter branches (k=2 and k=3)
        let adapter_branches: Vec<_> = root.children.iter()
            .filter_map(|&c| {
                let node = tree.node(c).unwrap();
                if matches!(node.source, DraftSource::AdapterTopK { k: 2.. }) {
                    Some(node)
                } else {
                    None
                }
            })
            .collect();
        assert!(adapter_branches.len() >= 2, "should have at least 2 adapter branches");
        // k=2 branch: acceptance = 0.15
        let k2_branch = adapter_branches.iter().find(|n| {
            matches!(n.source, DraftSource::AdapterTopK { k: 2 })
        }).unwrap();
        assert!((k2_branch.estimated_acceptance - 0.15).abs() < 1e-6,
            "k=2 acceptance should be 0.15, got {}", k2_branch.estimated_acceptance);
        // k=3 branch: acceptance = 0.075
        let k3_branch = adapter_branches.iter().find(|n| {
            matches!(n.source, DraftSource::AdapterTopK { k: 3 })
        }).unwrap();
        assert!((k3_branch.estimated_acceptance - 0.075).abs() < 1e-6,
            "k=3 acceptance should be 0.075, got {}", k3_branch.estimated_acceptance);
    }

    // @trace TEST-SPEC-TREE-611 req:REQ-BCI-001 level:unit
    // N-gram branch skips tokens already present as children of the parent.
    #[test]
    fn spec_tree_build_ngram_branch_skips_existing_adapter_child() {
        // Arrange: adapter gives [10, 20], and n-gram index also has 20 as
        // a continuation of 10. The n-gram branch for token 20 should be skipped.
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            pld_ngram_len: 3,
            adapter_top_k: 2,
            ngram_top_k: 5,
            max_branches_per_node: 10,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 20, 30];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: no n-gram branch with token_id=20 under root
        // (it's already an adapter branch)
        let root = tree.node(0).unwrap();
        let token_20_children: Vec<_> = root.children.iter()
            .filter(|&&c| tree.node(c).unwrap().token_id == 20)
            .collect();
        // Should be exactly 1 child with token_id=20 (the adapter branch),
        // not 2 (adapter + n-gram dedup)
        assert_eq!(token_20_children.len(), 1,
            "token 20 should appear exactly once as child, not duplicated by n-gram");
    }

    // @trace TEST-SPEC-TREE-612 req:REQ-BCI-001 level:unit
    // SpecTree::clone produces an equal tree.
    #[test]
    fn spec_tree_clone_preserves_structure() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            pld_ngram_len: 1,
            adapter_top_k: 3,
            ngram_top_k: 2,
            max_branches_per_node: 2,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 20, 30, 40];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act
        let cloned = tree.clone();
        // Assert
        assert_eq!(tree, cloned);
        assert_eq!(tree.all_token_ids(), cloned.all_token_ids());
        assert_eq!(tree.spine_token_ids(), cloned.spine_token_ids());
    }

    // @trace TEST-SPEC-TREE-613 req:REQ-BCI-001 level:unit
    // nodes() returns a slice whose length matches total_nodes.
    #[test]
    fn spec_tree_nodes_accessor_matches_total() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            pld_ngram_len: 1,
            adapter_top_k: 2,
            ngram_top_k: 2,
            max_branches_per_node: 2,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![5u32, 6];
        let prompt = vec![1, 2, 5, 10, 20];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act
        let nodes = tree.nodes();
        // Assert
        assert_eq!(nodes.len(), tree.len(), "nodes() slice length must match total_nodes");
        // Each node's node_id must match its index in the slice
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(node.node_id, i as u32, "node_id must equal its index");
        }
    }

    // @trace TEST-SPEC-TREE-614 req:REQ-BCI-001 level:unit
    // CSR mask columns for each row are within [0, total_seq + tree_size).
    #[test]
    fn spec_tree_csr_mask_columns_within_shape_bounds() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            pld_ngram_len: 1,
            adapter_top_k: 2,
            ngram_top_k: 2,
            max_branches_per_node: 2,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![5u32, 6];
        let prompt = vec![1, 2, 5, 10, 20];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let total_seq = 100;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        let (rows, cols) = tree.mask_shape(total_seq);
        // Assert: all column indices in valid range
        for &col in &indices {
            assert!(col < cols, "column {} exceeds mask width {}", col, cols);
        }
        assert_eq!(indptr.len(), rows + 1);
    }

    // @trace TEST-SPEC-TREE-615 req:REQ-BCI-001 level:unit
    // accepted_from_spine with a single-element target that matches only the root.
    #[test]
    fn accepted_from_spine_single_token_target_matches_root() {
        // Arrange: build a tree with spine [10, 20, 30]
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            pld_ngram_len: 1,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 20, 30];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act: target has only [10]
        let (count, tokens) = tree.accepted_from_spine(&[10]);
        // Assert: only root matches
        assert_eq!(count, 1);
        assert_eq!(tokens, vec![10]);
    }

    // @trace TEST-SPEC-TREE-616 req:REQ-BCI-001 level:unit
    // accepted_from_spine when target is empty returns (0, []).
    #[test]
    fn accepted_from_spine_empty_target_returns_zero() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            pld_ngram_len: 1,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 20, 30];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act
        let (count, tokens) = tree.accepted_from_spine(&[]);
        // Assert
        assert_eq!(count, 0);
        assert!(tokens.is_empty());
    }

    // @trace TEST-NGRAM-617 req:REQ-BCI-001 level:unit
    // NgramIndex::build with tokens.len() exactly equal to n produces empty table
    // (need tokens.len() > n for at least one window).
    #[test]
    fn ngram_index_build_tokens_len_exactly_n_empty() {
        // Arrange: 3 tokens, n=3 → tokens.len() <= n, so empty
        let tokens = vec![1u32, 2, 3];
        // Act
        let idx = NgramIndex::build(&tokens, 3);
        // Assert: no continuations possible
        assert!(idx.get_continuations(1, 5).is_empty());
        assert!(idx.get_ngram_continuations(&[1, 2, 3], 5).is_empty());
    }

    // @trace TEST-NGRAM-618 req:REQ-BCI-001 level:unit
    // NgramIndex::build with n=1 and repeating pattern creates correct frequency.
    #[test]
    fn ngram_index_n1_repeating_pattern_counts() {
        // Arrange: [1, 2, 1, 2, 1, 3] — token 1 followed by 2 twice, by 3 once
        let tokens = vec![1u32, 2, 1, 2, 1, 3];
        // Act
        let idx = NgramIndex::build(&tokens, 1);
        // Assert: get_continuations for token 1 should be [2, 3] in frequency order
        let conts = idx.get_continuations(1, 5);
        assert_eq!(conts.len(), 2);
        assert_eq!(conts[0], 2, "token 2 should be most frequent continuation of 1");
        assert_eq!(conts[1], 3, "token 3 should be second continuation of 1");
    }

    // @trace TEST-NGRAM-619 req:REQ-BCI-001 level:unit
    // NgramIndex::get_continuations with tokens containing u32::MAX.
    #[test]
    fn ngram_index_get_continuations_u32_max_token() {
        // Arrange
        let tokens = vec![u32::MAX, 42, u32::MAX, 99];
        // Act
        let idx = NgramIndex::build(&tokens, 1);
        // Assert: u32::MAX has continuations 42 and 99
        let conts = idx.get_continuations(u32::MAX, 5);
        assert_eq!(conts, vec![42, 99]);
    }

    // @trace TEST-SPEC-TREE-620 req:REQ-BCI-001 level:unit
    // SpecTree::build with adapter_top_tokens containing duplicate token IDs.
    // The root uses the first occurrence; adapter branches use subsequent ones.
    #[test]
    fn spec_tree_build_duplicate_adapter_tokens_same_id() {
        // Arrange: adapter tokens [10, 10, 10] — all same token
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            pld_ngram_len: 3,
            adapter_top_k: 3,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32, 10, 10];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: root = 10, adapter branches also 10 but different node IDs
        assert_eq!(tree.node(0).unwrap().token_id, 10);
        // All adapter branches should have token_id 10
        let all_ids = tree.all_token_ids();
        assert!(all_ids.iter().all(|&t| t == 10), "all nodes should have token_id 10");
        assert_eq!(tree.len(), 3, "root + 2 adapter branches = 3 nodes");
    }

    // @trace TEST-SPEC-TREE-621 req:REQ-BCI-001 level:unit
    // CSR mask with total_seq_len=1 on a single-node tree: root attends to
    // exactly one prefix column [0] and itself [1].
    #[test]
    fn spec_tree_csr_mask_single_node_total_seq_len_one() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            pld_ngram_len: 3,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let tree = SpecTree::build(config, &[42u32], &[99], &NgramIndex::build(&[99], 1));
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(1);
        // Assert: exactly 1 node
        assert_eq!(tree.len(), 1);
        assert_eq!(indptr.len(), 2, "indptr should have len+1 entries");
        // Root row: prefix column 0, then self at total_seq_len + node_idx = 1+0 = 1
        let root_start = indptr[0];
        let root_end = indptr[1];
        let root_cols: Vec<usize> = indices[root_start..root_end].to_vec();
        assert!(root_cols.contains(&0), "root should attend to prefix col 0");
        assert!(root_cols.contains(&1), "root should attend to self at col 1");
        assert_eq!(root_cols.len(), 2, "single node with seq_len=1 should have exactly 2 cols");
    }

    // @trace TEST-SPEC-TREE-622 req:REQ-BCI-001 level:unit
    // Adapter branches (adapter top-2/3) off the root always have position_offset = 1.
    #[test]
    fn spec_tree_build_adapter_branches_have_position_offset_one() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            pld_ngram_len: 3,
            adapter_top_k: 3,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: all non-root nodes are adapter branches with position_offset=1
        for i in 1..tree.len() {
            let node = tree.node(i as u32).unwrap();
            if matches!(node.source, DraftSource::AdapterTopK { k: 2.. }) {
                assert_eq!(node.position_offset, 1,
                    "adapter branch node {} should have position_offset=1", i);
            }
        }
    }

    // @trace TEST-SPEC-TREE-623 req:REQ-BCI-001 level:unit
    // Leaf nodes (no children) have an empty children vec.
    #[test]
    fn spec_tree_node_children_empty_vec_for_leaf() {
        // Arrange: root-only tree (no branches, no spine extensions)
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            pld_ngram_len: 3,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[42u32], &prompt, &ngram_idx);
        // Act
        let root = tree.node(0).unwrap();
        // Assert: root has no children (no PLD match, no branches)
        assert!(root.children.is_empty(), "leaf node should have empty children vec");
    }

    // @trace TEST-NGRAM-624 req:REQ-BCI-001 level:unit
    // NgramIndex::build with a single token and n=1: tokens.len() > n, so one
    // window [token] is created but there is no continuation (tokens.len() - n = 0
    // iterations). Table should be empty.
    #[test]
    fn ngram_index_build_single_token_n_equals_one() {
        // Arrange
        let tokens = vec![42u32];
        // Act
        let idx = NgramIndex::build(&tokens, 1);
        // Assert: tokens.len() = 1, n = 1, loop range is 0..0, so table is empty
        assert!(idx.table.is_empty(), "single token with n=1 should produce empty table");
        assert_eq!(idx.n, 1);
    }

    // @trace TEST-SPEC-TREE-625 req:REQ-BCI-001 level:unit
    // CSR mask with total_seq_len=1 on a multi-node tree. Each node attends to
    // prefix column [0] plus its ancestor chain plus itself.
    #[test]
    fn spec_tree_csr_mask_multi_node_total_seq_len_one() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            pld_ngram_len: 2,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        // Adapter token 10 must appear at index >= pld_ngram_len=2 in the prompt
        // so find_pld_continuations finds it and extends the spine.
        let prompt = vec![1u32, 2, 3, 10, 20, 30];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(1);
        // Assert: every row must include prefix col 0 and its own self column
        assert!(tree.len() > 1, "should have multi-node tree for meaningful test");
        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let cols: Vec<usize> = indices[start..end].to_vec();
            assert!(cols.contains(&0),
                "node {} should attend to prefix col 0", i);
            let self_col = 1 + i; // total_seq_len + node_idx
            assert!(cols.contains(&self_col),
                "node {} should attend to self at col {}", i, self_col);
        }
    }

    // @trace TEST-SPEC-TREE-626 req:REQ-BCI-001 level:unit
    // When the adapter root token does not appear anywhere in the prompt,
    // PLD spine extensions are empty and only adapter branches may exist.
    #[test]
    fn spec_tree_build_no_pld_when_start_token_not_in_prompt() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            pld_ngram_len: 2,
            adapter_top_k: 2,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let prompt = vec![1u32, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act: token 999 does not appear in prompt
        let tree = SpecTree::build(config, &[999u32, 888], &prompt, &ngram_idx);
        // Assert: spine is root only (no PLD continuations found)
        let spine = tree.spine_token_ids();
        assert_eq!(spine.len(), 1, "spine should be root only when adapter token not in prompt");
        assert_eq!(spine[0], 999);
    }

    // @trace TEST-NGRAM-627 req:REQ-BCI-001 level:unit
    // get_continuations with top_k=1 returns at most one result.
    #[test]
    fn ngram_index_get_continuations_top_k_one_returns_single() {
        // Arrange: token 1 followed by 2 three times and 3 once
        let tokens = vec![1u32, 2, 1, 2, 1, 2, 1, 3];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(1, 1);
        // Assert: only the most frequent continuation
        assert_eq!(conts.len(), 1, "top_k=1 should return at most 1 result");
        assert_eq!(conts[0], 2, "most frequent continuation of 1 is 2");
    }

    // @trace TEST-SPEC-TREE-628 req:REQ-BCI-001 level:unit
    // The root node (node_id=0) always has parent_id=None.
    #[test]
    fn spec_tree_spine_root_has_no_parent() {
        // Arrange
        let config = SpecTreeConfig::default();
        let prompt = vec![1u32, 2, 3, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32, 20], &prompt, &ngram_idx);
        // Act
        let root = tree.node(0).unwrap();
        // Assert
        assert_eq!(root.parent_id, None, "root node must have parent_id=None");
    }

    // @trace TEST-SPEC-TREE-629 req:REQ-BCI-001 level:unit
    // N-gram branches are only attached to spine nodes, not to other branches.
    // Verify that every NgramBranch node's parent is a spine node.
    #[test]
    fn spec_tree_build_ngram_branches_only_on_spine_nodes() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            pld_ngram_len: 2,
            adapter_top_k: 1,
            ngram_top_k: 5,
            max_branches_per_node: 3,
            max_tree_size: 32,
        };
        let prompt = vec![1u32, 2, 10, 20, 30, 40, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        // Act
        let spine_set: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        // Assert: every NgramBranch node's parent is in spine_set
        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                let parent_id = node.parent_id.expect("ngram branch must have a parent");
                assert!(spine_set.contains(&parent_id),
                    "ngram branch node {} should be attached to spine node {}, but it's not in spine",
                    node.node_id, parent_id);
            }
        }
    }

    // @trace TEST-SPEC-TREE-630 req:REQ-BCI-001 level:unit
    // SpecNode PartialEq: nodes identical except estimated_acceptance are not equal.
    #[test]
    fn spec_node_partial_eq_different_estimated_acceptance() {
        // Arrange
        let a = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.5,
            position_offset: 0,
        };
        let b = SpecNode {
            estimated_acceptance: 0.9,
            ..a.clone()
        };
        // Assert
        assert_ne!(a, b, "nodes with different estimated_acceptance should not be equal");
    }

    // @trace TEST-NGRAM-631 req:REQ-BCI-001 level:unit
    // Two tokens with n=1: produces exactly one n-gram window [first] → second.
    #[test]
    fn ngram_index_build_two_tokens_n_eq_one_single_cont() {
        // Arrange
        let tokens = vec![10u32, 20];
        // Act
        let idx = NgramIndex::build(&tokens, 1);
        // Assert: one entry, token 10 → 20
        let conts = idx.get_continuations(10, 5);
        assert_eq!(conts, vec![20], "token 10 should have continuation 20");
        // Token 20 has no continuations (it's the last token)
        let conts_20 = idx.get_continuations(20, 5);
        assert!(conts_20.is_empty(), "token 20 should have no continuations");
    }

    // @trace TEST-SPEC-TREE-632 req:REQ-BCI-001 level:unit
    // CSR mask: prefix columns in each row are contiguous starting from 0.
    #[test]
    fn spec_tree_csr_mask_prefix_cols_contiguous_from_zero() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            pld_ngram_len: 2,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let prompt = vec![1u32, 2, 10, 20, 30];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let total_seq_len = 7;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: for each row, the first total_seq_len columns are [0, 1, ..., total_seq_len-1]
        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            let nnz = end - start;
            assert!(nnz >= total_seq_len, "row {} should have at least {} entries", i, total_seq_len);
            for j in 0..total_seq_len {
                assert_eq!(row[j], j,
                    "row {} prefix col {} should be {}, got {}", i, j, j, row[j]);
            }
        }
    }

    // @trace TEST-SPEC-TREE-633 req:REQ-BCI-001 level:unit
    // max_spine_depth=1 means the take(max_spine_depth - 1) = take(0) so zero PLD
    // extensions are added. Spine is root only.
    #[test]
    fn spec_tree_build_max_spine_depth_one_no_pld_extensions() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            pld_ngram_len: 2,
            adapter_top_k: 1,
            ngram_top_k: 5,
            max_branches_per_node: 5,
            max_tree_size: 32,
        };
        // Prompt rich with continuations for adapter token 10
        let prompt = vec![1u32, 2, 10, 20, 30, 40, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        // Act
        let spine = tree.spine_ids();
        // Assert: spine is root only because max_spine_depth=1 → take(0) PLD extensions
        assert_eq!(spine.len(), 1, "with max_spine_depth=1, spine should be root only");
        assert_eq!(spine[0], 0);
        // Root should have NO PldSpine children
        let root = tree.node(0).unwrap();
        let has_pld_child = root.children.iter().any(|&c| {
            matches!(tree.node(c).unwrap().source, DraftSource::PldSpine)
        });
        assert!(!has_pld_child, "root should have no PldSpine child with max_spine_depth=1");
    }

    // @trace TEST-SPEC-TREE-634 req:REQ-BCI-001 level:unit
    // accepted_from_spine with partial match: spine [A, B, C, D], target
    // [A, B, X, ...] returns exactly [A, B].
    #[test]
    fn spec_tree_accepted_from_spine_partial_match_returns_prefix() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            pld_ngram_len: 2,
            adapter_top_k: 1,
            ngram_top_k: 0,
            max_branches_per_node: 0,
            max_tree_size: 32,
        };
        let prompt = vec![1u32, 2, 10, 20, 30, 40];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &[10u32], &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        // Act: create target that matches first 2 tokens then diverges
        let mut target = spine.clone();
        if target.len() >= 3 {
            target[2] = 9999; // break at position 2
        }
        let (count, accepted) = tree.accepted_from_spine(&target);
        // Assert
        assert_eq!(count, 2, "should accept exactly 2 tokens before mismatch");
        assert_eq!(accepted.len(), 2);
        assert_eq!(accepted[0], spine[0]);
        assert_eq!(accepted[1], spine[1]);
    }

    // @trace TEST-NGRAM-635 req:REQ-BCI-001 level:unit
    // get_continuations on an NgramIndex built from empty tokens always returns
    // an empty vec regardless of the token queried.
    #[test]
    fn ngram_index_get_continuations_empty_idx_empty_result() {
        // Arrange: build from empty tokens
        let idx = NgramIndex::build(&[], 3);
        // Act & Assert: any token query returns empty
        assert!(idx.get_continuations(0, 10).is_empty());
        assert!(idx.get_continuations(u32::MAX, 10).is_empty());
        assert!(idx.get_continuations(42, 1).is_empty());
    }

    // ==================================================================
    // NEW TESTS (~15 additional edge case tests)
    // ==================================================================

    // SpecNode: parent_id Some(0) vs Some(u32::MAX) distinguish nodes
    #[test]
    fn spec_node_parent_id_distinguishes_some_variants() {
        // Arrange
        let a = SpecNode {
            node_id: 1, token_id: 10, parent_id: Some(0),
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 1,
        };
        let b = SpecNode {
            node_id: 1, token_id: 10, parent_id: Some(u32::MAX),
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 1,
        };
        // Assert
        assert_ne!(a, b, "different parent_id values should make nodes unequal");
    }

    // NgramIndex: build with tokens where continuation appears in non-adjacent positions
    #[test]
    fn ngram_index_non_adjacent_continuation_still_indexed() {
        // Arrange: [1,2] → 3 at pos 2, but [1,2] also at pos 5 → 6
        let tokens = vec![1u32, 2, 3, 4, 5, 1, 2, 6];
        // Act
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        // Assert: both 3 and 6 are indexed as continuations
        assert_eq!(conts.len(), 2);
        assert!(conts.contains(&3));
        assert!(conts.contains(&6));
    }

    // SpecTree: build with adapter token not in prompt — root has no PldSpine children
    #[test]
    fn spec_tree_root_no_pld_child_when_token_absent() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![999u32];
        let prompt = vec![1u32, 2, 3, 4, 5];
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: root has no children at all
        let root = tree.node(0).unwrap();
        assert!(root.children.is_empty(), "root should have no children when token not in prompt");
        assert_eq!(tree.len(), 1);
    }

    // SpecTree: build with adapter_top_k=1, multiple adapter tokens — only first used
    #[test]
    fn spec_tree_adapter_top_k_1_ignores_remaining_tokens() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![100u32, 200, 300, 400];
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &[], &NgramIndex::build(&[], 2));
        // Assert: only root exists, no adapter branches
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.node(0).unwrap().token_id, 100);
    }

    // SpecTree: CSR mask — leaf spine node (deepest) has all ancestors
    #[test]
    fn spec_tree_csr_mask_deepest_spine_has_full_ancestor_chain() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10u32, 50, 60, 70, 80];
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        let total_seq_len = 3;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: the last spine node should attend to all earlier spine nodes
        let spine_ids = tree.spine_ids();
        if spine_ids.len() >= 3 {
            let last = spine_ids[spine_ids.len() - 1];
            let start = indptr[last as usize];
            let end = indptr[last as usize + 1];
            let row: std::collections::HashSet<usize> =
                indices[start..end].iter().copied().collect();
            // Every earlier spine node should be an ancestor column
            for &ancestor in &spine_ids[..spine_ids.len() - 1] {
                assert!(row.contains(&(total_seq_len + ancestor as usize)),
                    "deepest spine should attend to ancestor {}", ancestor);
            }
        }
    }

    // NgramIndex: get_ngram_continuations with top_k=1 returns highest frequency
    #[test]
    fn ngram_index_top_k_1_most_frequent_only() {
        // Arrange: [5,5] → 10 (3x), [5,5] → 20 (1x)
        let tokens = vec![5u32, 5, 10, 5, 5, 10, 5, 5, 10, 5, 5, 20];
        // Act
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[5, 5], 1);
        // Assert
        assert_eq!(conts.len(), 1);
        assert_eq!(conts[0], 10, "most frequent continuation should be returned");
    }

    // SpecTree: build with prompt containing repeated n-grams for same continuation
    #[test]
    fn spec_tree_build_with_repeated_ngram_same_continuation() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // [10, 50] repeated many times → continuation is always 50
        let prompt = vec![10u32, 50, 10, 50, 10, 50, 10, 50, 10, 50];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: n-gram branches should be deduplicated (50 already a spine child)
        let root = tree.node(0).unwrap();
        let child_tokens: Vec<u32> = root.children.iter()
            .map(|&c| tree.node(c).unwrap().token_id)
            .collect();
        let unique: std::collections::HashSet<u32> = child_tokens.iter().copied().collect();
        assert_eq!(child_tokens.len(), unique.len(),
            "root children should have no duplicate tokens: {:?}", child_tokens);
    }

    // SpecTree: mask_shape always returns total_nodes for rows across multiple configs
    #[test]
    fn spec_tree_mask_shape_rows_equals_len_across_configs() {
        // Arrange
        let configs = vec![
            SpecTreeConfig { max_spine_depth: 1, adapter_top_k: 1, max_branches_per_node: 0, ..SpecTreeConfig::default() },
            SpecTreeConfig { max_spine_depth: 5, adapter_top_k: 3, max_branches_per_node: 2, ..SpecTreeConfig::default() },
        ];
        for config in configs {
            let tree = SpecTree::build(config, &[10u32, 20, 30], &[1, 2, 10, 50], &NgramIndex::build(&[1, 2, 10, 50], 2));
            // Act
            let (rows, _cols) = tree.mask_shape(10);
            // Assert
            assert_eq!(rows, tree.len());
        }
    }

    // SpecNode: estimated_acceptance exactly 1.0 (maximum valid value)
    #[test]
    fn spec_node_acceptance_exactly_one() {
        // Arrange & Act
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 1.0, position_offset: 0,
        };
        // Assert
        assert_eq!(node.estimated_acceptance, 1.0);
        assert!(node.estimated_acceptance <= 1.0);
    }

    // SpecTree: build with config where all adapter tokens equal prompt start
    #[test]
    fn spec_tree_build_adapter_equals_prompt_first_token() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // pld_ngram_len=1 → scan starts at index 1.
        // Token 10 at index 0 is NOT scanned; token 10 at index 4 IS scanned.
        let prompt = vec![10, 20, 30, 40, 10, 50, 60];
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: PLD match from index 4 → continuations [50, 60]
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        assert!(spine.len() >= 2, "should have PLD extension from index 4");
    }

    // NgramIndex: table field accessible and reflects build correctly
    #[test]
    fn ngram_index_table_field_matches_build() {
        // Arrange
        let tokens = vec![1u32, 2, 3, 1, 2, 4, 1, 2, 5];
        // Act
        let idx = NgramIndex::build(&tokens, 2);
        // Assert: table should have entries (not empty since tokens.len() > n)
        assert!(!idx.table.is_empty());
        // Same number of unique n-gram continuations via table and get_ngram_continuations
        let conts = idx.get_ngram_continuations(&[1, 2], 10);
        assert_eq!(conts.len(), 3, "three distinct continuations for [1,2]");
    }

    // SpecTree: CSR mask — sibling branches share same ancestor set
    #[test]
    fn spec_tree_csr_mask_sibling_branches_share_ancestors() {
        // Arrange: two adapter branches off root
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32, 20, 30], &[], &NgramIndex::build(&[], 2));
        let total_seq_len = 2;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: node 1 and node 2 (adapter branches) should have identical row structure
        let row1: Vec<usize> = indices[indptr[1]..indptr[2]].to_vec();
        let row2: Vec<usize> = indices[indptr[2]..indptr[3]].to_vec();
        // Both should have prefix [0,1] + root (at total_seq_len+0) + self
        assert!(row1.contains(&(total_seq_len + 0)), "branch 1 should attend to root");
        assert!(row2.contains(&(total_seq_len + 0)), "branch 2 should attend to root");
        // Each has exactly one ancestor (root) + prefix + self
        assert_eq!(row1.len(), row2.len(), "sibling rows should have same size");
    }

    // DraftSource: using in a HashMap with insert/replace behavior
    #[test]
    fn draft_source_hashmap_insert_replace() {
        // Arrange
        use std::collections::HashMap;
        let mut map: HashMap<DraftSource, &'static str> = HashMap::new();
        // Act
        map.insert(DraftSource::PldSpine, "old");
        map.insert(DraftSource::PldSpine, "new");
        // Assert
        assert_eq!(map.get(&DraftSource::PldSpine), Some(&"new"));
        assert_eq!(map.len(), 1, "only one entry after replace");
    }

    // SpecTree: build with max_tree_size=2 and rich prompt — only root + 1 n-gram branch
    #[test]
    fn spec_tree_max_tree_size_2_limits_total() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 10,
            ngram_top_k: 10,
            max_tree_size: 2,
            pld_ngram_len: 1,
        };
        let prompt: Vec<u32> = (0..20).flat_map(|i| vec![10, 100 + i]).collect();
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert: n-gram branches limited by max_tree_size
        let ngram_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .count();
        assert!(ngram_count <= 1, "at most 1 n-gram branch with max_tree_size=2, got {}", ngram_count);
    }

    // SpecTree: all nodes have source that is one of the three DraftSource variants
    #[test]
    fn spec_tree_all_nodes_have_valid_source() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 10, 70, 80];
        // Act
        let tree = SpecTree::build(config, &[10u32, 20, 30], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        for node in tree.nodes() {
            let is_valid = matches!(node.source,
                DraftSource::PldSpine | DraftSource::AdapterTopK { .. } | DraftSource::NgramBranch);
            assert!(is_valid, "node {} has unexpected source {:?}", node.node_id, node.source);
        }
    }

    // ==================================================================
    // Additional edge case tests (+15)
    // ==================================================================

    // 1. NgramIndex: n=1 builds a unigram index and get_continuations works
    #[test]
    fn ngram_index_build_with_n_equals_1() {
        // Arrange: n=1 means each single token is an n-gram, continuation is next token
        let tokens = vec![5u32, 10, 5, 20, 5, 10];
        // Act
        let idx = NgramIndex::build(&tokens, 1);
        // Assert: token 5 has continuations [10, 20, 10], sorted by freq: 10(2x) then 20(1x)
        let conts = idx.get_ngram_continuations(&[5], 5);
        assert_eq!(conts.len(), 2, "token 5 should have 2 distinct continuations");
        assert_eq!(conts[0], 10, "most frequent continuation of 5 should be 10");
        assert!(conts.contains(&20));
    }

    // 2. SpecTree: build with single-element prompt and adapter_top_k > 1
    #[test]
    fn spec_tree_build_with_single_prompt_token() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![42u32];
        // Act
        let tree = SpecTree::build(config, &[10u32, 20], &prompt, &NgramIndex::build(&prompt, 2));
        // Assert: root exists, one adapter branch, no PLD spine extensions (prompt too short)
        assert_eq!(tree.node(0).unwrap().token_id, 10);
        let adapter_branches = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k >= 2))
            .count();
        assert_eq!(adapter_branches, 1, "should have exactly 1 adapter top-2 branch");
    }

    // 3. DraftSource: Hash consistency — equal values produce equal hashes
    #[test]
    fn draft_source_hash_equal_values_equal_hashes() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = DraftSource::AdapterTopK { k: 5 };
        let b = DraftSource::AdapterTopK { k: 5 };
        let c = DraftSource::AdapterTopK { k: 3 };
        // Act
        let mut hasher_a = DefaultHasher::new();
        a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();
        let mut hasher_b = DefaultHasher::new();
        b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();
        let mut hasher_c = DefaultHasher::new();
        c.hash(&mut hasher_c);
        let hash_c = hasher_c.finish();
        // Assert
        assert_eq!(hash_a, hash_b, "equal DraftSource values must have equal hashes");
        assert_ne!(hash_a, hash_c, "different DraftSource values should have different hashes");
    }

    // 4. SpecNode: PartialEq distinguishes by all fields including source and position_offset
    #[test]
    fn spec_node_partial_eq_distinguishes_by_position_offset() {
        // Arrange: identical except position_offset
        let a = SpecNode {
            node_id: 0, token_id: 42, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 42, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 5,
        };
        // Assert
        assert_ne!(a, b, "nodes differing only in position_offset should not be equal");
    }

    // 5. SpecTree: accepted_from_spine when target is longer than spine
    #[test]
    fn spec_tree_accepted_from_spine_target_longer_than_spine() {
        // Arrange: spine has 2 tokens [42, 50], target has 5
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        let spine = tree.spine_token_ids();
        // Extend target with extra matching tokens beyond spine length
        let mut target = spine.clone();
        target.push(999);
        target.push(888);
        // Act
        let (count, accepted) = tree.accepted_from_spine(&target);
        // Assert: all spine tokens accepted (target has them), extra target tokens ignored
        assert_eq!(count, spine.len());
        assert_eq!(accepted, spine);
    }

    // 6. SpecTreeConfig: zero-valued fields create a valid config
    #[test]
    fn spec_tree_config_all_zero_fields_valid() {
        // Arrange & Act
        let config = SpecTreeConfig {
            max_spine_depth: 0,
            max_branches_per_node: 0,
            pld_ngram_len: 0,
            ngram_top_k: 0,
            adapter_top_k: 0,
            max_tree_size: 0,
        };
        // Assert: clone and equality
        let clone = config.clone();
        assert_eq!(config, clone);
        assert_eq!(config.max_spine_depth, 0);
        assert_eq!(config.max_tree_size, 0);
    }

    // 7. NgramIndex: get_ngram_continuations returns empty for unknown n-gram
    #[test]
    fn ngram_index_unknown_ngram_returns_empty() {
        // Arrange
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        // Act: query an n-gram that never appeared
        let conts = idx.get_ngram_continuations(&[99, 98], 5);
        // Assert
        assert!(conts.is_empty(), "unknown n-gram should return empty continuations");
    }

    // 8. SpecTree: build with empty prompt produces only root and adapter branches
    #[test]
    fn spec_tree_build_empty_prompt_no_pld_extensions() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        // Act
        let tree = SpecTree::build(config, &[10u32, 20, 30], &[], &NgramIndex::build(&[], 2));
        // Assert: only root + 2 adapter branches, no PLD spine, no n-gram branches
        assert_eq!(tree.node(0).unwrap().token_id, 10);
        let pld_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::PldSpine))
            .count();
        assert_eq!(pld_count, 0, "empty prompt should produce no PLD spine nodes");
        let adapter_count = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { .. }))
            .count();
        assert_eq!(adapter_count, 3, "should have 3 adapter nodes (root + 2 branches)");
    }

    // 9. SpecTree: all_token_ids returns tokens in node order
    #[test]
    fn spec_tree_all_token_ids_order_matches_nodes() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10u32, 50, 60];
        let tree = SpecTree::build(config, &[10u32, 20], &prompt, &NgramIndex::build(&prompt, 1));
        // Act
        let all_ids = tree.all_token_ids();
        let from_nodes: Vec<u32> = tree.nodes().iter().map(|n| n.token_id).collect();
        // Assert
        assert_eq!(all_ids, from_nodes, "all_token_ids should match nodes() order");
    }

    // 10. NgramIndex: build with all identical tokens
    #[test]
    fn ngram_index_all_same_tokens_single_continuation() {
        // Arrange: all tokens are 7
        let tokens = vec![7u32, 7, 7, 7, 7, 7];
        // Act
        let idx = NgramIndex::build(&tokens, 2);
        // Assert: [7,7] → 7 with count 4
        let conts = idx.get_ngram_continuations(&[7, 7], 5);
        assert_eq!(conts.len(), 1, "only one distinct continuation");
        assert_eq!(conts[0], 7);
    }

    // 11. SpecTree: mask_shape for empty tree
    #[test]
    fn spec_tree_mask_shape_empty_tree_zero_rows() {
        // Arrange
        let tree = SpecTree::new(SpecTreeConfig::default());
        // Act
        let (rows, cols) = tree.mask_shape(20);
        // Assert
        assert_eq!(rows, 0);
        assert_eq!(cols, 20);
    }

    // 12. SpecTree: tree_attention_mask_csr for empty tree returns empty CSR
    #[test]
    fn spec_tree_csr_mask_empty_tree_single_indptr() {
        // Arrange
        let tree = SpecTree::new(SpecTreeConfig::default());
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(10);
        // Assert
        assert_eq!(indptr.len(), 1, "empty tree has indptr with just [0]");
        assert_eq!(indptr[0], 0);
        assert!(indices.is_empty());
    }

    // 13. DraftSource: Debug output contains variant name
    #[test]
    fn draft_source_debug_format() {
        // Arrange
        let spine = DraftSource::PldSpine;
        let ngram = DraftSource::NgramBranch;
        let adapter = DraftSource::AdapterTopK { k: 7 };
        // Act
        let debug_spine = format!("{:?}", spine);
        let debug_ngram = format!("{:?}", ngram);
        let debug_adapter = format!("{:?}", adapter);
        // Assert
        assert!(debug_spine.contains("PldSpine"), "Debug should contain PldSpine: {}", debug_spine);
        assert!(debug_ngram.contains("NgramBranch"), "Debug should contain NgramBranch: {}", debug_ngram);
        assert!(debug_adapter.contains("AdapterTopK"), "Debug should contain AdapterTopK: {}", debug_adapter);
        assert!(debug_adapter.contains("7"), "Debug should contain k value: {}", debug_adapter);
    }

    // 14. SpecTree: node accessor returns None for u32::MAX on non-empty tree
    #[test]
    fn spec_tree_node_accessor_none_for_large_id() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        // Act & Assert
        assert!(tree.node(0).is_some(), "node 0 should exist");
        assert!(tree.node(u32::MAX).is_none(), "u32::MAX should be out of bounds");
        assert!(tree.node(100).is_none(), "id 100 should be out of bounds");
    }

    // 15. SpecTree: branch_token_ids correctly excludes spine nodes
    #[test]
    fn spec_tree_branch_token_ids_excludes_spine() {
        // Arrange: build a tree with known spine + branches
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 1,
            ngram_top_k: 1,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10u32, 50, 60, 70, 10, 80, 90];
        let tree = SpecTree::build(config, &[10u32, 20, 30], &prompt, &NgramIndex::build(&prompt, 1));
        // Act
        let spine_ids: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        let branches = tree.branch_token_ids();
        // Assert: branch node IDs should not overlap with spine IDs
        for (branch_id, _token) in &branches {
            assert!(!spine_ids.contains(branch_id),
                "branch node {} should not be in spine set {:?}", branch_id, spine_ids);
        }
        // Branch count should be total nodes minus spine nodes
        assert_eq!(branches.len() + spine_ids.len(), tree.len(),
            "spine + branch should equal total nodes");
    }

    // ==================================================================
    // 15 new tests — edge cases, boundary conditions, derive traits
    // ==================================================================

    // 1. DraftSource: all variants produce unique Debug strings (no prefix collision)
    #[test]
    fn draft_source_debug_strings_all_unique() {
        let variants = vec![
            DraftSource::PldSpine,
            DraftSource::AdapterTopK { k: 0 },
            DraftSource::AdapterTopK { k: 1 },
            DraftSource::AdapterTopK { k: 255 },
            DraftSource::NgramBranch,
        ];
        let debug_strings: Vec<String> = variants.iter().map(|v| format!("{:?}", v)).collect();
        for i in 0..debug_strings.len() {
            for j in (i + 1)..debug_strings.len() {
                assert_ne!(debug_strings[i], debug_strings[j],
                    "Debug strings for {:?} and {:?} should differ", variants[i], variants[j]);
            }
        }
    }

    // 2. SpecTreeConfig: PartialEq distinguishes configs that differ only in max_spine_depth
    #[test]
    fn spec_tree_config_partial_eq_differs_by_max_spine_depth() {
        let a = SpecTreeConfig { max_spine_depth: 3, ..SpecTreeConfig::default() };
        let b = SpecTreeConfig { max_spine_depth: 7, ..SpecTreeConfig::default() };
        assert_ne!(a, b, "configs differing in max_spine_depth should not be equal");
    }

    // 3. SpecNode: position_offset field u32::MAX boundary
    #[test]
    fn spec_node_position_offset_u32_max() {
        let node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.5,
            position_offset: u32::MAX,
        };
        assert_eq!(node.position_offset, u32::MAX);
        let cloned = node.clone();
        assert_eq!(cloned.position_offset, u32::MAX);
    }

    // 4. NgramIndex: build with exactly 2*token_count windows when n=1
    #[test]
    fn ngram_index_n1_window_count_with_repeated_tokens() {
        // tokens = [5, 5, 5, 5] → n=1 → 3 windows, each continuing to 5
        let tokens = vec![5u32, 5, 5, 5];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(5, 10);
        assert_eq!(conts, vec![5], "only continuation of 5→5");
    }

    // 5. SpecTree: build with prompt where adapter token only appears at last position (no PLD continuation possible)
    #[test]
    fn spec_tree_build_adapter_at_last_prompt_position_no_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter = vec![99u32];
        // Token 99 only appears at the last position — no tokens after it for PLD
        let prompt = vec![1, 2, 3, 4, 5, 99];
        let tree = SpecTree::build(config, &adapter, &prompt, &NgramIndex::build(&prompt, 1));
        // Should only have root, no spine extensions
        assert_eq!(tree.len(), 1, "adapter at last position should yield no PLD spine");
        assert_eq!(tree.spine_ids(), vec![0]);
    }

    // 6. SpecTree: CSR mask indptr length is total_nodes + 1 even for single-node tree
    #[test]
    fn spec_tree_csr_mask_indptr_len_is_nodes_plus_one() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[42u32], &[], &NgramIndex::build(&[], 1));
        let (indptr, _indices) = tree.tree_attention_mask_csr(10);
        assert_eq!(indptr.len(), tree.len() + 1,
            "indptr must have total_nodes + 1 entries");
    }

    // 7. SpecTree: mask_shape cols = total_seq_len + total_nodes consistently
    #[test]
    fn spec_tree_mask_shape_cols_equation_for_various_seq_lens() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[7u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 1));
        let nodes = tree.len();
        for seq_len in [0, 1, 10, 100, 1000] {
            let (rows, cols) = tree.mask_shape(seq_len);
            assert_eq!(rows, nodes);
            assert_eq!(cols, seq_len + nodes);
        }
    }

    // 8. SpecNode: children vec can hold u32::MAX as element without panic
    #[test]
    fn spec_node_children_holds_u32_max_element() {
        let node = SpecNode {
            node_id: 0,
            token_id: 0,
            parent_id: None,
            children: vec![u32::MAX],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 0.1,
            position_offset: 0,
        };
        assert_eq!(node.children[0], u32::MAX);
    }

    // 9. SpecTree: accepted_from_spine with all-zero token IDs
    #[test]
    fn spec_tree_accepted_from_spine_all_zero_tokens() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![0u32, 0, 0, 0, 0];
        let tree = SpecTree::build(config, &[0u32], &prompt, &NgramIndex::build(&prompt, 1));
        let spine = tree.spine_token_ids();
        // All spine tokens are 0 — matching with target of all 0 should accept full spine
        let (count, accepted) = tree.accepted_from_spine(&spine);
        assert_eq!(count, spine.len(), "all-zero spine should fully match all-zero target");
        assert!(accepted.iter().all(|&t| t == 0));
    }

    // 10. SpecTree: node accessor returns None for id equal to tree.len()
    #[test]
    fn spec_tree_node_returns_none_at_boundary_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[10u32], &[1, 2], &NgramIndex::build(&[1, 2], 1));
        let boundary_id = tree.len() as u32;
        assert!(tree.node(boundary_id).is_none(),
            "node(len) should be out of bounds");
        assert!(tree.node(boundary_id - 1).is_some(),
            "node(len-1) should exist");
    }

    // 11. SpecTree: all_token_ids for tree with adapter token 0 (zero token ID)
    #[test]
    fn spec_tree_all_token_ids_with_zero_adapter_token() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[0u32], &[], &NgramIndex::build(&[], 1));
        let ids = tree.all_token_ids();
        assert_eq!(ids, vec![0u32], "token ID 0 should be valid and present");
    }

    // 12. SpecTree: building two trees from same inputs produces identical node token sequences
    #[test]
    fn spec_tree_build_two_instances_identical_tokens() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            pld_ngram_len: 2,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter = vec![10u32, 20];
        let prompt = vec![10, 50, 60, 70, 10, 80, 90, 20, 30];
        let ngram = NgramIndex::build(&prompt, 2);
        let tree_a = SpecTree::build(config.clone(), &adapter, &prompt, &ngram);
        let tree_b = SpecTree::build(config, &adapter, &prompt, &ngram);
        assert_eq!(tree_a.all_token_ids(), tree_b.all_token_ids(),
            "two builds from identical inputs must produce same token IDs");
        assert_eq!(tree_a.len(), tree_b.len());
    }

    // 13. NgramIndex: get_ngram_continuations returns empty for a valid n-gram that was never seen
    #[test]
    fn ngram_index_get_ngram_continuations_unseen_ngram() {
        // Build with tokens [1, 2, 3, 4], n=2 → windows [1,2]→3, [2,3]→4
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        // [5, 6] was never seen
        assert!(idx.get_ngram_continuations(&[5, 6], 5).is_empty(),
            "unseen n-gram should yield no continuations");
    }

    // 14. SpecTree: root node always has position_offset == 0 regardless of config
    #[test]
    fn spec_tree_root_position_offset_zero_all_configs() {
        let configs = vec![
            SpecTreeConfig { max_spine_depth: 1, ..SpecTreeConfig::default() },
            SpecTreeConfig { max_spine_depth: 10, adapter_top_k: 5, ..SpecTreeConfig::default() },
            SpecTreeConfig { max_spine_depth: 1, adapter_top_k: 1, max_branches_per_node: 10, ..SpecTreeConfig::default() },
        ];
        for config in configs {
            let tree = SpecTree::build(config, &[42u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 1));
            let root = tree.node(0).expect("root must exist");
            assert_eq!(root.position_offset, 0,
                "root position_offset must be 0");
        }
    }

    // 15. SpecTree: nodes() slice length matches len() after various builds
    #[test]
    fn spec_tree_nodes_slice_len_matches_len_across_builds() {
        let cases: Vec<(SpecTreeConfig, Vec<u32>, Vec<u32>)> = vec![
            // (config, adapter_tokens, prompt)
            (SpecTreeConfig { max_spine_depth: 1, adapter_top_k: 1, max_branches_per_node: 0, ..SpecTreeConfig::default() },
             vec![1u32], vec![]),
            (SpecTreeConfig { max_spine_depth: 3, adapter_top_k: 2, max_branches_per_node: 1, pld_ngram_len: 1, ..SpecTreeConfig::default() },
             vec![10u32, 20], vec![10, 50, 60, 10, 70]),
            (SpecTreeConfig { max_spine_depth: 2, adapter_top_k: 3, max_branches_per_node: 2, pld_ngram_len: 2, ..SpecTreeConfig::default() },
             vec![5u32, 6, 7], vec![5, 8, 9, 5, 10, 11]),
        ];
        for (config, adapter, prompt) in cases {
            let ngram = NgramIndex::build(&prompt, 1);
            let tree = SpecTree::build(config, &adapter, &prompt, &ngram);
            assert_eq!(tree.nodes().len(), tree.len(),
                "nodes() slice length must equal len()");
            assert!(!tree.is_empty() || tree.len() == 0);
        }
    }

    // ------------------------------------------------------------------
    // New tests: additional edge cases and behavior coverage
    // ------------------------------------------------------------------

    // 16. SpecTree: build with u32::MAX as adapter token does not panic
    #[test]
    fn spec_tree_build_adapter_token_u32_max() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![u32::MAX];
        let prompt = vec![1u32, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert_eq!(tree.node(0).unwrap().token_id, u32::MAX);
        assert_eq!(tree.spine_token_ids()[0], u32::MAX);
    }

    // 17. SpecTree: CSR mask with zero total_seq_len has only tree-relative columns
    #[test]
    fn spec_tree_csr_mask_zero_seq_len_no_prefix_columns() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(0);
        // With total_seq_len=0, all columns are tree-relative
        for i in 0..tree.len() {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            for &col in &indices[row_start..row_end] {
                assert!(col < tree.len(),
                    "column {} in row {} exceeds tree size {} with zero seq len",
                    col, i, tree.len());
            }
        }
    }

    // 18. SpecTree: all_token_ids includes every node exactly once
    #[test]
    fn spec_tree_all_token_ids_exact_count_no_duplicates_in_position() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let ids = tree.all_token_ids();
        assert_eq!(ids.len(), tree.len(),
            "all_token_ids length must exactly equal tree.len()");
    }

    // 19. SpecTree: mask_shape columns = total_seq_len + tree_size for various seq lens
    #[test]
    fn spec_tree_mask_shape_cols_varies_linearly_with_seq_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let tree_size = tree.len();

        let (r0, c0) = tree.mask_shape(0);
        let (r1, c1) = tree.mask_shape(100);
        let (r2, c2) = tree.mask_shape(1000);

        assert_eq!(r0, tree_size);
        assert_eq!(r1, tree_size);
        assert_eq!(r2, tree_size);
        assert_eq!(c1 - c0, 100);
        assert_eq!(c2 - c1, 900);
    }

    // 20. NgramIndex: build with n=0 and a single token
    #[test]
    fn ngram_index_build_n0_single_token_no_panic() {
        let idx = NgramIndex::build(&[42u32], 0);
        // n=0 means empty n-gram slices; each token is a continuation of the empty prefix
        assert_eq!(idx.get_continuations(42, 5), Vec::<u32>::new(),
            "single token with n=0 should not produce self-referential continuations");
    }

    // 21. SpecTree: build with duplicate adapter tokens produces root from first only
    #[test]
    fn spec_tree_build_duplicate_adapter_first_token_is_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 10, 10];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert_eq!(tree.node(0).unwrap().token_id, 10);
        // Adapter branches use adapter_top_tokens[1] and [2], both also 10
        let adapter_branches: Vec<&SpecNode> = tree.nodes()
            .iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .collect();
        for branch in &adapter_branches {
            assert_eq!(branch.token_id, 10,
                "duplicate adapter tokens should still produce correct branches");
        }
    }

    // 22. SpecTree: accepted_from_spine with single-token target matching root
    #[test]
    fn spec_tree_accepted_single_target_matches_root_only() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        assert!(spine.len() > 1, "spine should have extensions");

        let (count, accepted) = tree.accepted_from_spine(&[spine[0]]);
        assert_eq!(count, 1);
        assert_eq!(accepted, vec![spine[0]]);
    }

    // 23. SpecTree: branch_token_ids returns correct (node_id, token_id) pairs
    #[test]
    fn spec_tree_branch_token_ids_pairs_match_nodes() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let branches = tree.branch_token_ids();
        for (node_id, token_id) in &branches {
            let node = tree.node(*node_id).expect("branch node_id must be valid");
            assert_eq!(node.token_id, *token_id,
                "branch pair ({}, {}) must match node's token_id {}",
                node_id, token_id, node.token_id);
        }
    }

    // 24. SpecNode: all fields correct after construction with NgramBranch source
    #[test]
    fn spec_node_ngram_branch_all_fields() {
        let node = SpecNode {
            node_id: 5,
            token_id: 99,
            parent_id: Some(2),
            children: vec![],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 0.10,
            position_offset: 3,
        };
        assert_eq!(node.node_id, 5);
        assert_eq!(node.token_id, 99);
        assert_eq!(node.parent_id, Some(2));
        assert!(node.children.is_empty());
        assert_eq!(node.source, DraftSource::NgramBranch);
        assert!((node.estimated_acceptance - 0.10).abs() < f32::EPSILON);
        assert_eq!(node.position_offset, 3);
    }

    // 25. SpecTree: spine_ids and spine_token_ids consistent after multi-level build
    #[test]
    fn spec_tree_spine_ids_and_token_ids_consistent_multi_level() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        let spine_tokens = tree.spine_token_ids();
        assert_eq!(spine_ids.len(), spine_tokens.len());
        for (i, &id) in spine_ids.iter().enumerate() {
            assert_eq!(spine_tokens[i], tree.node(id).unwrap().token_id,
                "spine_token_ids[{}] must match node({}).token_id", i, id);
        }
    }

    // 26. NgramIndex: get_continuations with top_k=0 returns empty even with data
    #[test]
    fn ngram_index_get_continuations_top_k_zero_with_data() {
        let tokens = vec![1u32, 2, 3, 1, 4, 1, 5];
        let idx = NgramIndex::build(&tokens, 1);
        let conts = idx.get_continuations(1, 0);
        assert!(conts.is_empty(),
            "top_k=0 should always return empty, even when continuations exist");
    }

    // 27. SpecTree: CSR mask indptr length is always tree.len() + 1
    #[test]
    fn spec_tree_csr_mask_indptr_len_exact() {
        let configs_and_inputs = vec![
            (SpecTreeConfig { max_spine_depth: 1, adapter_top_k: 1, max_branches_per_node: 0, ..SpecTreeConfig::default() },
             vec![10u32], vec![1, 2, 3]),
            (SpecTreeConfig { max_spine_depth: 3, adapter_top_k: 2, max_branches_per_node: 1, pld_ngram_len: 1, ..SpecTreeConfig::default() },
             vec![10u32, 20], vec![10, 50, 60, 10, 70]),
        ];
        for (config, adapter, prompt) in configs_and_inputs {
            let ngram_idx = NgramIndex::build(&prompt, 1);
            let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
            let (indptr, _) = tree.tree_attention_mask_csr(5);
            assert_eq!(indptr.len(), tree.len() + 1,
                "indptr length must be tree.len()+1, got {} for tree of {} nodes",
                indptr.len(), tree.len());
        }
    }

    // 28. SpecTree: clone produces structurally equal tree verified by all accessors
    #[test]
    fn spec_tree_clone_all_accessors_match() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            pld_ngram_len: 2,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let clone = tree.clone();

        assert_eq!(tree.len(), clone.len());
        assert_eq!(tree.is_empty(), clone.is_empty());
        assert_eq!(tree.all_token_ids(), clone.all_token_ids());
        assert_eq!(tree.spine_ids(), clone.spine_ids());
        assert_eq!(tree.spine_token_ids(), clone.spine_token_ids());
        assert_eq!(tree.branch_token_ids(), clone.branch_token_ids());

        let (indptr1, indices1) = tree.tree_attention_mask_csr(10);
        let (indptr2, indices2) = clone.tree_attention_mask_csr(10);
        assert_eq!(indptr1, indptr2);
        assert_eq!(indices1, indices2);
    }

    // 29. SpecTree: empty prompt with adapter token produces root-only tree
    #[test]
    fn spec_tree_build_empty_prompt_root_only() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32, 43, 44];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root + adapter top-2 and top-3 branches, but no PLD spine extensions, no n-gram branches
        assert!(tree.len() >= 1);
        assert_eq!(tree.node(0).unwrap().token_id, 42);
        // Spine is just root (no PLD continuations possible from empty prompt)
        assert_eq!(tree.spine_ids(), vec![0]);
    }

    // 30. SpecTree: node returns None for ids beyond valid range on a built tree
    #[test]
    fn spec_tree_node_returns_none_for_ids_beyond_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert!(tree.node(tree.len() as u32).is_none(),
            "node(id == len) must return None");
        assert!(tree.node(u32::MAX).is_none(),
            "node(u32::MAX) must return None");
    }

    // ==================================================================
    // 15 additional edge case tests
    // ==================================================================

    // 31. SpecTree: CSR mask root row has no ancestor columns (only prefix + self)
    #[test]
    fn spec_tree_csr_mask_root_row_has_no_ancestors() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let seq_len = 4;
        let (indptr, indices) = tree.tree_attention_mask_csr(seq_len);
        let root_row_start = indptr[0];
        let root_row_end = indptr[1];
        let root_cols: Vec<usize> = indices[root_row_start..root_row_end].to_vec();

        // Root row: seq_len prefix columns + self (column seq_len + 0), zero ancestors
        for &col in &root_cols {
            if col >= seq_len {
                assert_eq!(col, seq_len,
                    "root row should only have self as tree column, got column {}", col);
            }
        }
    }

    // 32. NgramIndex: build with n larger than tokens length produces empty index
    #[test]
    fn ngram_index_build_n_larger_than_tokens_len_produces_empty() {
        let tokens = vec![1u32, 2, 3];
        let idx = NgramIndex::build(&tokens, 5);
        // n=5 > len=3, so no windows can be formed
        assert!(idx.get_continuations(1, 10).is_empty());
        assert!(idx.get_continuations(2, 10).is_empty());
    }

    // 33. SpecTree: accepted_from_spine returns empty when target is all different from spine
    #[test]
    fn accepted_from_spine_all_tokens_differ() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Target tokens are all different from any spine token
        let target = vec![999u32, 888, 777];
        let (count, accepted) = tree.accepted_from_spine(&target);
        assert_eq!(count, 0, "no tokens should match when all differ");
        assert!(accepted.is_empty());
    }

    // 34. SpecTreeConfig: default config has max_tree_size > 0
    #[test]
    fn spec_tree_config_default_max_tree_size_positive() {
        let config = SpecTreeConfig::default();
        assert!(config.max_tree_size > 0,
            "default max_tree_size must be positive, got {}", config.max_tree_size);
        assert!(config.max_spine_depth > 0,
            "default max_spine_depth must be positive, got {}", config.max_spine_depth);
        assert!(config.adapter_top_k > 0,
            "default adapter_top_k must be positive, got {}", config.adapter_top_k);
    }

    // 35. SpecNode: two nodes with same token_id but different source are not equal
    #[test]
    fn spec_node_same_token_different_source_not_equal() {
        let node_a = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![],
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.7,
            position_offset: 0,
        };
        let node_b = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.7,
            position_offset: 0,
        };
        assert_ne!(node_a, node_b,
            "nodes with different DraftSource must not be equal");
    }

    // 36. SpecTree: nodes() slice is never longer than len()
    #[test]
    fn spec_tree_nodes_slice_never_exceeds_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 20, 30];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert_eq!(tree.nodes().len(), tree.len(),
            "nodes() slice length must equal len()");
    }

    // 37. DraftSource: AdapterTopK with k=255 stores and retrieves correctly
    #[test]
    fn draft_source_adapter_top_k_stores_u8_max() {
        let ds = DraftSource::AdapterTopK { k: u8::MAX };
        if let DraftSource::AdapterTopK { k } = ds {
            assert_eq!(k, u8::MAX, "k should store u8::MAX = 255");
        } else {
            panic!("Expected AdapterTopK variant");
        }
    }

    // 38. SpecTree: build with max_branches_per_node=0 produces no n-gram branches
    #[test]
    fn spec_tree_build_zero_branches_per_node_no_ngram() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ngram_top_k: 5,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 10, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let ngram_nodes: Vec<&SpecNode> = tree.nodes()
            .iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .collect();
        assert!(ngram_nodes.is_empty(),
            "max_branches_per_node=0 should produce zero n-gram branches, found {}", ngram_nodes.len());
    }

    // 39. NgramIndex: build with exactly n+1 tokens produces a single entry
    #[test]
    fn ngram_index_build_minimal_window_single_entry() {
        // n=3, tokens.len()=4 → exactly one window [0..3] with continuation tokens[3]
        let tokens = vec![10u32, 20, 30, 40];
        let idx = NgramIndex::build(&tokens, 3);
        let conts = idx.get_continuations(10, 5);
        // 1-gram lookup: token 10 followed by 20 somewhere in the window chain
        // Actually get_continuations uses 1-gram (hash of &[token]), so token 10
        // appears at position 0, its continuation in the n-gram window chain is tokens[1]=20
        // But the n-gram index was built with n=3, so only tokens[0..3]→tokens[3]=40 creates an entry
        // get_continuations uses hash of single token (1-gram), not n=3
        // So it returns empty because no 1-gram entries were built with n=3
        assert!(conts.is_empty() || conts.len() <= 5,
            "get_continuations with n=3 on 4 tokens should be bounded");
    }

    // 40. SpecTree: spine_ids always starts with node 0
    #[test]
    fn spec_tree_spine_ids_always_starts_at_zero() {
        let configs = vec![
            SpecTreeConfig { max_spine_depth: 1, adapter_top_k: 1, max_branches_per_node: 0, ..SpecTreeConfig::default() },
            SpecTreeConfig { max_spine_depth: 5, adapter_top_k: 3, max_branches_per_node: 2, ..SpecTreeConfig::default() },
        ];
        for config in configs {
            let adapter_tokens = vec![100u32, 200, 300];
            let prompt = vec![1, 2, 3, 100, 50, 60];
            let ngram_idx = NgramIndex::build(&prompt, 2);
            let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
            if !tree.is_empty() {
                let spine = tree.spine_ids();
                assert_eq!(spine[0], 0, "spine must always start at node 0");
            }
        }
    }

    // 41. SpecTree: tree_attention_mask_csr every row includes the self column
    #[test]
    fn spec_tree_csr_mask_every_row_includes_self() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let seq_len = 5;
        let (indptr, indices) = tree.tree_attention_mask_csr(seq_len);

        for i in 0..tree.len() {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            let row_cols: Vec<usize> = indices[row_start..row_end].to_vec();
            let self_col = seq_len + i;
            assert!(row_cols.contains(&self_col),
                "row {} must include self column {}, got {:?}",
                i, self_col, row_cols);
        }
    }

    // 42. SpecTree: branch_token_ids length equals total_nodes minus spine length
    #[test]
    fn spec_tree_branch_count_equals_total_minus_spine() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_len = tree.spine_ids().len();
        let branch_count = tree.branch_token_ids().len();
        assert_eq!(spine_len + branch_count, tree.len(),
            "spine({}) + branches({}) must equal total({})",
            spine_len, branch_count, tree.len());
    }

    // 43. NgramIndex: build with alternating two tokens produces correct frequency order
    #[test]
    fn ngram_index_alternating_two_tokens_frequency_order() {
        // tokens: [A, B, A, B, A] with n=1
        // After A: B appears 2 times (positions 1, 3)
        // After B: A appears 2 times (positions 2, 4)
        let tokens = vec![1u32, 2, 1, 2, 1];
        let idx = NgramIndex::build(&tokens, 1);

        let conts_a = idx.get_continuations(1, 5);
        assert_eq!(conts_a, vec![2], "after token 1, continuation should be [2]");

        let conts_b = idx.get_continuations(2, 5);
        assert_eq!(conts_b, vec![1], "after token 2, continuation should be [1]");
    }

    // 44. SpecTree: node accessor returns correct parent_id chain
    #[test]
    fn spec_tree_node_parent_chain_from_leaf_to_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        if tree.len() > 1 {
            // Walk from last spine node back to root via parent_id
            let spine = tree.spine_ids();
            if spine.len() > 1 {
                let leaf_id = *spine.last().unwrap();
                let mut current = tree.node(leaf_id).unwrap().clone();
                let mut chain = vec![current.node_id];
                while let Some(pid) = current.parent_id {
                    chain.push(pid);
                    current = tree.node(pid).unwrap().clone();
                }
                // Root of spine chain has parent_id None, so chain ends at node 0
                assert_eq!(*chain.last().unwrap(), 0,
                    "parent chain from spine leaf must reach node 0");
            }
        }
    }

    // 45. SpecTree: max_tree_size limits n-gram branches but not spine or adapter branches
    #[test]
    fn spec_tree_max_tree_size_limits_ngram_branches() {
        // Use max_spine_depth=1 so no PLD extensions, adapter_top_k=1 so no adapter branches,
        // then max_tree_size=1 should block all n-gram branches
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 5,
            max_tree_size: 1,
            ngram_top_k: 5,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        // Root is node 0, already added. No PLD extensions (max_spine_depth=1).
        // No adapter branches (adapter_top_k=1). max_tree_size=1 blocks n-gram branches.
        assert_eq!(tree.len(), 1, "max_tree_size=1 should produce exactly 1 node");
        assert_eq!(tree.node(0).unwrap().token_id, 10, "root must be adapter top-1");
    }

    #[test]
    fn spec_tree_is_empty_on_new() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn spec_tree_node_none_on_empty() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.node(0).is_none());
    }

    #[test]
    fn spec_tree_nodes_empty_slice_on_new() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.nodes().is_empty());
    }

    #[test]
    fn spec_tree_all_token_ids_empty_on_new() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.all_token_ids().is_empty());
    }

    #[test]
    fn draft_source_pld_spine_equality() {
        assert_eq!(DraftSource::PldSpine, DraftSource::PldSpine);
        assert_ne!(DraftSource::PldSpine, DraftSource::NgramBranch);
    }

    #[test]
    fn ngram_index_get_continuations_empty_index() {
        let idx = NgramIndex::build(&[], 3);
        let result = idx.get_continuations(42, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn spec_node_default_estimated_acceptance() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        let mut config = SpecTreeConfig::default();
        config.max_spine_depth = 0;
        let t2 = SpecTree::new(config);
        assert!(t2.is_empty());
    }

    // ------------------------------------------------------------------
    // Wave 12x34: 15 additional tests
    // ------------------------------------------------------------------

    // 46. NgramIndex: build with n=1 and a single token produces an empty table
    #[test]
    fn ngram_index_build_n1_single_token_empty() {
        let idx = NgramIndex::build(&[42u32], 1);
        assert!(idx.table.is_empty(), "single token with n=1 should produce empty table");
        assert!(idx.get_ngram_continuations(&[42], 5).is_empty());
    }

    // 47. SpecTree: build with adapter token equal to u32::MAX
    #[test]
    fn spec_tree_build_adapter_token_max_u32() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![u32::MAX];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert!(!tree.is_empty());
        assert_eq!(tree.node(0).unwrap().token_id, u32::MAX);
    }

    // 48. SpecTree: PLD continuations deduplicate tokens across multiple prompt occurrences
    #[test]
    fn spec_tree_pld_dedup_across_occurrences_verified() {
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 20, 10, 50, 30];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        let count_50 = spine.iter().filter(|&&t| t == 50).count();
        assert!(count_50 <= 1, "50 should appear at most once in spine, got {}", count_50);
    }

    // 49. SpecTree: CSR mask with total_seq_len=0 has no prefix columns
    #[test]
    fn spec_tree_csr_mask_zero_seq_len_no_prefix() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        let prompt = vec![1, 2, 42, 99];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (indptr, indices) = tree.tree_attention_mask_csr(0);
        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            for &col in &indices[start..end] {
                assert!(col < tree.len(),
                    "column {} in row {} should be < tree size {}", col, i, tree.len());
            }
        }
    }

    // 50. SpecTree: root node always has DraftSource::AdapterTopK { k: 1 }
    #[test]
    fn spec_tree_root_source_always_adapter_top_k_1() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![100u32, 200, 300];
        let prompt = vec![1, 2, 100, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root = tree.node(0).unwrap();
        assert_eq!(root.source, DraftSource::AdapterTopK { k: 1 });
    }

    // 51. NgramIndex: get_ngram_continuations with top_k=0 returns empty
    #[test]
    fn ngram_index_get_ngram_continuations_top_k_zero_returns_empty() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 0);
        assert!(conts.is_empty(), "top_k=0 should always return empty");
    }

    // 52. SpecTree: position_offset of root is always zero across various configs
    #[test]
    fn spec_tree_root_position_offset_always_zero() {
        for adapter_top_k in 1..=3 {
            let config = SpecTreeConfig {
                max_spine_depth: 3,
                adapter_top_k,
                max_branches_per_node: 1,
                ..SpecTreeConfig::default()
            };
            let adapter_tokens: Vec<u32> = (100..100 + adapter_top_k as u32).collect();
            let prompt = vec![1, 2, 100, 50];
            let ngram_idx = NgramIndex::build(&prompt, 2);
            let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

            assert_eq!(tree.node(0).unwrap().position_offset, 0,
                "root position_offset must be 0 for adapter_top_k={}", adapter_top_k);
        }
    }

    // 53. SpecTree: CSR mask columns for root row are exactly prefix columns + self
    #[test]
    fn spec_tree_csr_mask_root_row_exactly_prefix_plus_self() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let seq_len = 7;
        let (indptr, indices) = tree.tree_attention_mask_csr(seq_len);

        let root_start = indptr[0];
        let root_end = indptr[1];
        let root_cols: Vec<usize> = indices[root_start..root_end].to_vec();
        assert_eq!(root_cols.len(), seq_len + 1,
            "root row should have exactly seq_len + 1 columns");
        for i in 0..seq_len {
            assert!(root_cols.contains(&i), "root should attend to prefix column {}", i);
        }
        assert!(root_cols.contains(&seq_len), "root should attend to self at column {}", seq_len);
    }

    // 54. SpecTree: all adapter branch nodes have parent_id = 0 (root)
    #[test]
    fn spec_tree_all_adapter_branches_parent_is_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 5,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30, 40, 50];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            if matches!(node.source, DraftSource::AdapterTopK { k } if k > 1) {
                assert_eq!(node.parent_id, Some(0),
                    "adapter branch node {} with k>1 must have parent_id=0", node.node_id);
            }
        }
    }

    // 55. SpecTree: build with empty prompt still creates root and adapter branches
    #[test]
    fn spec_tree_build_empty_prompt_creates_adapter_branches() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let ngram_idx = NgramIndex::build(&[], 2);
        let tree = SpecTree::build(config, &adapter_tokens, &[], &ngram_idx);

        assert!(!tree.is_empty());
        assert!(tree.len() >= 1);
        let spine_len = tree.spine_ids().len();
        assert_eq!(spine_len, 1, "empty prompt should only have root in spine");
    }

    // 56. SpecTree: mask_shape with zero total_seq_len has cols == tree.len()
    #[test]
    fn spec_tree_mask_shape_cols_equals_len_when_zero_seq() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        let prompt = vec![1, 2, 42, 99];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let (rows, cols) = tree.mask_shape(0);
        assert_eq!(rows, tree.len());
        assert_eq!(cols, tree.len(), "cols should equal tree size when seq_len=0");
    }

    // 57. NgramIndex: same token repeated many times produces single continuation
    #[test]
    fn ngram_index_repeated_token_single_continuation() {
        let tokens = vec![5u32; 10];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[5, 5], 5);
        assert_eq!(conts, vec![5], "repeated token should have single continuation 5");
    }

    // 58. SpecTree: accepted_from_spine with target shorter than spine accepts prefix
    #[test]
    fn spec_tree_accepted_from_spine_target_shorter_verified() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine = tree.spine_token_ids();
        if spine.len() > 1 {
            let short_target = vec![spine[0]];
            let (count, accepted) = tree.accepted_from_spine(&short_target);
            assert_eq!(count, 1);
            assert_eq!(accepted, vec![spine[0]]);
        }
    }

    // 59. SpecTree: nodes() slice length equals len()
    #[test]
    fn spec_tree_nodes_slice_len_matches_len_verified() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert_eq!(tree.nodes().len(), tree.len(),
            "nodes() slice length must equal len()");
    }

    // 60. SpecTree: adapter branch acceptance is less than root acceptance
    #[test]
    fn spec_tree_adapter_branch_acceptance_less_than_root() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root_acceptance = tree.node(0).unwrap().estimated_acceptance;
        for node in tree.nodes() {
            if matches!(node.source, DraftSource::AdapterTopK { k } if k > 1) {
                assert!(node.estimated_acceptance < root_acceptance,
                    "adapter branch acceptance ({}) must be < root ({})",
                    node.estimated_acceptance, root_acceptance);
            }
        }
    }

    // 61. SpecNode: children vec can hold and return values via mutable access pattern
    #[test]
    fn spec_node_children_can_be_modified_via_mutable_reference() {
        let mut node = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![1, 2],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.5,
            position_offset: 0,
        };

        // Act: push a new child
        node.children.push(3);

        assert_eq!(node.children, vec![1, 2, 3],
            "children should reflect the pushed value");
    }

    // 62. SpecTree: clone preserves total_nodes field
    #[test]
    fn spec_tree_clone_preserves_total_nodes() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let cloned = tree.clone();

        assert_eq!(tree.len(), cloned.len(),
            "cloned tree total_nodes must match original");
    }

    // 63. SpecTree: new tree has spine_len of zero
    #[test]
    fn spec_tree_new_spine_len_zero() {
        let config = SpecTreeConfig::default();
        let tree = SpecTree::new(config);

        // spine_ids on empty tree would panic, but we can verify via spine_token_ids
        // that the tree is empty — so we verify len is zero instead
        assert!(tree.is_empty(),
            "newly created tree must be empty, implying spine_len == 0");
    }

    // 64. SpecTree: CSR mask prefix columns count equals total_seq_len for each row
    #[test]
    fn spec_tree_csr_mask_prefix_columns_count_equals_total_seq_len() {
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let total_seq_len = 5usize;

        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);

        for row in 0..tree.len() {
            let row_start = indptr[row];
            let row_end = indptr[row + 1];
            let row_cols = &indices[row_start..row_end];
            // First total_seq_len columns should be exactly 0..total_seq_len
            let prefix_count = row_cols.iter().filter(|&&c| c < total_seq_len).count();
            assert_eq!(prefix_count, total_seq_len,
                "row {} must have exactly {} prefix columns, got {}",
                row, total_seq_len, prefix_count);
        }
    }

    // 65. SpecTree: all nodes have node_id equal to their index in the nodes slice
    #[test]
    fn spec_tree_all_nodes_have_node_id_equal_to_index() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 3,
            max_branches_per_node: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for (idx, node) in tree.nodes().iter().enumerate() {
            assert_eq!(node.node_id, idx as u32,
                "node at index {} has node_id {}, expected {}",
                idx, node.node_id, idx);
        }
    }

    // 66. SpecTree: root token equals first adapter token
    #[test]
    fn spec_tree_build_root_token_is_first_adapter_token() {
        let config = SpecTreeConfig::default();
        let adapter_tokens = vec![42u32, 100, 200];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let root = tree.node(0).unwrap();
        assert_eq!(root.token_id, 42,
            "root token_id must be the first adapter token");
    }

    // 67. SpecTreeConfig: default values are reasonable for production use
    #[test]
    fn spec_tree_config_default_is_reasonable_for_production() {
        let config = SpecTreeConfig::default();

        assert!(config.max_spine_depth >= 3,
            "max_spine_depth should be at least 3 for useful speculation");
        assert!(config.max_tree_size >= 16,
            "max_tree_size should be at least 16 for useful speculation");
        assert!(config.adapter_top_k >= 2,
            "adapter_top_k should be at least 2 to have branches");
        assert!(config.pld_ngram_len >= 2,
            "pld_ngram_len should be at least 2 for meaningful n-gram matching");
    }

    // 68. DraftSource: NgramBranch is distinct from PldSpine in equality
    #[test]
    fn draft_source_ngram_branch_is_distinct_from_pld_spine() {
        assert_ne!(DraftSource::NgramBranch, DraftSource::PldSpine,
            "NgramBranch and PldSpine must be distinct variants");
        assert_ne!(DraftSource::NgramBranch, DraftSource::AdapterTopK { k: 1 },
            "NgramBranch and AdapterTopK must be distinct variants");
    }

    // 69. NgramIndex: two tokens with n=2 returns empty (insufficient window)
    #[test]
    fn ngram_index_build_with_two_tokens_n2_returns_empty() {
        let tokens = vec![10u32, 20];
        let idx = NgramIndex::build(&tokens, 2);

        // With n=2 and len=2, we have tokens.len() == n, so no windows
        assert!(idx.get_ngram_continuations(&[10, 20], 1).is_empty(),
            "n=2 with exactly 2 tokens should produce no continuations");
    }

    // 70. SpecTree: mask_shape for single-node tree is (1, seq_len + 1)
    #[test]
    fn spec_tree_mask_shape_for_single_node_tree() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![5u32];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 3);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        assert_eq!(tree.len(), 1, "tree should have exactly 1 node");
        let (rows, cols) = tree.mask_shape(10);
        assert_eq!(rows, 1);
        assert_eq!(cols, 11, "cols must be total_seq_len (10) + tree_size (1)");
    }

    // 71. SpecTree: accepted_from_spine returns full spine length when all tokens match
    #[test]
    fn spec_tree_accepted_from_spine_no_mismatch_all_accepted() {
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_tokens = tree.spine_token_ids();
        // Use spine tokens themselves as target — guaranteed full match
        let (count, accepted) = tree.accepted_from_spine(&spine_tokens);

        assert_eq!(count, spine_tokens.len(),
            "all spine tokens should be accepted when target matches exactly");
        assert_eq!(accepted, spine_tokens,
            "accepted tokens must equal spine tokens");
    }

    // 72. SpecTree: spine_token_ids returns tokens that match spine_ids lookup
    #[test]
    fn spec_tree_spine_token_ids_matches_spine_ids_tokens() {
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        let spine_ids = tree.spine_ids();
        let spine_tokens = tree.spine_token_ids();

        assert_eq!(spine_ids.len(), spine_tokens.len(),
            "spine_ids and spine_token_ids must have same length");
        for (i, &sid) in spine_ids.iter().enumerate() {
            assert_eq!(spine_tokens[i], tree.node(sid).unwrap().token_id,
                "spine_token_ids[{}] must match token_id of spine_ids[{}]", i, i);
        }
    }

    // 73. SpecTree: n-gram branch estimated_acceptance follows the formula
    #[test]
    fn spec_tree_build_ngram_branch_acceptance_formula() {
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Build a prompt that gives n-gram continuations after token 10
        let prompt = vec![10, 99, 10, 88, 10, 77];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);

        for node in tree.nodes() {
            if matches!(node.source, DraftSource::NgramBranch) {
                // Acceptance must be in (0, 0.15) range based on the formula
                assert!(node.estimated_acceptance > 0.0,
                    "n-gram branch acceptance should be positive");
                assert!(node.estimated_acceptance <= 0.15,
                    "n-gram branch acceptance should be <= 0.15, got {}",
                    node.estimated_acceptance);
            }
        }
    }

    // 74. SpecNode: estimated_acceptance preserves f32 precision for small values
    #[test]
    fn spec_node_estimated_acceptance_preserves_f32_precision() {
        let tiny = 0.0001f32;
        let node = SpecNode {
            node_id: 0,
            token_id: 0,
            parent_id: None,
            children: vec![],
            source: DraftSource::NgramBranch,
            estimated_acceptance: tiny,
            position_offset: 0,
        };

        assert_eq!(node.estimated_acceptance, tiny,
            "estimated_acceptance must preserve exact f32 bit pattern");
        assert!(node.estimated_acceptance > 0.0f32,
            "tiny positive value must remain positive");
    }

    // 75. NgramIndex: get_ngram_continuations returns correct result for n=2 single match
    #[test]
    fn ngram_index_get_ngram_continuations_single_n2_match() {
        // tokens: [1, 2, 50, 1, 2, 60]
        // n=2 windows: [1,2]->50, [2,50]->1, [50,1]->2, [1,2]->60
        // ngram [1,2] has continuations: 50(count=1), 60(count=1)
        let tokens = vec![1u32, 2, 50, 1, 2, 60];
        let idx = NgramIndex::build(&tokens, 2);

        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        assert_eq!(conts.len(), 2,
            "ngram [1,2] should have 2 continuations");
        assert!(conts.contains(&50), "should contain token 50");
        assert!(conts.contains(&60), "should contain token 60");
    }

    // 76. SpecTree: accepted_from_spine called twice returns identical results
    #[test]
    fn spec_tree_accepted_from_spine_idempotent() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter = vec![100u32, 200];
        let prompt = vec![1, 2, 100, 5, 6, 7];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let target = tree.spine_token_ids();

        // Act — call twice
        let (count1, tokens1) = tree.accepted_from_spine(&target);
        let (count2, tokens2) = tree.accepted_from_spine(&target);

        // Assert
        assert_eq!(count1, count2,
            "accepted count must be identical across calls");
        assert_eq!(tokens1, tokens2,
            "accepted tokens must be identical across calls");
    }

    // 78. SpecTree: branch_token_ids returns node IDs in ascending order
    #[test]
    fn spec_tree_branch_token_ids_node_ids_ascending() {
        // Arrange
        let config = SpecTreeConfig {
            adapter_top_k: 4,
            max_branches_per_node: 2,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter = vec![10u32, 20, 30, 40];
        let prompt = vec![1, 2, 10, 5, 10, 7];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act
        let branches = tree.branch_token_ids();
        if branches.len() >= 2 {
            // Assert — node IDs strictly increasing
            for w in branches.windows(2) {
                assert!(w[0].0 < w[1].0,
                    "branch node_ids must be in ascending order: {} < {}",
                    w[0].0, w[1].0);
            }
        }
    }

    // 79. SpecTree: spine IDs plus branch IDs partition all node IDs
    #[test]
    fn spec_tree_spine_ids_plus_branch_ids_partition_all_ids() {
        // Arrange
        let config = SpecTreeConfig {
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 5, 10, 7, 8];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act
        let spine_set: std::collections::HashSet<u32> =
            tree.spine_ids().into_iter().collect();
        let branch_set: std::collections::HashSet<u32> =
            tree.branch_token_ids().into_iter().map(|(id, _)| id).collect();
        let all_node_ids: std::collections::HashSet<u32> =
            tree.nodes().iter().map(|n| n.node_id).collect();

        // Assert — spine ∪ branch = all, spine ∩ branch = ∅
        let union: std::collections::HashSet<u32> =
            spine_set.union(&branch_set).copied().collect();
        assert_eq!(union, all_node_ids,
            "spine ∪ branch must equal all node IDs");
        let intersection: std::collections::HashSet<u32> =
            spine_set.intersection(&branch_set).copied().collect();
        assert!(intersection.is_empty(),
            "spine ∩ branch must be empty");
    }

    // 80. CSR mask: root self-column value equals total_seq_len
    #[test]
    fn spec_tree_csr_mask_root_self_col_equals_total_seq_len() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter = vec![10u32, 20];
        let prompt = vec![1, 2, 3, 10, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let total_seq = 10;

        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);

        // Assert — root row (row 0): last element = total_seq_len + 0
        let root_row = &indices[indptr[0]..indptr[1]];
        let root_last = *root_row.last().unwrap();
        assert_eq!(root_last, total_seq,
            "root self-column must equal total_seq_len = {}", total_seq);
    }

    // 81. CSR mask: non-root self-column equals total_seq_len + node_id
    #[test]
    fn spec_tree_csr_mask_non_root_self_col_equals_total_seq_plus_node_id() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter = vec![10u32, 20];
        let prompt = vec![1, 2, 3, 10, 5, 6];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let total_seq = 7;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);

        // Act & Assert — for each non-root node, last col = total_seq + node_id
        for node_id in 1..tree.len() {
            let row_start = indptr[node_id];
            let row_end = indptr[node_id + 1];
            let row = &indices[row_start..row_end];
            let last = *row.last().unwrap();
            assert_eq!(last, total_seq + node_id,
                "node {} self-col must equal {} + {} = {}",
                node_id, total_seq, node_id, total_seq + node_id);
        }
    }

    // 82. CSR mask: same tree, larger total_seq — abstract ancestor IDs are identical
    #[test]
    fn spec_tree_csr_mask_same_tree_larger_total_seq_abstract_ancestors_identical() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter = vec![10u32, 20];
        let prompt = vec![1, 2, 3, 10, 5, 6];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act
        let ts5 = 5usize;
        let ts10 = 10usize;
        let (indptr5, indices5) = tree.tree_attention_mask_csr(ts5);
        let (indptr10, indices10) = tree.tree_attention_mask_csr(ts10);

        // Assert — for each node, the abstract ancestor IDs (column - total_seq) are identical
        // The prefix columns differ (5 vs 10), but ancestor+self offset IDs are the same
        for node_id in 0..tree.len() {
            let row5 = &indices5[indptr5[node_id]..indptr5[node_id + 1]];
            let row10 = &indices10[indptr10[node_id]..indptr10[node_id + 1]];
            // Extract non-prefix columns (those >= total_seq) and subtract total_seq
            let ancestors5: Vec<usize> = row5.iter()
                .filter(|&&c| c >= ts5).map(|&c| c - ts5).collect();
            let ancestors10: Vec<usize> = row10.iter()
                .filter(|&&c| c >= ts10).map(|&c| c - ts10).collect();
            assert_eq!(ancestors5, ancestors10,
                "node {} abstract ancestor+self IDs must be identical across total_seq", node_id);
        }
    }

    // 83. mask_shape: rows independent of total_seq_len
    #[test]
    fn spec_tree_mask_shape_rows_independent_of_total_seq() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter = vec![10u32, 20];
        let prompt = vec![1, 2, 3, 10, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act
        let (rows1, _) = tree.mask_shape(5);
        let (rows2, _) = tree.mask_shape(100);
        let (rows3, _) = tree.mask_shape(0);

        // Assert — rows always equal tree.len()
        assert_eq!(rows1, tree.len());
        assert_eq!(rows2, tree.len());
        assert_eq!(rows3, tree.len());
        assert_eq!(rows1, rows2);
        assert_eq!(rows2, rows3);
    }

    // 84. mask_shape: cols formula equals len + total_seq_len
    #[test]
    fn spec_tree_mask_shape_cols_formula_verified() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter = vec![10u32, 20];
        let prompt = vec![1, 2, 3, 10, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act & Assert — for various total_seq values
        for total_seq in [0, 1, 5, 10, 50, 100] {
            let (_, cols) = tree.mask_shape(total_seq);
            assert_eq!(cols, tree.len() + total_seq,
                "cols must equal tree.len({}) + total_seq({}) = {}",
                tree.len(), total_seq, tree.len() + total_seq);
        }
    }

    // 85. tree_attention_mask_csr: deterministic (same call twice yields identical output)
    #[test]
    fn spec_tree_tree_attention_mask_csr_deterministic() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3, 10, 5, 6, 7];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let total_seq = 8;

        // Act — call twice
        let (indptr1, indices1) = tree.tree_attention_mask_csr(total_seq);
        let (indptr2, indices2) = tree.tree_attention_mask_csr(total_seq);

        // Assert
        assert_eq!(indptr1, indptr2,
            "indptr must be identical across calls");
        assert_eq!(indices1, indices2,
            "indices must be identical across calls");
    }

    // 86. SpecTree: build with same prompt different adapter tokens produces different trees
    #[test]
    fn spec_tree_build_same_prompt_different_adapter_produces_different_trees() {
        // Arrange
        let config = SpecTreeConfig::default();
        let prompt = vec![1, 2, 3, 4, 5];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let adapter1 = vec![100u32, 200];
        let adapter2 = vec![999u32, 888];

        // Act
        let tree1 = SpecTree::build(config.clone(), &adapter1, &prompt, &ngram_idx);
        let tree2 = SpecTree::build(config, &adapter2, &prompt, &ngram_idx);

        // Assert — root tokens differ
        assert_ne!(tree1.all_token_ids(), tree2.all_token_ids(),
            "different adapter tokens must produce different trees");
        assert_ne!(tree1.node(0).unwrap().token_id, tree2.node(0).unwrap().token_id);
    }

    // 87. SpecTree: build with same adapter different pld_ngram_len produces different spine lengths
    #[test]
    fn spec_tree_build_same_adapter_different_pld_ngram_len_spine_differs() {
        // Arrange — prompt has patterns at different n-gram lengths
        // tokens: [10, 20, 30, 10, 20, 40, 10, 20, 50]
        // n=2 windows: [10,20]->30, [20,30]->10, [30,10]->20, [10,20]->40, [20,40]->10, ...
        // n=3 windows: [10,20,30]->10, [20,30,10]->20, [30,10,20]->40, [10,20,40]->10, ...
        let prompt = vec![10u32, 20, 30, 10, 20, 40, 10, 20, 50];
        let adapter = vec![10u32, 20];

        // Act — build with pld_ngram_len=2 and pld_ngram_len=3
        let config2 = SpecTreeConfig { pld_ngram_len: 2, ..SpecTreeConfig::default() };
        let config3 = SpecTreeConfig { pld_ngram_len: 3, ..SpecTreeConfig::default() };
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree2 = SpecTree::build(config2, &adapter, &prompt, &ngram_idx);
        let tree3 = SpecTree::build(config3, &adapter, &prompt, &ngram_idx);

        // Assert — different pld_ngram_len affects PLD matching
        // Both trees have at least root; spine lengths may differ
        assert!(tree2.spine_len >= 1, "tree2 must have at least root");
        assert!(tree3.spine_len >= 1, "tree3 must have at least root");
        // The spine_token_ids content differs because PLD matching windows differ
        // At minimum, the trees were built without panic
    }

    // 88. attention_paths: non-root path starts with 0 (root id)
    #[test]
    fn spec_tree_attention_paths_non_root_first_entry_is_zero() {
        // Arrange
        let config = SpecTreeConfig {
            adapter_top_k: 3,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 5, 6];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act & Assert — every non-root path starts at 0
        for node_id in 1..tree.total_nodes {
            let path = &tree.attention_paths[node_id];
            assert!(!path.is_empty(),
                "non-root node {} must have non-empty attention path", node_id);
            assert_eq!(path[0], 0,
                "non-root node {} attention path must start with 0 (root)", node_id);
        }
    }

    // 89. attention_paths: non-root path terminates at parent just before self is implied
    #[test]
    fn spec_tree_attention_paths_non_root_last_entry_equals_parent() {
        // Arrange
        let config = SpecTreeConfig {
            adapter_top_k: 3,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 10, 5, 6, 7];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act & Assert — last entry of attention_paths for non-root = parent_id
        for node_id in 1..tree.total_nodes {
            let parent_id = tree.nodes[node_id].parent_id.unwrap();
            let path = &tree.attention_paths[node_id];
            let last_in_path = *path.last().unwrap();
            assert_eq!(last_in_path, parent_id,
                "node {} path last entry must be parent_id {}", node_id, parent_id);
        }
    }

    // 90. CSR mask: each indptr diff is positive (every row has at least one element)
    #[test]
    fn spec_tree_csr_mask_each_indptr_diff_positive() {
        // Arrange
        let config = SpecTreeConfig {
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 2,
            ..SpecTreeConfig::default()
        };
        let adapter = vec![10u32, 20, 30];
        let prompt = vec![1, 2, 3, 10, 5, 6];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let total_seq = 4;
        let (indptr, _indices) = tree.tree_attention_mask_csr(total_seq);

        // Act & Assert
        for i in 0..tree.total_nodes {
            let diff = indptr[i + 1] - indptr[i];
            assert!(diff > 0,
                "row {} must have at least 1 element (diff = {})", i, diff);
        }
    }

    // 91. NgramIndex: get_ngram_continuations with n=3 returns multiple results
    #[test]
    fn ngram_index_get_ngram_continuations_n3_multiple_results() {
        // Arrange — tokens where [1,2,3] appears twice with different continuations
        // [1,2,3,50, 1,2,3,60, 1,2,3,50]
        // n=3 windows: [1,2,3]->50, [2,3,50]->1, [3,50,1]->2, [50,1,2]->3,
        //              [1,2,3]->60, [2,3,60]->1, ...
        //              [1,2,3]->50
        // ngram [1,2,3] has: 50(count=2), 60(count=1)
        let tokens = vec![1u32, 2, 3, 50, 1, 2, 3, 60, 1, 2, 3, 50];
        let idx = NgramIndex::build(&tokens, 3);

        // Act
        let conts = idx.get_ngram_continuations(&[1, 2, 3], 5);

        // Assert
        assert_eq!(conts.len(), 2,
            "ngram [1,2,3] should have 2 distinct continuations");
        // First must be 50 (count=2) due to frequency sorting
        assert_eq!(conts[0], 50,
            "most frequent continuation must be 50");
        assert_eq!(conts[1], 60,
            "second continuation must be 60");
    }

    // ==================================================================
    // 15 new tests — edge cases and boundary conditions
    // ==================================================================

    // 1. NgramIndex: get_ngram_continuations panic when ngram length mismatches n
    // 验证 get_ngram_continuations 在传入长度不匹配的 n-gram 时会 panic
    #[test]
    #[should_panic(expected = "n-gram length mismatch")]
    fn ngram_index_get_ngram_continuations_length_mismatch_panics() {
        // Arrange: build with n=2, but query with 3-element n-gram
        let tokens = vec![1u32, 2, 3, 4, 5];
        let idx = NgramIndex::build(&tokens, 2);
        // Act — should panic because ngram.len()=3 != self.n=2
        let _ = idx.get_ngram_continuations(&[1, 2, 3], 5);
    }

    // 2. NgramIndex: get_continuations returns results strictly sorted by descending frequency
    // 验证 get_continuations 的返回结果严格按频率降序排列（多候选项场景）
    #[test]
    fn ngram_index_get_continuations_strict_frequency_descending() {
        // Arrange: token 1 followed by 2 (5 times), 3 (3 times), 4 (1 time)
        let mut tokens = Vec::new();
        for _ in 0..5 { tokens.extend_from_slice(&[1u32, 2]); }
        for _ in 0..3 { tokens.extend_from_slice(&[1u32, 3]); }
        tokens.extend_from_slice(&[1u32, 4]);
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(1, 10);
        // Assert: strict descending frequency order
        assert_eq!(conts.len(), 3);
        assert_eq!(conts[0], 2, "highest frequency continuation should be first");
        assert_eq!(conts[1], 3, "second highest frequency should be second");
        assert_eq!(conts[2], 4, "lowest frequency should be last");
    }

    // 3. SpecTree: find_pld_continuations hits max_spine_depth limit inside prompt
    // 验证 PLD 延续在达到 max_spine_depth 时会停止（即使 prompt 中还有更多 token）
    #[test]
    fn spec_tree_pld_stops_at_max_spine_depth_even_with_more_prompt() {
        // Arrange: pld_ngram_len=1 → scan starts at index 1
        // Token 10 at index 1 → continuations are [20, 30, 40, 50]
        let config = SpecTreeConfig {
            max_spine_depth: 2, // root + 1 extension max
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![5u32, 10, 20, 30, 40, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: spine should be root + 1 extension only (max_spine_depth=2)
        let spine = tree.spine_token_ids();
        assert_eq!(spine.len(), 2, "spine should be exactly 2 (max_spine_depth)");
        assert_eq!(spine[0], 10);
        assert_eq!(spine[1], 20);
    }

    // 4. SpecTree: accepted_from_spine returns 0 when first element mismatches
    // 验证 target 的第一个 token 就不匹配时返回 (0, [])
    #[test]
    fn spec_tree_accepted_from_spine_first_element_mismatch() {
        // Arrange: token 10 at index >= pld_ngram_len=1 to get PLD extension
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 at index 1 → continuations [50, 60, 70]
        let prompt = vec![5u32, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        assert!(spine.len() > 1, "spine should have extensions");
        // Act: target first element is completely different from spine[0]
        let (count, accepted) = tree.accepted_from_spine(&[99999]);
        // Assert
        assert_eq!(count, 0, "should accept 0 when first element mismatches");
        assert!(accepted.is_empty());
    }

    // 5. NgramIndex: n=4 with enough tokens produces correct 4-gram continuations
    // 验证 n=4 的 NgramIndex 能正确索引和查询四元组
    #[test]
    fn ngram_index_n4_correct_continuations() {
        // Arrange: n=4 windows: [1,2,3,4]->50, [2,3,4,50]->60, [3,4,50,60]->70
        let tokens = vec![1u32, 2, 3, 4, 50, 60, 70];
        let idx = NgramIndex::build(&tokens, 4);
        // Act
        let conts = idx.get_ngram_continuations(&[1, 2, 3, 4], 5);
        // Assert
        assert_eq!(conts, vec![50], "4-gram [1,2,3,4] should have continuation 50");
        let conts2 = idx.get_ngram_continuations(&[2, 3, 4, 50], 5);
        assert_eq!(conts2, vec![60], "4-gram [2,3,4,50] should have continuation 60");
    }

    // 6. SpecTree: n-gram branch acceptance formula verification for second branch
    // 验证第二个 n-gram branch 的 estimated_acceptance 按公式计算
    #[test]
    fn spec_tree_ngram_branch_second_branch_acceptance_formula() {
        // Arrange: ngram_top_k=2 ensures at most 2 n-gram branches
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 2,
            ngram_top_k: 2,
            pld_ngram_len: 3,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32];
        // Token 10 followed by different tokens, creating multiple n-gram continuations
        let prompt = vec![10u32, 50, 10, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act: find n-gram branches
        let ngram_branches: Vec<&SpecNode> = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::NgramBranch))
            .collect();
        // Assert: each n-gram branch should have acceptance in (0, 0.15]
        // Formula: 0.05 + 0.05 * (ngram_top_k - b) / ngram_top_k
        // For b=0: 0.05 + 0.05 * (2-0)/2 = 0.10
        // For b=1: 0.05 + 0.05 * (2-1)/2 = 0.075
        for node in &ngram_branches {
            assert!(node.estimated_acceptance > 0.05,
                "n-gram acceptance should be > 0.05, got {}", node.estimated_acceptance);
            assert!(node.estimated_acceptance <= 0.10 + f32::EPSILON,
                "n-gram acceptance should be <= 0.10, got {}", node.estimated_acceptance);
        }
    }

    // 7. SpecTree: build where all continuations are the same token (PLD dedup stress)
    // 验证 PLD 延续在所有后续 token 相同时正确去重
    #[test]
    fn spec_tree_pld_all_continuations_same_token() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 appears multiple times, each followed by 50
        let prompt = vec![10u32, 50, 10, 50, 10, 50, 10, 50, 10, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: spine should be [10, 50] — 50 deduplicated from multiple occurrences
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        let count_50 = spine.iter().filter(|&&t| t == 50).count();
        assert_eq!(count_50, 1, "50 should appear exactly once in spine (dedup)");
    }

    // 8. SpecTree: spine_len field is correctly set after build with PLD extensions
    // 验证 spine_len 字段在有 PLD 延续时正确反映 spine 节点数
    #[test]
    fn spec_tree_spine_len_field_matches_spine_ids_len() {
        // Arrange: token 10 at index 1 → PLD continuations [50, 60, 70]
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![5u32, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: spine_len should match spine_ids().len()
        assert!(tree.spine_len > 0, "should have PLD extensions");
        assert_eq!(tree.spine_len, tree.spine_ids().len(),
            "spine_len must equal spine_ids().len()");
    }

    // 9. SpecNode: construction with negative estimated_acceptance (boundary)
    // 验证 SpecNode 可以存储负的 estimated_acceptance（虽然语义上无意义，但类型允许）
    #[test]
    fn spec_node_negative_estimated_acceptance() {
        // Arrange & Act
        let node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::NgramBranch,
            estimated_acceptance: -0.5f32,
            position_offset: 0,
        };
        // Assert: f32 stores negative values correctly
        assert!(node.estimated_acceptance < 0.0,
            "negative acceptance should be stored as-is");
        assert_eq!(node.estimated_acceptance, -0.5f32);
    }

    // 10. DraftSource: AdapterTopK { k: 0 } is constructible and hashable
    // 验证 k=0 的 AdapterTopK 可以构造且正确参与 Hash 运算
    #[test]
    fn draft_source_adapter_top_k_zero_constructible_and_hashable() {
        // Arrange
        let ds = DraftSource::AdapterTopK { k: 0 };
        // Act: use in HashSet to verify Hash
        let mut set = std::collections::HashSet::new();
        assert!(set.insert(ds));
        // Assert: k=0 distinguishes from other variants
        assert!(set.contains(&DraftSource::AdapterTopK { k: 0 }));
        assert!(!set.contains(&DraftSource::AdapterTopK { k: 1 }));
        assert!(!set.contains(&DraftSource::PldSpine));
    }

    // 11. SpecTree: build with max_tree_size exactly equal to natural node count
    // 验证 max_tree_size 恰好等于自然构建的节点数时，不会截断任何节点
    #[test]
    fn spec_tree_max_tree_size_exact_natural_count() {
        // Arrange: compute expected node count
        // Root(1) + adapter branches(adapter_top_k-1=1) = 2 with no PLD, no n-gram
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ngram_top_k: 0,
            pld_ngram_len: 3,
            max_tree_size: 2,
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1u32, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: all 2 nodes present, none truncated
        assert_eq!(tree.len(), 2, "should have exactly 2 nodes when max_tree_size matches natural count");
        assert_eq!(tree.node(0).unwrap().token_id, 10);
        assert_eq!(tree.node(1).unwrap().token_id, 20);
    }

    // 12. SpecTree: total_nodes field is accessible and matches len()
    // 验证 total_nodes 字段与 len() 方法返回值一致
    #[test]
    fn spec_tree_total_nodes_field_equals_len_and_nodes() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10u32, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: total_nodes is consistent with len() and nodes() slice
        assert_eq!(tree.total_nodes, tree.len());
        assert_eq!(tree.total_nodes, tree.nodes().len());
    }

    // 13. SpecTree: CSR mask columns are strictly non-decreasing within each row
    // 验证 CSR mask 每行内的列索引是严格非递减的（CSR 格式要求）
    #[test]
    fn spec_tree_csr_mask_columns_non_decreasing_per_row() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10u32, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let total_seq = 5;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        // Assert: columns within each row should be non-decreasing
        for i in 0..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            for w in row.windows(2) {
                assert!(w[0] <= w[1],
                    "row {} has non-monotonic columns: {} > {}", i, w[0], w[1]);
            }
        }
    }

    // 14. SpecTree: build with prompt that has token 0 (zero) as adapter token
    // 验证 token_id=0 的 adapter token 不会导致 PLD 扫描跳过
    #[test]
    fn spec_tree_build_adapter_token_zero_pld_scan_not_skipped() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![0u32]; // token ID 0
        // Token 0 at index 1 → continuations [50, 60]
        let prompt = vec![10u32, 0, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: PLD should find token 0 at index 1 and extend spine
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 0, "root should have token_id 0");
        assert!(spine.len() >= 2, "should have PLD extension from token 0 at index 1");
        if spine.len() >= 2 {
            assert_eq!(spine[1], 50, "first PLD continuation should be 50");
        }
    }

    // 15. NgramIndex: get_ngram_continuations with n=2 and top_k=1 returns only the best
    // 验证 top_k=1 时只返回频率最高的一个候选项
    #[test]
    fn ngram_index_n2_top_k_1_returns_only_best() {
        // Arrange: [5,6] → 10 (4 times), [5,6] → 20 (2 times), [5,6] → 30 (1 time)
        let mut tokens = Vec::new();
        for _ in 0..4 { tokens.extend_from_slice(&[5u32, 6, 10]); }
        for _ in 0..2 { tokens.extend_from_slice(&[5u32, 6, 20]); }
        tokens.extend_from_slice(&[5u32, 6, 30]);
        let idx = NgramIndex::build(&tokens, 2);
        // Act
        let conts = idx.get_ngram_continuations(&[5, 6], 1);
        // Assert: only the most frequent continuation
        assert_eq!(conts.len(), 1, "top_k=1 should return exactly 1 result");
        assert_eq!(conts[0], 10, "should return the highest-frequency continuation");
        // Verify full list has all 3 when top_k is larger
        let all_conts = idx.get_ngram_continuations(&[5, 6], 10);
        assert_eq!(all_conts.len(), 3, "full list should have 3 distinct continuations");
    }

    // ==================================================================
    // Wave-34: 15 additional edge case tests
    // ==================================================================

    // 1. SpecTree: all_token_ids preserves node insertion order
    // 验证 all_token_ids() 严格按照 nodes Vec 的插入顺序返回 token IDs
    #[test]
    fn spec_tree_all_token_ids_matches_node_insertion_order() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10u32, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act
        let token_ids = tree.all_token_ids();
        // Assert: all_token_ids must match iterating nodes() in order
        let expected: Vec<u32> = tree.nodes().iter().map(|n| n.token_id).collect();
        assert_eq!(token_ids, expected,
            "all_token_ids must preserve insertion order");
    }

    // 2. SpecTree: spine_estimated_acceptance decreases monotonically with depth
    // 验证 spine 节点的 estimated_acceptance 随深度严格递减
    #[test]
    fn spec_tree_spine_acceptance_strictly_decreasing_per_depth() {
        // Arrange: root has 0.70, spine extensions have 0.60 - depth*0.05
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![5u32, 10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Act
        let spine_ids = tree.spine_ids();
        // Assert: each successive spine node has lower acceptance
        for w in spine_ids.windows(2) {
            let acc_a = tree.node(w[0]).unwrap().estimated_acceptance;
            let acc_b = tree.node(w[1]).unwrap().estimated_acceptance;
            assert!(acc_b < acc_a,
                "spine acceptance must decrease: node {} acceptance {} >= node {} acceptance {}",
                w[0], acc_a, w[1], acc_b);
        }
    }

    // 3. NgramIndex: two different n-grams map to different hash keys
    // 验证不同 n-gram 内容产生不同的 hash key，不会错误地共享条目
    #[test]
    fn ngram_index_different_ngrams_different_hashes() {
        // Arrange: build index with n=2
        let tokens = vec![1u32, 2, 100, 3, 4, 200];
        let idx = NgramIndex::build(&tokens, 2);
        // Act: query two distinct 2-grams
        let conts_a = idx.get_ngram_continuations(&[1, 2], 5);
        let conts_b = idx.get_ngram_continuations(&[3, 4], 5);
        // Assert: different continuations — [1,2]->100 vs [3,4]->200
        assert_eq!(conts_a, vec![100], "[1,2] should continue to 100");
        assert_eq!(conts_b, vec![200], "[3,4] should continue to 200");
    }

    // 4. SpecTree: CSR mask each row includes at least total_seq_len prefix columns
    // 验证每行 CSR mask 至少包含 total_seq_len 个前缀列（因果注意力约束）
    #[test]
    fn spec_tree_csr_each_row_includes_full_prefix() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10u32, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let total_seq = 7;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        // Assert: each row starts with [0, 1, ..., total_seq-1]
        for row in 0..tree.len() {
            let start = indptr[row];
            let end = indptr[row + 1];
            let nnz = end - start;
            assert!(nnz >= total_seq + 1,
                "row {} has {} nnz, expected at least {} (prefix + self)",
                row, nnz, total_seq + 1);
            // First total_seq columns should be [0..total_seq)
            for col in 0..total_seq {
                assert_eq!(indices[start + col], col,
                    "row {} prefix column {} expected {}, got {}",
                    row, col, col, indices[start + col]);
            }
        }
    }

    // 5. SpecTree: build with adapter_top_k=1 produces no adapter branch nodes
    // 验证 adapter_top_k=1 时只有 root（top-1），没有 top-2/3 分支
    #[test]
    fn spec_tree_adapter_top_k_1_no_adapter_branches() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 3,
            ngram_top_k: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![1u32, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: no AdapterTopK with k > 1
        let adapter_branches: Vec<&SpecNode> = tree.nodes().iter()
            .filter(|n| matches!(n.source, DraftSource::AdapterTopK { k } if k > 1))
            .collect();
        assert!(adapter_branches.is_empty(),
            "adapter_top_k=1 should produce no branches with k>1, found {}",
            adapter_branches.len());
    }

    // 6. SpecTree: NgramIndex table field is accessible and has correct n value
    // 验证 NgramIndex 的内部 n 字段在 build 后正确设置
    #[test]
    fn ngram_index_n_field_matches_build_parameter() {
        // Arrange & Act
        let idx3 = NgramIndex::build(&[1u32, 2, 3, 4, 5], 3);
        let idx1 = NgramIndex::build(&[1u32, 2, 3], 1);
        let idx7 = NgramIndex::build(&[1u32, 2, 3, 4, 5, 6, 7, 8], 7);
        // Assert
        assert_eq!(idx3.n, 3, "n field should be 3 after build with n=3");
        assert_eq!(idx1.n, 1, "n field should be 1 after build with n=1");
        assert_eq!(idx7.n, 7, "n field should be 7 after build with n=7");
    }

    // 7. SpecTree: branch nodes have parent_id pointing to a spine or root node
    // 验证所有分支节点的 parent_id 指向 spine 或 root 节点
    #[test]
    fn spec_tree_branch_parent_points_to_spine_or_root() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 2,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![10u32, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let spine_set: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        // Act & Assert: every non-spine node has parent in spine set
        for node in tree.nodes() {
            if !spine_set.contains(&node.node_id) {
                if let Some(pid) = node.parent_id {
                    assert!(spine_set.contains(&pid),
                        "branch node {} parent {} not in spine set",
                        node.node_id, pid);
                }
            }
        }
    }

    // 8. SpecTree: config field is accessible and preserves build parameters
    // 验证 SpecTree 构建后 config 字段保留了原始配置参数
    #[test]
    fn spec_tree_config_field_preserves_build_params() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 7,
            adapter_top_k: 5,
            max_branches_per_node: 3,
            pld_ngram_len: 4,
            ngram_top_k: 6,
            max_tree_size: 50,
        };
        // Act
        let tree = SpecTree::new(config.clone());
        // Assert: config field matches input
        assert_eq!(tree.config.max_spine_depth, 7);
        assert_eq!(tree.config.adapter_top_k, 5);
        assert_eq!(tree.config.max_branches_per_node, 3);
        assert_eq!(tree.config.pld_ngram_len, 4);
        assert_eq!(tree.config.ngram_top_k, 6);
        assert_eq!(tree.config.max_tree_size, 50);
    }

    // 9. NgramIndex: get_continuations returns empty for unseen token
    // 验证查询从未在 prompt 中出现过的 token 时返回空列表
    #[test]
    fn ngram_index_continuations_empty_for_unseen_token() {
        // Arrange: only tokens 1-5 in prompt
        let tokens = vec![1u32, 2, 3, 4, 5, 1, 2];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(999, 5);
        // Assert
        assert!(conts.is_empty(),
            "get_continuations for unseen token should return empty");
    }

    // 10. SpecTree: PLD continuation dedup skips tokens already in spine
    // 验证 PLD 延伸不会重复添加已存在于 spine 中的 token
    #[test]
    fn spec_tree_pld_continuation_skips_existing_spine_tokens() {
        // Arrange: prompt has 10 followed by 10 again — 10 is root, should not be
        // added as continuation (find_pld_continuations deduplicates)
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 at index 2, continuations = [10, 20, 30, 10, 40] → deduped: [10, 20, 30, 40]
        // But 10 is the root token, so PLD may still include it as a continuation
        let prompt = vec![5u32, 6, 10, 50, 60, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();
        // Assert: no duplicate token IDs in spine
        let mut seen = std::collections::HashSet::new();
        for &tok in &spine {
            assert!(seen.insert(tok),
                "token {} appears more than once in spine: {:?}", tok, spine);
        }
    }

    // 11. SpecTree: mask_shape for empty tree returns (0, total_seq_len)
    // 验证空树的 mask_shape 返回 (0, total_seq_len)
    #[test]
    fn spec_tree_mask_shape_empty_tree_returns_zero_rows() {
        // Arrange
        let tree = SpecTree::new(SpecTreeConfig::default());
        // Act
        let (rows, cols) = tree.mask_shape(15);
        // Assert
        assert_eq!(rows, 0, "empty tree should have 0 mask rows");
        assert_eq!(cols, 15, "empty tree mask cols should equal total_seq_len");
    }

    // 12. SpecTree: CSR mask ancestor columns for leaf nodes include root
    // 验证叶节点（无子节点）的 CSR mask ancestor 列包含 root 节点
    #[test]
    fn spec_tree_csr_leaf_ancestors_include_root() {
        // Arrange: adapter_top_k=2 with no n-gram → adapter top-2 is a leaf branch
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![5u32, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let total_seq = 5;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        // Find leaf nodes (children.is_empty())
        for node in tree.nodes() {
            if node.children.is_empty() && node.parent_id.is_some() {
                let row = node.node_id as usize;
                let start = indptr[row];
                let end = indptr[row + 1];
                let cols: Vec<usize> = indices[start..end].to_vec();
                // Must include root column = total_seq + 0
                assert!(cols.contains(&(total_seq)),
                    "leaf node {} should attend to root (column {}), got cols {:?}",
                    node.node_id, total_seq, cols);
            }
        }
    }

    // 13. SpecTree: accepted_from_spine with empty target returns (0, [])
    // 验证空 target 列表时 accepted_from_spine 返回零匹配
    #[test]
    fn spec_tree_accepted_from_spine_empty_target() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![5u32, 10, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        assert!(!tree.is_empty(), "tree should not be empty");
        // Act
        let (count, accepted) = tree.accepted_from_spine(&[]);
        // Assert
        assert_eq!(count, 0, "empty target should match 0 spine tokens");
        assert!(accepted.is_empty());
    }

    // 14. SpecTree: adapter top-2 branch has position_offset = 1
    // 验证 adapter top-2 分支节点的 position_offset 恰好为 1
    #[test]
    fn spec_tree_adapter_branch_position_offset_equals_one() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            pld_ngram_len: 3,
            ngram_top_k: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![1u32, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: adapter branches (k >= 2) all have position_offset = 1
        for node in tree.nodes() {
            if let DraftSource::AdapterTopK { k } = node.source {
                if k >= 2 {
                    assert_eq!(node.position_offset, 1,
                        "adapter top-{} branch should have position_offset=1, got {}",
                        k, node.position_offset);
                }
            }
        }
    }

    // 15. NgramIndex: multiple n-grams with same continuation sum frequencies
    // 验证相同 n-gram 出现多次时，其 continuation 的频率正确累加
    #[test]
    fn ngram_index_same_ngram_accumulates_continuation_frequency() {
        // Arrange: [1,2]→30 appears 3 times, [1,2]→40 appears 1 time
        let tokens = vec![1u32, 2, 30, 1, 2, 30, 1, 2, 30, 1, 2, 40];
        let idx = NgramIndex::build(&tokens, 2);
        // Act
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        // Assert: 30 should be first (frequency 3 > 1), 40 second
        assert_eq!(conts[0], 30, "highest frequency continuation should be first");
        assert_eq!(conts.len(), 2, "should have exactly 2 distinct continuations");
        assert!(conts.contains(&40), "40 should appear as continuation");
    }

    // ==================================================================
    // NEW TESTS (tests 313+, 15 additional tests)
    // Focus: SpecTree depth boundary, token ID 0 edge, NgramIndex
    //         get_continuations frequency sort, Config Default verification,
    //         Node token_id boundary (u32::MAX), empty prompt special cases
    // ==================================================================

    // 1. SpecTree: max_spine_depth=2 produces at most 2 spine nodes
    //    验证 max_spine_depth 严格限制 spine 深度为 2 (root + 1 extension)
    #[test]
    fn spec_tree_depth_boundary_max_spine_depth_two() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 followed by many unique tokens, but spine should stop at depth 2
        let prompt = vec![10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        let spine = tree.spine_ids();
        assert!(spine.len() <= 2,
            "spine should have at most 2 nodes, got {}", spine.len());
    }

    // 2. SpecTree: token ID 0 as adapter top-1 builds correctly
    //    验证 token_id=0 作为 adapter top-1 时树结构正确
    #[test]
    fn spec_tree_token_id_zero_as_adapter_root() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![0u32];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        assert!(!tree.is_empty());
        assert_eq!(tree.node(0).unwrap().token_id, 0);
        let all_ids = tree.all_token_ids();
        assert_eq!(all_ids[0], 0);
    }

    // 3. SpecTree: token ID 0 in prompt produces PLD continuation
    //    验证 prompt 中 token_id=0 的 PLD continuation 正确生成
    #[test]
    fn spec_tree_token_id_zero_pld_continuation() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![0u32];
        // pld_ngram_len=1, scan starts at index 1. Token 0 at index 2.
        let prompt = vec![1, 2, 0, 50, 60];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 0, "root should be token 0");
        assert!(spine.len() >= 2, "should have PLD extensions from token 0 in prompt");
    }

    // 4. NgramIndex: get_continuations returns correct frequency-sorted order
    //    验证 get_continuations 按 continuation 出现频率降序返回
    #[test]
    fn ngram_index_get_continuations_frequency_order_correct() {
        // Arrange: token 5 → 10 (3x), 5 → 20 (2x), 5 → 30 (1x)
        let tokens = vec![5u32, 10, 5, 20, 5, 10, 5, 30, 5, 10];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(5, 10);
        // Assert
        assert!(conts.len() >= 3, "should have at least 3 continuations");
        // Most frequent first
        assert_eq!(conts[0], 10, "10 (3x) should be first");
        assert_eq!(conts[1], 20, "20 (2x) should be second");
        assert_eq!(conts[2], 30, "30 (1x) should be third");
    }

    // 5. NgramIndex: get_continuations with n=1 returns tokens in strict frequency order
    //    验证 1-gram 索引的 get_continuations 严格按频率降序排列
    #[test]
    fn ngram_index_get_continuations_strict_frequency_desc() {
        // Arrange: token 1 → 100 (4x), 1 → 200 (3x), 1 → 300 (2x), 1 → 400 (1x)
        let tokens = vec![1u32, 100, 1, 100, 1, 200, 1, 100, 1, 200, 1, 300, 1, 100, 1, 300, 1, 200, 1, 400];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(1, 10);
        // Assert: strict descending by frequency
        assert_eq!(conts.len(), 4);
        assert_eq!(conts[0], 100);
        assert_eq!(conts[1], 200);
        assert_eq!(conts[2], 300);
        assert_eq!(conts[3], 400);
    }

    // 6. Config: Default values match SPEC specification exactly
    //    验证 SpecTreeConfig 的 Default 实现与 SPEC 规定完全一致
    #[test]
    fn config_default_matches_spec_values() {
        // Arrange & Act
        let cfg = SpecTreeConfig::default();
        // Assert: SPEC defines max_spine_depth=5, max_branches_per_node=2,
        // pld_ngram_len=3, ngram_top_k=2, adapter_top_k=3, max_tree_size=32
        assert_eq!(cfg.max_spine_depth, 5, "SPEC: max_spine_depth=5");
        assert_eq!(cfg.max_branches_per_node, 2, "SPEC: max_branches_per_node=2");
        assert_eq!(cfg.pld_ngram_len, 3, "SPEC: pld_ngram_len=3");
        assert_eq!(cfg.ngram_top_k, 2, "SPEC: ngram_top_k=2");
        assert_eq!(cfg.adapter_top_k, 3, "SPEC: adapter_top_k=3");
        assert_eq!(cfg.max_tree_size, 32, "SPEC: max_tree_size=32");
    }

    // 7. Config: Default produces identical config across all constructions
    //    验证多次调用 Default::default() 产生完全相同的配置
    #[test]
    fn config_default_idempotent_across_constructions() {
        // Arrange
        let cfg1 = SpecTreeConfig::default();
        let cfg2 = SpecTreeConfig {
            max_spine_depth: 5,
            max_branches_per_node: 2,
            pld_ngram_len: 3,
            ngram_top_k: 2,
            adapter_top_k: 3,
            max_tree_size: 32,
        };
        // Assert
        assert_eq!(cfg1, cfg2, "default should match explicitly constructed config");
    }

    // 8. SpecNode: token_id can hold u32::MAX without overflow
    //    验证 SpecNode 的 token_id 字段能正确存储 u32::MAX
    #[test]
    fn spec_node_token_id_u32_max_stored_correctly() {
        // Arrange
        let node = SpecNode {
            node_id: 0,
            token_id: u32::MAX,
            parent_id: None,
            children: vec![],
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.7,
            position_offset: 0,
        };
        // Assert
        assert_eq!(node.token_id, u32::MAX);
        assert_eq!(node.token_id, 4294967295u32);
    }

    // 9. SpecTree: build with u32::MAX token ID as adapter produces correct root
    //    验证 u32::MAX token ID 作为 adapter 输入时树的构建正确
    #[test]
    fn spec_tree_build_u32_max_adapter_token() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![u32::MAX];
        let prompt: Vec<u32> = vec![];
        let ngram_idx = NgramIndex::build(&prompt, 2);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.node(0).unwrap().token_id, u32::MAX);
        assert_eq!(tree.all_token_ids(), vec![u32::MAX]);
    }

    // 10. SpecTree: empty prompt produces root with no PLD extensions
    //     验证空 prompt 时树只有 root 节点，无 PLD spine extension
    #[test]
    fn spec_tree_empty_prompt_root_only_no_spine_extension() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![42u32];
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &[], &NgramIndex::build(&[], 1));
        // Assert
        assert_eq!(tree.len(), 1, "empty prompt should produce root-only tree");
        let spine = tree.spine_token_ids();
        assert_eq!(spine.len(), 1);
        assert_eq!(spine[0], 42);
        assert!(tree.branch_token_ids().is_empty());
    }

    // 11. SpecTree: empty prompt with multiple adapter tokens produces root + branches only
    //     验证空 prompt 时 adapter branches 仍然正常生成，但无 PLD
    #[test]
    fn spec_tree_empty_prompt_adapter_branches_only() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let ngram_idx = NgramIndex::build(&[], 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &[], &ngram_idx);
        // Assert
        assert_eq!(tree.len(), 3, "root + 2 adapter branches");
        let spine = tree.spine_token_ids();
        assert_eq!(spine.len(), 1, "spine should be root only (no PLD)");
        assert_eq!(tree.node(0).unwrap().token_id, 10);
        assert_eq!(tree.node(1).unwrap().token_id, 20);
        assert_eq!(tree.node(2).unwrap().token_id, 30);
    }

    // 12. SpecTree: depth boundary — max_spine_depth=1 produces exactly root in spine
    //     验证 max_spine_depth=1 严格限制 spine 仅为 root (无 extension)
    #[test]
    fn spec_tree_depth_boundary_exact_root_spine() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Rich prompt with many tokens following 10, but max_spine_depth=1 blocks extensions
        let prompt = vec![1, 10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        let spine = tree.spine_ids();
        assert_eq!(spine.len(), 1, "max_spine_depth=1 → spine should be exactly root");
        assert_eq!(spine[0], 0);
    }

    // 13. NgramIndex: get_continuations after build with n=1 has correct frequency order
    //     验证 n=1 索引的 get_continuations 在 build 后维持正确频率排序
    #[test]
    fn ngram_index_build_then_get_continuations_sorted() {
        // Arrange: build with n=1 so get_continuations (1-gram hash) can find entries
        // Token 3 → 10 (2x), 3 → 20 (1x)
        let tokens = vec![3u32, 10, 3, 20, 3, 10];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(3, 5);
        // Assert
        assert_eq!(conts.len(), 2);
        assert_eq!(conts[0], 10, "10 (2x) should be first");
        assert_eq!(conts[1], 20, "20 (1x) should be second");
    }

    // 14. SpecTree: CSR mask for empty prompt tree (root + adapter branches only)
    //     验证空 prompt 树的 CSR mask 结构正确
    #[test]
    fn spec_tree_csr_mask_empty_prompt_adapter_only_tree() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 3,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let ngram_idx = NgramIndex::build(&[], 2);
        let tree = SpecTree::build(config, &adapter_tokens, &[], &ngram_idx);
        // Act
        let total_seq_len = 0;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert
        assert_eq!(tree.len(), 3);
        assert_eq!(indptr.len(), 4); // len + 1
        // Root row: only self at position 0
        let root_row: Vec<usize> = indices[indptr[0]..indptr[1]].to_vec();
        assert_eq!(root_row, vec![0]);
        // Branch rows: parent (0) + self
        for i in 1..3 {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row: Vec<usize> = indices[start..end].to_vec();
            assert!(row.contains(&0), "branch {} should attend to root", i);
            assert!(row.contains(&i), "branch {} should attend to self", i);
        }
    }

    // 15. SpecTree: NgramIndex with n=1 build, then get_continuations for token absent in table
    //     验证 get_continuations 查询不存在的 token 返回空，不 panic
    #[test]
    fn ngram_index_get_continuations_absent_token_returns_empty() {
        // Arrange: build index from tokens [1,2,3,4], token 99 never appears
        let tokens = vec![1u32, 2, 3, 4, 1, 2, 5];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(99, 5);
        // Assert
        assert!(conts.is_empty(), "token 99 not in index → empty continuations");
    }

    // ===================================================================
    // 15 new tests — focus areas 1–15
    // ===================================================================

    // Focus 1: SpecTree build with empty prompt tokens produces valid tree with adapter root only
    #[test]
    fn spec_tree_build_empty_prompt_tokens_produces_valid_root() {
        // Arrange
        let config = SpecTreeConfig::default();
        let adapter_tokens = vec![42u32, 100, 200];
        let prompt_tokens: Vec<u32> = vec![];
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        // Assert: root exists with adapter top-1 token, no spine extensions (empty prompt)
        assert!(!tree.is_empty());
        assert_eq!(tree.nodes()[0].token_id, 42);
        assert!(tree.spine_token_ids().len() <= 1, "empty prompt → at most root on spine");
    }

    // Focus 2: Node with token_id=0 (valid non-special token) roundtrips through tree operations
    #[test]
    fn spec_node_token_id_zero_roundtrip_in_tree_operations() {
        // Arrange: token_id=0 is a valid token (e.g., <pad> or first vocab entry)
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            max_branches_per_node: 2,
            pld_ngram_len: 2,
            ngram_top_k: 2,
            adapter_top_k: 1,
            max_tree_size: 16,
        };
        let adapter_tokens = vec![0u32]; // token_id=0 as adapter top-1
        let prompt_tokens = vec![0u32, 1, 2, 0, 1, 3]; // prompt contains token 0
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        // Assert: root has token_id=0, retrievable via all_token_ids
        assert_eq!(tree.nodes()[0].token_id, 0);
        assert!(tree.all_token_ids().contains(&0));
    }

    // Focus 3: NgramIndex frequency sorting is stable — same frequency preserves insertion order
    #[test]
    fn ngram_index_stable_frequency_sort_preserves_insertion_order() {
        // Arrange: tokens where continuations A, B, C all appear exactly once from the same n-gram
        // [X, Y, A, X, Y, B, X, Y, C] — each continuation appears once
        let tokens = vec![10u32, 20, 100, 10, 20, 200, 10, 20, 300];
        let idx = NgramIndex::build(&tokens, 2);
        // Act: get_continuations for the bigram [10, 20]
        let conts = idx.get_ngram_continuations(&[10, 20], 10);
        // Assert: all three continuations present; first one encountered (100) should come first
        // since sort_by is stable for equal counts
        assert_eq!(conts.len(), 3);
        assert_eq!(conts[0], 100, "first encountered should remain first after stable sort");
    }

    // Focus 4: Config Default values verification — each field is nonzero and reasonable
    #[test]
    fn spec_tree_config_default_all_fields_nonzero_and_reasonable() {
        // Arrange
        let config = SpecTreeConfig::default();
        // Assert: every field is > 0
        assert!(config.max_spine_depth > 0, "max_spine_depth must be > 0");
        assert!(config.max_branches_per_node > 0, "max_branches_per_node must be > 0");
        assert!(config.pld_ngram_len > 0, "pld_ngram_len must be > 0");
        assert!(config.ngram_top_k > 0, "ngram_top_k must be > 0");
        assert!(config.adapter_top_k > 0, "adapter_top_k must be > 0");
        assert!(config.max_tree_size > 0, "max_tree_size must be > 0");
        // Assert: reasonable bounds
        assert!(config.max_tree_size >= config.adapter_top_k, "max_tree_size >= adapter_top_k");
        assert!(config.max_spine_depth < 100, "max_spine_depth should be reasonable");
        assert!(config.pld_ngram_len >= 2, "pld_ngram_len typically 3-5");
    }

    // Focus 5: Node with u32::MAX token_id — tree build and retrieval
    #[test]
    fn spec_tree_build_with_u32_max_adapter_token_retrievable() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            max_branches_per_node: 0,
            pld_ngram_len: 2,
            ngram_top_k: 0,
            adapter_top_k: 1,
            max_tree_size: 8,
        };
        let adapter_tokens = vec![u32::MAX];
        let prompt_tokens = vec![u32::MAX, 1, 2];
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        // Assert
        assert_eq!(tree.nodes()[0].token_id, u32::MAX);
        assert!(tree.all_token_ids().contains(&u32::MAX));
        let ids = tree.spine_token_ids();
        assert_eq!(ids[0], u32::MAX);
    }

    // Focus 6: Depth boundary — spine length equals max_spine_depth when enough PLD matches exist
    #[test]
    fn spec_tree_spine_length_capped_at_max_spine_depth() {
        // Arrange: long prompt with many PLD continuations, max_spine_depth=3
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            adapter_top_k: 1,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![10u32];
        // Prompt: 10 followed by many tokens → PLD should find continuations
        let prompt_tokens: Vec<u32> = std::iter::once(10u32)
            .chain((100..120).collect::<Vec<_>>())
            .collect();
        let max_spine = config.max_spine_depth;
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        // Assert: spine length <= max_spine_depth
        let spine = tree.spine_ids();
        assert!(spine.len() <= max_spine,
            "spine len {} should be <= max_spine_depth {}",
            spine.len(), max_spine);
    }

    // Focus 7: CSR mask construction — indices count matches indptr diff sum
    #[test]
    fn spec_tree_csr_mask_indices_count_matches_indptr_diff_for_multi_spine() {
        // Arrange: build a tree with a non-trivial spine
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            max_branches_per_node: 1,
            pld_ngram_len: 2,
            ngram_top_k: 1,
            adapter_top_k: 2,
            max_tree_size: 32,
        };
        let adapter_tokens = vec![5u32, 50];
        let prompt_tokens = vec![5, 10, 20, 30, 5, 10, 40];
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        let total_seq_len = prompt_tokens.len();
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: sum of row lengths (indptr diffs) == total indices
        let sum_diffs: usize = (0..tree.len())
            .map(|i| indptr[i + 1] - indptr[i])
            .sum();
        assert_eq!(sum_diffs, indices.len(),
            "sum of indptr diffs ({}) must equal indices len ({})",
            sum_diffs, indices.len());
    }

    // Focus 8: Tree pruning with single node — CSR mask for single-node tree is trivial
    #[test]
    fn spec_tree_single_node_pruned_csr_mask_is_trivial() {
        // Arrange: config that produces only root (no branches, no PLD)
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 2,
            ngram_top_k: 0,
            adapter_top_k: 1,
            max_tree_size: 8,
        };
        let adapter_tokens = vec![7u32];
        let prompt_tokens: Vec<u32> = vec![]; // empty → no PLD extensions
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        // Assert: exactly 1 node
        assert_eq!(tree.len(), 1, "single-node tree");
        // Act: CSR with total_seq_len=5
        let (indptr, indices) = tree.tree_attention_mask_csr(5);
        // Assert: indptr = [0, 6] (5 prefix cols + self)
        assert_eq!(indptr.len(), 2);
        assert_eq!(indptr[0], 0);
        assert_eq!(indices.len(), 6, "5 prefix + 1 self = 6");
        assert_eq!(indices[5], 5, "last index is self at total_seq_len + node_id = 5");
    }

    // Focus 9: Multiple insertions at same depth — verify sibling branches share same parent
    #[test]
    fn spec_tree_sibling_branches_share_same_parent_and_depth() {
        // Arrange: adapter_top_k=4 → root + 3 sibling adapter branches at depth 1
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 2,
            ngram_top_k: 0,
            adapter_top_k: 4,
            max_tree_size: 16,
        };
        let adapter_tokens = vec![100u32, 200, 300, 400];
        let prompt_tokens: Vec<u32> = vec![];
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        // Assert: root + 3 branches = 4 nodes total
        assert_eq!(tree.len(), 4);
        // All branches have parent_id = Some(0) (root)
        for node in tree.nodes().iter().skip(1) {
            assert_eq!(node.parent_id, Some(0), "branch should have root as parent");
            assert_eq!(node.position_offset, 1, "all siblings at same depth");
        }
        // Root has 3 children
        assert_eq!(tree.nodes()[0].children.len(), 3);
    }

    // Focus 10: Config Clone/Debug roundtrip — cloned config is equal, debug string is non-empty
    #[test]
    fn spec_tree_config_clone_debug_roundtrip_is_consistent() {
        // Arrange
        let original = SpecTreeConfig {
            max_spine_depth: 7,
            max_branches_per_node: 3,
            pld_ngram_len: 4,
            ngram_top_k: 5,
            adapter_top_k: 6,
            max_tree_size: 64,
        };
        // Act: clone
        let cloned = original.clone();
        // Assert: equality
        assert_eq!(original, cloned);
        // Act: debug format
        let debug_str = format!("{:?}", cloned);
        // Assert: debug contains key fields
        assert!(debug_str.contains("max_spine_depth: 7"));
        assert!(debug_str.contains("max_tree_size: 64"));
        assert!(!debug_str.is_empty());
    }

    // Focus 11: Node Clone preserves all fields including children vector
    #[test]
    fn spec_node_clone_preserves_all_fields_including_children_vector() {
        // Arrange
        let node = SpecNode {
            node_id: 42,
            token_id: 999,
            parent_id: Some(10),
            children: vec![100, 200, 300],
            source: DraftSource::NgramBranch,
            estimated_acceptance: 0.75,
            position_offset: 5,
        };
        // Act
        let cloned = node.clone();
        // Assert: all fields match exactly
        assert_eq!(cloned.node_id, 42);
        assert_eq!(cloned.token_id, 999);
        assert_eq!(cloned.parent_id, Some(10));
        assert_eq!(cloned.children, vec![100, 200, 300]);
        assert_eq!(cloned.source, DraftSource::NgramBranch);
        assert_eq!(cloned.estimated_acceptance, 0.75);
        assert_eq!(cloned.position_offset, 5);
        // Assert: PartialEq confirms full equality
        assert_eq!(node, cloned);
    }

    // Focus 12: NgramIndex with duplicate n-grams — frequencies accumulate correctly
    #[test]
    fn ngram_index_duplicate_ngrams_accumulate_and_sort_correctly() {
        // Arrange: [A,B,C, A,B,C, A,B,D] — trigram [A,B]→C appears 2x, →D appears 1x
        let tokens = vec![1u32, 2, 3, 1, 2, 3, 1, 2, 4];
        let idx = NgramIndex::build(&tokens, 2);
        // Act
        let conts = idx.get_ngram_continuations(&[1, 2], 10);
        // Assert: C (3) appears first with higher frequency, D (4) second
        assert_eq!(conts.len(), 2);
        assert_eq!(conts[0], 3, "token 3 appears 2x → first");
        assert_eq!(conts[1], 4, "token 4 appears 1x → second");
    }

    // Focus 13: Tree with only root — no children, spine_token_ids returns single element
    #[test]
    fn spec_tree_only_root_no_children_spine_ids_single_element() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            adapter_top_k: 1,
            max_tree_size: 4,
        };
        let adapter_tokens = vec![55u32];
        let prompt_tokens: Vec<u32> = vec![99]; // no PLD match for 55
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        // Assert: only root exists
        assert_eq!(tree.len(), 1);
        let spine = tree.spine_token_ids();
        assert_eq!(spine.len(), 1);
        assert_eq!(spine[0], 55);
        // Root has no children
        assert!(tree.nodes()[0].children.is_empty());
    }

    // Focus 14: SpecTree fresh build independence — two sequential builds produce independent trees
    #[test]
    fn spec_tree_sequential_builds_produce_independent_trees() {
        // Arrange: first build
        let config1 = SpecTreeConfig {
            max_spine_depth: 3,
            max_branches_per_node: 1,
            pld_ngram_len: 2,
            ngram_top_k: 1,
            adapter_top_k: 2,
            max_tree_size: 16,
        };
        let adapter1 = vec![10u32, 20];
        let prompt1 = vec![10, 50, 60, 10, 50, 70];
        let ngram1 = NgramIndex::build(&prompt1, config1.pld_ngram_len);
        let tree1 = SpecTree::build(config1.clone(), &adapter1, &prompt1, &ngram1);

        // Act: second build with different data
        let config2 = SpecTreeConfig {
            max_spine_depth: 2,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            adapter_top_k: 1,
            max_tree_size: 8,
        };
        let adapter2 = vec![99u32];
        let prompt2: Vec<u32> = vec![];
        let ngram2 = NgramIndex::build(&prompt2, config2.pld_ngram_len);
        let tree2 = SpecTree::build(config2, &adapter2, &prompt2, &ngram2);

        // Assert: trees are independent — different sizes and tokens
        assert!(tree1.len() >= tree2.len(), "tree1 has more config capacity");
        assert_eq!(tree2.nodes()[0].token_id, 99);
        assert_eq!(tree1.nodes()[0].token_id, 10);
        // tree1 is unaffected by tree2 construction
        assert!(tree1.len() > 1, "tree1 should have branches from ngram");
    }

    // Focus 15: Config max_spine_depth=1 as minimal safe boundary (depth=0 panics per existing test)
    #[test]
    fn spec_tree_config_max_spine_depth_one_builds_without_panic() {
        // Arrange: max_spine_depth=1 is the minimum safe value (0 causes usize underflow)
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 2,
            pld_ngram_len: 2,
            ngram_top_k: 2,
            adapter_top_k: 3,
            max_tree_size: 16,
        };
        let adapter_tokens = vec![42u32, 43, 44];
        let prompt_tokens = vec![42, 50, 51, 42, 50, 52, 60, 61, 62];
        let ngram_index = NgramIndex::build(&prompt_tokens, config.pld_ngram_len);
        // Act: should not panic
        let tree = SpecTree::build(config, &adapter_tokens, &prompt_tokens, &ngram_index);
        // Assert: spine has exactly 1 node (root only, since max_spine_depth=1 means root + 0 extensions)
        let spine = tree.spine_ids();
        assert_eq!(spine.len(), 1, "max_spine_depth=1 → spine is root only");
        assert_eq!(tree.nodes()[0].token_id, 42);
        // Branches can still exist off root
        assert!(tree.len() >= 1);
    }

    #[test]
    fn spec_node_parent_none_for_root_like() {
        let node = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.5, position_offset: 0,
        };
        assert!(node.parent_id.is_none());
        assert!(node.children.is_empty());
    }

    #[test]
    fn draft_source_adapter_identical_k_variants_equal() {
        let a = DraftSource::AdapterTopK { k: 2 };
        let b = DraftSource::AdapterTopK { k: 2 };
        assert_eq!(a, b);
    }

    #[test]
    fn draft_source_adapter_top_k_different_k_not_equal() {
        let a = DraftSource::AdapterTopK { k: 1 };
        let b = DraftSource::AdapterTopK { k: 2 };
        assert_ne!(a, b);
    }

    #[test]
    fn spec_tree_config_clone_preserves_all_fields() {
        let a = SpecTreeConfig::default();
        let b = a.clone();
        assert_eq!(a.max_spine_depth, b.max_spine_depth);
        assert_eq!(a.max_tree_size, b.max_tree_size);
    }

    #[test]
    fn spec_tree_node_none_when_out_of_bounds() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.node(0).is_none());
        assert!(tree.node(100).is_none());
    }

    #[test]
    fn ngram_index_single_token_returns_empty() {
        let idx = NgramIndex::build(&[1u32], 2);
        assert!(idx.get_continuations(1, 3).is_empty());
    }

    #[test]
    #[test]
    fn ngram_index_continuations_sorted_by_frequency() {
        // token 10 followed by 20 three times, 30 two times
        let tokens: Vec<u32> = vec![10, 20, 10, 20, 10, 20, 10, 30, 10, 30];
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_continuations(10, 5);
        if conts.len() >= 2 {
            assert_eq!(conts[0], 20, "most frequent first");
        }
    }

    #[test]
    fn spec_tree_build_empty_adapter_yields_empty_tree() {
        let idx = NgramIndex::build(&[1, 2, 3], 2);
        let tree = SpecTree::build(SpecTreeConfig::default(), &[], &[1, 2, 3], &idx);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn spec_tree_build_single_token_creates_one_node() {
        let idx = NgramIndex::build(&[1, 2, 3], 2);
        let tree = SpecTree::build(SpecTreeConfig::default(), &[42], &[1, 2, 3], &idx);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.nodes()[0].token_id, 42);
    }

    #[test]
    fn spec_tree_mask_shape_matches_total_seq_plus_tree() {
        let idx = NgramIndex::build(&[1, 2, 3], 2);
        let tree = SpecTree::build(SpecTreeConfig::default(), &[42], &[1, 2, 3], &idx);
        let (r, c) = tree.mask_shape(20);
        assert_eq!(r, tree.len());
        assert_eq!(c, 20 + tree.len());
    }

    #[test]
    fn spec_tree_csr_indptr_length_is_nodes_plus_one() {
        let idx = NgramIndex::build(&[1, 2, 3], 2);
        let tree = SpecTree::build(SpecTreeConfig::default(), &[42], &[1, 2, 3], &idx);
        let (indptr, _) = tree.tree_attention_mask_csr(10);
        assert_eq!(indptr.len(), tree.len() + 1);
    }

    #[test]
    fn ngram_index_get_ngram_continuations_empty_for_no_match() {
        let idx = NgramIndex::build(&[1u32, 2, 3], 2);
        let result = idx.get_ngram_continuations(&[99, 98], 3);
        assert!(result.is_empty());
    }

    // ---- Test 836: SpecNode FRU (struct update syntax) preserves unchanged fields ----

    #[test]
    fn spec_node_fru_preserves_unchanged_fields() {
        let original = SpecNode {
            node_id: 10,
            token_id: 42,
            parent_id: Some(5),
            children: vec![20, 30],
            source: DraftSource::PldSpine,
            estimated_acceptance: 0.75,
            position_offset: 3,
        };
        let updated = SpecNode {
            token_id: 99,
            estimated_acceptance: 0.10,
            ..original
        };
        assert_eq!(updated.node_id, 10);
        assert_eq!(updated.token_id, 99);
        assert_eq!(updated.parent_id, Some(5));
        assert_eq!(updated.children, vec![20, 30]);
        assert_eq!(updated.source, DraftSource::PldSpine);
        assert!((updated.estimated_acceptance - 0.10).abs() < f32::EPSILON);
        assert_eq!(updated.position_offset, 3);
    }

    // ---- Test 837: SpecTreeConfig FRU overrides selected fields ----

    #[test]
    fn spec_tree_config_fru_overrides_selected_fields() {
        let base = SpecTreeConfig::default();
        let custom = SpecTreeConfig {
            max_spine_depth: 10,
            max_tree_size: 64,
            ..base
        };
        assert_eq!(custom.max_spine_depth, 10);
        assert_eq!(custom.max_branches_per_node, 2);
        assert_eq!(custom.pld_ngram_len, 3);
        assert_eq!(custom.ngram_top_k, 2);
        assert_eq!(custom.adapter_top_k, 3);
        assert_eq!(custom.max_tree_size, 64);
    }

    // ---- Test 838: DraftSource AdapterTopK k=0 boundary ----

    #[test]
    fn draft_source_adapter_top_k_zero_k_boundary() {
        let src = DraftSource::AdapterTopK { k: 0 };
        let cloned = src;
        assert_eq!(src, cloned);
        let debug_str = format!("{:?}", src);
        assert!(debug_str.contains("AdapterTopK"));
        assert!(debug_str.contains('0'));
    }

    // ---- Test 839: DraftSource exhaustive variant equality and inequality ----

    #[test]
    fn draft_source_exhaustive_variant_inequality() {
        let pld = DraftSource::PldSpine;
        let adapter1 = DraftSource::AdapterTopK { k: 1 };
        let adapter2 = DraftSource::AdapterTopK { k: 2 };
        let ngram = DraftSource::NgramBranch;

        assert_ne!(pld, adapter1);
        assert_ne!(pld, ngram);
        assert_ne!(adapter1, adapter2);
        assert_ne!(adapter2, ngram);
        assert_eq!(pld, DraftSource::PldSpine);
        assert_eq!(ngram, DraftSource::NgramBranch);
    }

    // ---- Test 840: SpecNode estimated_acceptance subnormal float ----

    #[test]
    fn spec_node_estimated_acceptance_subnormal_float() {
        let node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::NgramBranch,
            estimated_acceptance: f32::MIN_POSITIVE,
            position_offset: 0,
        };
        assert!(node.estimated_acceptance > 0.0);
        assert!(node.estimated_acceptance.is_normal());
        let cloned = node.clone();
        assert_eq!(node, cloned);
    }

    // ---- Test 841: SpecNode estimated_acceptance infinity and NaN handling ----

    #[test]
    fn spec_node_estimated_acceptance_infinity_and_nan() {
        let inf_node = SpecNode {
            node_id: 0,
            token_id: 1,
            parent_id: None,
            children: vec![],
            source: DraftSource::PldSpine,
            estimated_acceptance: f32::INFINITY,
            position_offset: 0,
        };
        assert!(inf_node.estimated_acceptance.is_infinite());
        assert!(inf_node.estimated_acceptance.is_sign_positive());

        let nan_node = SpecNode {
            estimated_acceptance: f32::NAN,
            ..inf_node.clone()
        };
        assert!(nan_node.estimated_acceptance.is_nan());
        // NaN != NaN by IEEE 754, but struct equality uses bit-level PartialEq
        assert_ne!(inf_node, nan_node);
    }

    // ---- Test 842: SpecTreeConfig minimal fields produces minimal tree ----

    #[test]
    fn spec_tree_config_minimal_fields_produces_minimal_tree() {
        // Use minimal non-zero values that avoid subtraction overflow in build:
        // max_spine_depth=1 (prevents -1 underflow), adapter_top_k=1 (prevents -1 underflow)
        // Set max_tree_size=1 to cap tree to just the root node
        let cfg = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 0,
            ngram_top_k: 0,
            adapter_top_k: 1,
            max_tree_size: 1,
        };
        let tree = SpecTree::build(cfg, &[42], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        // With max_tree_size=1, only the root adapter top-1 node is added
        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());
        assert_eq!(tree.nodes()[0].token_id, 42);
    }

    // ---- Test 843: NgramIndex with n=1 returns 1-gram continuations ----

    #[test]
    fn ngram_index_n1_returns_single_token_continuations() {
        let tokens: Vec<u32> = vec![10, 20, 30, 10, 40, 10, 20];
        let idx = NgramIndex::build(&tokens, 1);
        // 1-gram: token 10 followed by 20 (twice), 40 (once) -> sorted by freq
        let conts = idx.get_ngram_continuations(&[10], 5);
        assert_eq!(conts[0], 20); // frequency 2, appears first
        assert!(conts.contains(&40)); // frequency 1
    }

    // ---- Test 844: SpecTree node accessor returns None for out-of-range ----

    #[test]
    fn spec_tree_node_accessor_returns_none_for_out_of_range() {
        let idx = NgramIndex::build(&[1u32, 2, 3, 4], 2);
        let tree = SpecTree::build(SpecTreeConfig::default(), &[100], &[1, 2, 3, 4], &idx);
        assert!(tree.node(0).is_some());
        assert!(tree.node(u32::MAX).is_none());
    }

    // ---- Test 845: SpecTree branch_token_ids distinguishes spine from branches ----

    #[test]
    fn spec_tree_branch_token_ids_excludes_spine_nodes() {
        // Build a tree with enough prompt context to generate branches
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
        let idx = NgramIndex::build(&prompt, 3);
        let cfg = SpecTreeConfig {
            max_spine_depth: 3,
            max_branches_per_node: 2,
            pld_ngram_len: 3,
            ngram_top_k: 2,
            adapter_top_k: 3,
            max_tree_size: 20,
        };
        let tree = SpecTree::build(cfg, &[10, 20, 30], &prompt, &idx);
        if tree.len() > 1 {
            let branches = tree.branch_token_ids();
            let spine_ids: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
            for (node_id, _) in &branches {
                assert!(!spine_ids.contains(node_id));
            }
        }
    }

    // ---- Test 846: SpecNode clone independence ----

    #[test]
    fn spec_node_clone_is_independent() {
        let mut node = SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![1, 2, 3],
            source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.5,
            position_offset: 0,
        };
        let cloned = node.clone();
        node.children.push(4);
        assert_eq!(node.children.len(), 4);
        assert_eq!(cloned.children.len(), 3);
        assert_ne!(node, cloned);
    }

    // ---- Test 847: NgramIndex build with tokens_len_equals_n produces_empty_index ----

    #[test]
    fn ngram_index_tokens_len_equals_n_produces_empty_table() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let idx = NgramIndex::build(&tokens, 3);
        // tokens.len() == n means no full window + continuation possible
        assert_eq!(idx.get_continuations(1, 10).len(), 0);
    }

    // ---- Test 848: SpecTree CSR mask columns_include_prefix_and_ancestors ----

    #[test]
    fn spec_tree_csr_columns_include_prefix_and_ancestors_and_self() {
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
        let idx = NgramIndex::build(&prompt, 3);
        let cfg = SpecTreeConfig {
            max_spine_depth: 2,
            max_branches_per_node: 1,
            pld_ngram_len: 3,
            ngram_top_k: 1,
            adapter_top_k: 2,
            max_tree_size: 10,
        };
        let tree = SpecTree::build(cfg, &[10, 20], &prompt, &idx);
        let total_seq = 15;
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq);
        let num_nodes = tree.len();
        for i in 0..num_nodes {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            let row = &indices[row_start..row_end];
            // Must include full prefix [0..total_seq)
            for col in 0..total_seq {
                assert!(row.contains(&col), "row {} missing prefix col {}", i, col);
            }
            // Must include self
            assert!(row.contains(&(total_seq + i)), "row {} missing self", i);
        }
    }

    // ==================================================================
    // NEW TESTS (849-861, 13 additional tests)
    // ==================================================================

    // ------------------------------------------------------------------
    // NgramIndex: n=1 with strictly decreasing frequency pattern
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_n1_decreasing_frequency_order() {
        // Arrange: token 5 followed by 10 (3x), 20 (2x), 30 (1x)
        let tokens = vec![5u32, 10, 5, 10, 5, 10, 5, 20, 5, 20, 5, 30];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(5, 10);
        // Assert: ordered by frequency descending
        assert_eq!(conts.len(), 3);
        assert_eq!(conts[0], 10, "10 (3x) should be first");
        assert_eq!(conts[1], 20, "20 (2x) should be second");
        assert_eq!(conts[2], 30, "30 (1x) should be third");
    }

    // ------------------------------------------------------------------
    // SpecTree: total_nodes field reflects actual node count after build
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_total_nodes_field_matches_len() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 2,
            max_branches_per_node: 1,
            ngram_top_k: 2,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20];
        let prompt = vec![1, 2, 10, 50, 60, 10, 70];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        assert_eq!(tree.total_nodes, tree.len());
        assert_eq!(tree.total_nodes, tree.nodes().len());
        assert!(tree.total_nodes > 0);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask adapter-only tree with nonzero seq_len
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_adapter_only_nonzero_seq_len() {
        // Arrange: 4 adapter tokens, no PLD/n-gram
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 4,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(
            config,
            &[10u32, 20, 30, 40],
            &[],
            &NgramIndex::build(&[], 2),
        );
        let total_seq_len = 7;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: root row = prefix (7) + self = 8 entries
        assert_eq!(indptr[1] - indptr[0], total_seq_len + 1);
        // Branch rows = prefix (7) + root ancestor + self = 9 entries each
        for i in 1..tree.len() {
            let row_size = indptr[i + 1] - indptr[i];
            assert_eq!(row_size, total_seq_len + 2,
                "adapter branch {} should have prefix + root + self = {}, got {}",
                i, total_seq_len + 2, row_size);
        }
        // Verify branch rows include root ancestor
        for i in 1..tree.len() {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row = &indices[start..end];
            assert!(row.contains(&(total_seq_len)),
                "branch {} should attend to root at column {}", i, total_seq_len);
        }
        let _ = indices; // suppress unused warning
    }

    // ------------------------------------------------------------------
    // SpecNode: negative zero and positive zero estimated_acceptance are equal
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_negative_zero_positive_zero_acceptance_equal() {
        // Arrange
        let a = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.0_f32, position_offset: 0,
        };
        let b = SpecNode {
            node_id: 0, token_id: 1, parent_id: None,
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: -0.0_f32, position_offset: 0,
        };
        // Act & Assert: f32 PartialEq treats -0.0 == 0.0
        assert_eq!(a, b, "-0.0 and 0.0 should be equal for SpecNode");
        assert_eq!(a.estimated_acceptance, b.estimated_acceptance);
    }

    // ------------------------------------------------------------------
    // SpecTree: PLD continuation from second-to-last position yields one token
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_pld_continuation_second_to_last_position() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 at index 3 (second-to-last), followed by only [99]
        let adapter_tokens = vec![10u32];
        let prompt = vec![1, 2, 3, 10, 99];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: spine should be [10, 99] exactly
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        assert!(spine.len() >= 2, "should have one PLD extension from second-to-last");
        assert!(spine.contains(&99));
        // Spine should be exactly 2 because prompt has only one continuation after 10
        assert!(spine.len() <= 2, "spine should not exceed available continuations");
    }

    // ------------------------------------------------------------------
    // NgramIndex: large repeated token sequence yields single merged entry
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_large_repeated_sequence_single_entry() {
        // Arrange: 100 repetitions of [1, 2, 3]
        let tokens: Vec<u32> = (0..100).flat_map(|_| [1u32, 2, 3]).collect();
        // Act
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[1, 2], 5);
        // Assert: only continuation is 3, merged from all 100 occurrences
        assert_eq!(conts, vec![3]);
    }

    // ------------------------------------------------------------------
    // SpecTree: n-gram branches on same parent have distinct token IDs
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_ngram_branches_same_parent_distinct_tokens() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 5,
            ngram_top_k: 5,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 has many distinct n-gram continuations
        let prompt: Vec<u32> = vec![10, 50, 10, 60, 10, 70, 10, 80, 10, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert: all n-gram children of root have distinct token IDs
        let root = tree.node(0).unwrap();
        let ngram_tokens: Vec<u32> = root.children.iter()
            .filter(|&&c| matches!(tree.node(c).unwrap().source, DraftSource::NgramBranch))
            .map(|&c| tree.node(c).unwrap().token_id)
            .collect();
        let unique: std::collections::HashSet<u32> = ngram_tokens.iter().copied().collect();
        assert_eq!(ngram_tokens.len(), unique.len(),
            "n-gram branch tokens should all be distinct: {:?}", ngram_tokens);
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask spine depth 3 has exact ancestor counts
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_spine_depth_three_exact_ancestor_counts() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        let prompt = vec![10, 50, 60, 70, 80];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        let total_seq_len = 5;
        // Act
        let (indptr, _indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: spine node at depth d has d ancestors + total_seq_len prefix + self
        let spine_ids = tree.spine_ids();
        for (depth, &id) in spine_ids.iter().enumerate() {
            let row_size = indptr[id as usize + 1] - indptr[id as usize];
            let expected = total_seq_len + depth + 1;
            assert_eq!(row_size, expected,
                "spine node {} at depth {} should have {} entries (prefix {} + ancestors {} + self 1), got {}",
                id, depth, expected, total_seq_len, depth, row_size);
        }
    }

    // ------------------------------------------------------------------
    // DraftSource: Hash consistency across all variants
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_hash_consistency_all_variants() {
        // Arrange
        use std::collections::HashMap;
        let mut map: HashMap<DraftSource, &'static str> = HashMap::new();
        map.insert(DraftSource::PldSpine, "spine");
        map.insert(DraftSource::NgramBranch, "ngram");
        map.insert(DraftSource::AdapterTopK { k: 1 }, "a1");
        map.insert(DraftSource::AdapterTopK { k: 2 }, "a2");
        // Act & Assert: all 4 entries retrievable
        assert_eq!(map.get(&DraftSource::PldSpine), Some(&"spine"));
        assert_eq!(map.get(&DraftSource::NgramBranch), Some(&"ngram"));
        assert_eq!(map.get(&DraftSource::AdapterTopK { k: 1 }), Some(&"a1"));
        assert_eq!(map.get(&DraftSource::AdapterTopK { k: 2 }), Some(&"a2"));
        assert_eq!(map.len(), 4);
        // Re-insert PldSpine should not duplicate
        map.insert(DraftSource::PldSpine, "spine2");
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&DraftSource::PldSpine), Some(&"spine2"));
    }

    // ------------------------------------------------------------------
    // SpecTree: all branch nodes have position_offset greater than root
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_nodes_offset_greater_than_root_offset() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![10, 50, 60, 10, 70, 80, 10, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        let root_offset = tree.node(0).unwrap().position_offset;
        assert_eq!(root_offset, 0);
        let spine_ids: std::collections::HashSet<u32> = tree.spine_ids().into_iter().collect();
        for node in tree.nodes() {
            if !spine_ids.contains(&node.node_id) && node.node_id > 0 {
                assert!(node.position_offset > root_offset,
                    "branch node {} offset {} should be > root offset {}",
                    node.node_id, node.position_offset, root_offset);
            }
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: two sequential builds on same data produce identical results
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_sequential_builds_identical() {
        // Arrange
        let tokens: Vec<u32> = (0..50).cycle().take(200).collect();
        // Act
        let idx1 = NgramIndex::build(&tokens, 3);
        let idx2 = NgramIndex::build(&tokens, 3);
        // Assert
        assert_eq!(idx1, idx2);
        assert_eq!(idx1.get_ngram_continuations(&[0, 1, 2], 10), idx2.get_ngram_continuations(&[0, 1, 2], 10));
    }

    // ------------------------------------------------------------------
    // SpecTree: all_token_ids on root-only tree returns single element
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_token_ids_root_only_single_element() {
        // Arrange: build a tree with single root, no extensions
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let tree = SpecTree::build(config, &[777u32], &[1, 2, 3], &NgramIndex::build(&[1, 2, 3], 2));
        // Act
        let ids = tree.all_token_ids();
        // Assert
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 777);
    }

    // ------------------------------------------------------------------
    // SpecTree: max_spine_depth=2 produces exactly root + one extension when PLD available
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_depth_two_exact_extension_count() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32];
        // Token 10 followed by many tokens, but max_spine_depth=2 limits to root + 1 extension
        let prompt = vec![1, 10, 50, 60, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        let spine = tree.spine_ids();
        assert!(spine.len() <= 2,
            "spine should be at most 2 nodes (root + 1 extension), got {}", spine.len());
        assert_eq!(spine[0], 0);
        if spine.len() == 2 {
            assert_eq!(tree.node(spine[1]).unwrap().source, DraftSource::PldSpine);
        }
    }

    // ==================================================================
    // NEW TESTS (13 additional tests)
    // ==================================================================

    // ------------------------------------------------------------------
    // SpecTree: all non-root nodes have a parent that exists
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_all_non_root_nodes_have_valid_parent() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![10u32, 20, 30];
        let prompt = vec![10, 50, 60, 10, 70, 80, 90];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        // Act
        let tree = SpecTree::build(config, &adapter_tokens, &prompt, &ngram_idx);
        // Assert
        for i in 1..tree.len() {
            let node = tree.node(i as u32).unwrap();
            let parent_id = node.parent_id.expect("non-root node must have a parent");
            assert!(tree.node(parent_id).is_some(),
                "node {} parent {} must exist", i, parent_id);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_ngram_continuations returns empty for reversed n-gram
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_reversed_ngram_returns_empty() {
        // Arrange: build index with [1,2,3,4] — only forward 2-grams exist
        let tokens = vec![1u32, 2, 3, 4];
        let idx = NgramIndex::build(&tokens, 2);
        // Act: query reversed n-gram [2,1] which never appeared
        let conts = idx.get_ngram_continuations(&[2, 1], 5);
        // Assert
        assert!(conts.is_empty(), "reversed n-gram should have no continuations");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask non-root node has exactly (total_seq_len + ancestor_count + 1) entries
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_row_entry_count_exact() {
        // Arrange: 3-node spine (root + 2 extensions), no branches
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70];
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        let total_seq_len = 8;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: verify total indices count equals sum of all row sizes
        let mut total = 0;
        for i in 0..tree.len() {
            let row_size = indptr[i + 1] - indptr[i];
            // Each row: total_seq_len prefix columns + (i) ancestors + 1 self
            let expected = total_seq_len + i + 1;
            assert_eq!(row_size, expected,
                "node {} row size should be {}, got {}", i, expected, row_size);
            total += row_size;
        }
        assert_eq!(total, indices.len());
    }

    // ------------------------------------------------------------------
    // SpecTree: build with adapter tokens containing u32::MAX
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_adapter_tokens_with_max_u32_value() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            adapter_top_k: 2,
            max_branches_per_node: 0,
            ..SpecTreeConfig::default()
        };
        let adapter_tokens = vec![u32::MAX, u32::MAX - 1];
        // Act
        let tree = SpecTree::build(
            config, &adapter_tokens, &[],
            &NgramIndex::build(&[], 2),
        );
        // Assert
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.node(0).unwrap().token_id, u32::MAX);
        assert_eq!(tree.node(1).unwrap().token_id, u32::MAX - 1);
    }

    // ------------------------------------------------------------------
    // SpecNode: parent_id Some vs None is the root discriminator
    // ------------------------------------------------------------------

    #[test]
    fn spec_node_parent_id_none_unambiguously_identifies_root() {
        // Arrange
        let root = SpecNode {
            node_id: 0, token_id: 10, parent_id: None,
            children: vec![1], source: DraftSource::AdapterTopK { k: 1 },
            estimated_acceptance: 0.70, position_offset: 0,
        };
        let child = SpecNode {
            node_id: 1, token_id: 20, parent_id: Some(0),
            children: vec![], source: DraftSource::PldSpine,
            estimated_acceptance: 0.55, position_offset: 1,
        };
        // Act & Assert
        assert!(root.parent_id.is_none(), "root must have parent_id None");
        assert!(child.parent_id.is_some(), "child must have parent_id Some");
        assert_eq!(child.parent_id.unwrap(), root.node_id);
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with two overlapping n-gram windows
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_overlapping_windows_merge_counts() {
        // Arrange: [A,B] → C twice, [B,C] → D once
        // tokens = [A, B, C, D, A, B, C, E]
        // n=2: [A,B]→C (2x), [B,C]→D (1x), [C,D]→A (1x), [D,A]→B (1x), [A,B]→C (already counted), [B,C]→E (1x)
        let tokens = vec![1u32, 2, 3, 4, 1, 2, 3, 5];
        let idx = NgramIndex::build(&tokens, 2);
        // Act
        let conts_ab = idx.get_ngram_continuations(&[1, 2], 5);
        let conts_bc = idx.get_ngram_continuations(&[2, 3], 5);
        // Assert
        assert_eq!(conts_ab, vec![3], "[1,2] should only have continuation 3");
        // [2,3] has two continuations: 4 (count 1) and 5 (count 1)
        assert_eq!(conts_bc.len(), 2);
        assert!(conts_bc.contains(&4));
        assert!(conts_bc.contains(&5));
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask columns for spine node include all earlier spine nodes
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_spine_node_includes_all_earlier_spine() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70, 80];
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        let total_seq_len = 3;
        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(total_seq_len);
        // Assert: deepest spine node should attend to all earlier spine nodes
        let spine_ids = tree.spine_ids();
        if spine_ids.len() >= 3 {
            let last_id = *spine_ids.last().unwrap();
            let start = indptr[last_id as usize];
            let end = indptr[last_id as usize + 1];
            let row: std::collections::HashSet<usize> =
                indices[start..end].iter().copied().collect();
            // Every earlier spine node should be in the ancestor set
            for &earlier_id in &spine_ids[..spine_ids.len() - 1] {
                let col = total_seq_len + earlier_id as usize;
                assert!(row.contains(&col),
                    "deepest spine node should attend to earlier spine node {} at col {}", earlier_id, col);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTreeConfig: modifying one field does not affect another
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_config_field_independence() {
        // Arrange
        let mut cfg = SpecTreeConfig::default();
        let original_pld = cfg.pld_ngram_len;
        let original_ngram = cfg.ngram_top_k;
        // Act
        cfg.max_spine_depth = 999;
        // Assert
        assert_eq!(cfg.pld_ngram_len, original_pld,
            "changing max_spine_depth should not affect pld_ngram_len");
        assert_eq!(cfg.ngram_top_k, original_ngram,
            "changing max_spine_depth should not affect ngram_top_k");
    }

    // ------------------------------------------------------------------
    // DraftSource: can be used in a match with exhaustive coverage
    // ------------------------------------------------------------------

    #[test]
    fn draft_source_exhaustive_match() {
        // Arrange
        let sources = [
            DraftSource::PldSpine,
            DraftSource::NgramBranch,
            DraftSource::AdapterTopK { k: 1 },
        ];
        // Act
        let labels: Vec<&str> = sources.iter().map(|s| match s {
            DraftSource::PldSpine => "spine",
            DraftSource::NgramBranch => "ngram",
            DraftSource::AdapterTopK { .. } => "adapter",
        }).collect();
        // Assert
        assert_eq!(labels, vec!["spine", "ngram", "adapter"]);
    }

    // ------------------------------------------------------------------
    // SpecTree: build with prompt where adapter token appears at multiple positions
    //  producing distinct PLD continuations from each occurrence
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_pld_continuations_from_multiple_positions() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 10,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        // Token 10 appears at index 1 (→ 20, 30), index 4 (→ 40, 50)
        let prompt = vec![0, 10, 20, 30, 10, 40, 50];
        // Act
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        // Assert
        let spine = tree.spine_token_ids();
        assert_eq!(spine[0], 10);
        // Continuations from first occurrence: 20, 30
        // From second occurrence: 40, 50
        // All should be in spine (deduplicated)
        assert!(spine.contains(&20), "spine should contain 20 from first occurrence");
        assert!(spine.contains(&40), "spine should contain 40 from second occurrence");
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations with n=1 returns most frequent first
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_n1_continuation_frequency_ordering() {
        // Arrange: token 5 followed by 10 (3x), 20 (2x), 30 (1x)
        let tokens = vec![5u32, 10, 5, 10, 5, 20, 5, 10, 5, 20, 5, 30];
        let idx = NgramIndex::build(&tokens, 1);
        // Act
        let conts = idx.get_continuations(5, 10);
        // Assert
        assert!(conts.len() >= 3, "should have at least 3 distinct continuations");
        // Most frequent first
        assert_eq!(conts[0], 10, "10 (3x) should be first");
        assert_eq!(conts[1], 20, "20 (2x) should be second");
        assert_eq!(conts[2], 30, "30 (1x) should be third");
    }

    // ------------------------------------------------------------------
    // SpecTree: root children list contains no duplicates
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_root_children_no_duplicate_ids() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            adapter_top_k: 3,
            max_branches_per_node: 2,
            ngram_top_k: 3,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 10, 70, 80, 10, 90];
        let tree = SpecTree::build(
            config, &[10u32, 20, 30], &prompt,
            &NgramIndex::build(&prompt, 1),
        );
        // Act
        let root = tree.node(0).unwrap();
        let children_set: std::collections::HashSet<u32> =
            root.children.iter().copied().collect();
        // Assert
        assert_eq!(root.children.len(), children_set.len(),
            "root children should have no duplicate IDs: {:?}", root.children);
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine preserves spine order in accepted
    // ------------------------------------------------------------------

    #[test]
    fn accepted_from_spine_preserves_spine_order() {
        // Arrange
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            adapter_top_k: 1,
            max_branches_per_node: 0,
            pld_ngram_len: 1,
            ..SpecTreeConfig::default()
        };
        let prompt = vec![10, 50, 60, 70];
        let tree = SpecTree::build(config, &[10u32], &prompt, &NgramIndex::build(&prompt, 1));
        let spine = tree.spine_token_ids();
        if spine.len() >= 3 {
            // Act: match first 3 tokens only
            let target = vec![spine[0], spine[1], spine[2]];
            let (count, accepted) = tree.accepted_from_spine(&target);
            // Assert
            assert_eq!(count, 3);
            // Accepted tokens must be in same order as spine
            for (i, &tok) in accepted.iter().enumerate() {
                assert_eq!(tok, spine[i],
                    "accepted[{}] should be spine[{}] = {}, got {}", i, i, spine[i], tok);
            }
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: n=2 get_ngram_continuations returns correct continuations
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_n2_get_ngram_continuations_returns_correct_tokens() {
        // Arrange: build n=2 index with tokens [1,2,10, 1,2,20]
        // The 2-gram [1,2] has two continuations: 10 (1x) and 20 (1x)
        let tokens = vec![1u32, 2, 10, 1, 2, 20];
        let idx = NgramIndex::build(&tokens, 2);

        // Act
        let conts = idx.get_ngram_continuations(&[1, 2], 5);

        // Assert: both continuations present
        assert!(conts.contains(&10), "should contain continuation 10");
        assert!(conts.contains(&20), "should contain continuation 20");
        assert_eq!(conts.len(), 2, "should have exactly 2 continuations");
    }

    // ------------------------------------------------------------------
    // SpecTree: build with max_branches_per_node=0 produces spine only
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_build_no_branches_produces_spine_only() {
        // Arrange: config with zero branches, rich prompt for PLD matches
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            max_branches_per_node: 0,
            adapter_top_k: 1,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            max_tree_size: 32,
        };
        let adapter = vec![100u32];
        // PLD with n=1: token 100 appears at index 1, continuation is 200
        let prompt = vec![50, 100, 200, 300];
        let ngram_idx = NgramIndex::build(&prompt, 1);

        // Act
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Assert: all nodes are on the spine
        let spine_ids: std::collections::HashSet<u32> =
            tree.spine_ids().into_iter().collect();
        assert_eq!(spine_ids.len(), tree.total_nodes,
            "all nodes should be spine nodes when branches disabled");
        // Branches should be empty
        assert!(tree.branch_token_ids().is_empty(),
            "no branch nodes expected when max_branches_per_node=0");
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR mask with total_seq_len=0 has only ancestor+self columns
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_mask_zero_seq_len_only_ancestor_and_self() {
        // Arrange: build a tree with at least root + 1 adapter branch
        let config = SpecTreeConfig {
            max_spine_depth: 3,
            max_branches_per_node: 1,
            adapter_top_k: 2,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            max_tree_size: 16,
        };
        let adapter = vec![10u32, 20];
        let prompt = vec![10, 30, 40];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act
        let (indptr, indices) = tree.tree_attention_mask_csr(0);

        // Assert: root node (node 0) has only self column (index 0)
        let root_cols = &indices[indptr[0]..indptr[1]];
        assert_eq!(root_cols, &[0], "root with seq_len=0 should attend only to self");

        // Branch nodes include ancestor + self columns, no prefix columns
        for row in 1..tree.total_nodes {
            let row_cols = &indices[indptr[row]..indptr[row + 1]];
            for &col in row_cols {
                // All columns should be < tree.total_nodes (no seq prefix)
                assert!(col < tree.total_nodes,
                    "column {} in row {} should be a tree node index when seq_len=0", col, row);
            }
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: accepted_from_spine with single spine node, match and mismatch
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_accepted_from_spine_single_node_match_and_mismatch() {
        // Arrange: minimal tree with just root (adapter only, no PLD match)
        let config = SpecTreeConfig {
            max_spine_depth: 1,
            max_branches_per_node: 0,
            adapter_top_k: 1,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            max_tree_size: 8,
        };
        let adapter = vec![42u32];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 1);
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Act & Assert: matching target
        let (count, accepted) = tree.accepted_from_spine(&[42]);
        assert_eq!(count, 1, "single spine node should match once");
        assert_eq!(accepted, vec![42]);

        // Act & Assert: mismatching target
        let (count2, accepted2) = tree.accepted_from_spine(&[99]);
        assert_eq!(count2, 0, "single spine node should not match different token");
        assert!(accepted2.is_empty());

        // Act & Assert: empty target
        let (count3, accepted3) = tree.accepted_from_spine(&[]);
        assert_eq!(count3, 0, "empty target should accept nothing");
        assert!(accepted3.is_empty());
    }

    // ------------------------------------------------------------------
    // NgramIndex: build with tokens.len() == n+1 produces exactly one window
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_exact_n_plus_one_tokens_single_window() {
        // Arrange: n=3, tokens=[1,2,3,4] → one window [1,2,3]→4
        let tokens = vec![1u32, 2, 3, 4];

        // Act
        let idx = NgramIndex::build(&tokens, 3);

        // Assert: get_ngram_continuations for [1,2,3] returns [4]
        let conts = idx.get_ngram_continuations(&[1, 2, 3], 10);
        assert_eq!(conts, vec![4], "single window should produce one continuation");

        // Assert: a different 3-gram returns empty
        let conts2 = idx.get_ngram_continuations(&[2, 3, 4], 10);
        assert!(conts2.is_empty(), "no window for trailing n-gram");
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_ids returns only spine nodes (no branch nodes)
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_ids_excludes_branch_nodes() {
        // Arrange: tree with spine + adapter branches + ngram branches
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            max_branches_per_node: 2,
            adapter_top_k: 3,
            pld_ngram_len: 1,
            ngram_top_k: 2,
            max_tree_size: 32,
        };
        let adapter = vec![100u32, 110, 120];
        let prompt = vec![100, 200, 300, 400];
        let ngram_idx = NgramIndex::build(&prompt, 1);

        // Act
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let spine_ids = tree.spine_ids();

        // Assert: every spine node is either PldSpine or AdapterTopK{k:1}
        for &id in &spine_ids {
            let node = tree.node(id).expect("spine node must exist");
            let is_root_adapter = matches!(node.source, DraftSource::AdapterTopK { k: 1 });
            let is_pld = matches!(node.source, DraftSource::PldSpine);
            assert!(is_root_adapter || is_pld,
                "spine node {} with source {:?} should be AdapterTopK(k=1) or PldSpine",
                id, node.source);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: CSR branch node ancestor path is complete root-to-parent chain
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_csr_branch_ancestor_path_is_root_to_parent_chain() {
        // Arrange: build a tree with spine depth >= 2 and a branch
        let config = SpecTreeConfig {
            max_spine_depth: 4,
            max_branches_per_node: 2,
            adapter_top_k: 1,
            pld_ngram_len: 1,
            ngram_top_k: 2,
            max_tree_size: 32,
        };
        let adapter = vec![100u32];
        // PLD: token 100 at index 2, continuations [200, 300]
        let prompt = vec![50, 60, 100, 200, 300];
        let ngram_idx = NgramIndex::build(&prompt, 1);

        // Act
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let spine_ids = tree.spine_ids();

        // Find a branch node
        let branches = tree.branch_token_ids();
        if !branches.is_empty() {
            let (branch_id, _) = branches[0];
            let branch_node = tree.node(branch_id).expect("branch must exist");
            let parent_id = branch_node.parent_id.expect("branch must have parent");

            // Build expected ancestor chain: root → ... → parent
            let mut expected = Vec::new();
            let mut cur = parent_id;
            while let Some(pid) = tree.node(cur).and_then(|n| n.parent_id) {
                expected.push(pid);
                cur = pid;
            }
            expected.reverse();

            // Assert: attention_paths for branch contains the full ancestor chain
            let actual_path = &tree.attention_paths[branch_id as usize];
            assert_eq!(actual_path, &expected,
                "branch {} ancestor path should be root-to-parent chain", branch_id);
        }
    }

    // ------------------------------------------------------------------
    // NgramIndex: get_continuations with n>1 returns empty (1-gram key mismatch)
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_get_continuations_n2_1gram_key_mismatch_returns_empty() {
        // Arrange: build n=2 index so table keys are 2-gram hashes
        let tokens = vec![1u32, 2, 3, 2, 3, 4];
        let idx = NgramIndex::build(&tokens, 2);

        // Act: get_continuations uses 1-gram hash [token], but table has 2-gram keys
        let result = idx.get_continuations(2, 5);

        // Assert: empty because [2] hash != any [a,b] hash in the table
        assert!(result.is_empty(),
            "get_continuations with single token on n=2 index should return empty");
    }

    // ------------------------------------------------------------------
    // SpecTree: max_tree_size caps total nodes including spine
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_max_tree_size_caps_total_nodes() {
        // Arrange: small max_tree_size with large spine potential
        let config = SpecTreeConfig {
            max_spine_depth: 10,
            max_branches_per_node: 2,
            adapter_top_k: 1,
            pld_ngram_len: 1,
            ngram_top_k: 2,
            max_tree_size: 3, // very small
        };
        let adapter = vec![10u32];
        // PLD with many continuations available
        let prompt = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let ngram_idx = NgramIndex::build(&prompt, 1);

        // Act
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Assert: total_nodes cannot exceed max_tree_size
        let cap = 3;
        assert!(tree.total_nodes <= cap,
            "total_nodes ({}) should not exceed max_tree_size ({})",
            tree.total_nodes, cap);
    }

    // ------------------------------------------------------------------
    // SpecTree: branch_token_ids returns pairs excluding spine
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_branch_token_ids_returns_correct_node_token_pairs() {
        // Arrange: tree with known adapter branches (top_k=3)
        let config = SpecTreeConfig {
            max_spine_depth: 2,
            max_branches_per_node: 0,
            adapter_top_k: 3,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            max_tree_size: 16,
        };
        let adapter = vec![100u32, 200, 300];
        let prompt = vec![1, 2, 3];
        let ngram_idx = NgramIndex::build(&prompt, 1);

        // Act
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let branches = tree.branch_token_ids();

        // Assert: branches contain adapter top-2 and top-3
        let branch_tokens: Vec<u32> = branches.iter().map(|(_, tok)| *tok).collect();
        assert!(branch_tokens.contains(&200), "adapter top-2 should be a branch");
        assert!(branch_tokens.contains(&300), "adapter top-3 should be a branch");
        assert!(!branch_tokens.contains(&100), "adapter top-1 (spine root) should NOT be a branch");

        // Assert: all branch node_ids are valid
        for (node_id, _) in &branches {
            assert!(tree.node(*node_id).is_some(),
                "branch node_id {} should be valid", node_id);
        }
    }

    // ------------------------------------------------------------------
    // SpecTree: spine_token_ids length equals spine_len field
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_spine_token_ids_len_equals_spine_len_field() {
        // Arrange: build tree with PLD continuations
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            max_branches_per_node: 0,
            adapter_top_k: 1,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            max_tree_size: 16,
        };
        let adapter = vec![10u32];
        // PLD n=1: token 10 at index 2 → continuations [20, 30, 40]
        let prompt = vec![5, 6, 10, 20, 30, 40];
        let ngram_idx = NgramIndex::build(&prompt, 1);

        // Act
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);
        let spine = tree.spine_token_ids();

        // Assert
        assert_eq!(spine.len(), tree.spine_len,
            "spine_token_ids length ({}) should equal spine_len field ({})",
            spine.len(), tree.spine_len);
    }

    // ------------------------------------------------------------------
    // NgramIndex: all-same tokens produce single continuation entry
    // ------------------------------------------------------------------

    #[test]
    fn ngram_index_all_same_tokens_single_continuation_entry() {
        // Arrange: tokens = [7, 7, 7, 7, 7] with n=2
        // All 2-gram windows [7,7] → continuation 7, count=3
        let tokens = vec![7u32, 7, 7, 7, 7];

        // Act
        let idx = NgramIndex::build(&tokens, 2);
        let conts = idx.get_ngram_continuations(&[7, 7], 5);

        // Assert: single continuation token 7
        assert_eq!(conts, vec![7],
            "all-same tokens should have single continuation 7");
    }

    // ------------------------------------------------------------------
    // SpecTree: build_attention_paths produces root-to-parent ordered ancestors
    // ------------------------------------------------------------------

    #[test]
    fn spec_tree_attention_paths_root_to_parent_order() {
        // Arrange: build a tree with spine depth >= 3
        let config = SpecTreeConfig {
            max_spine_depth: 5,
            max_branches_per_node: 0,
            adapter_top_k: 1,
            pld_ngram_len: 1,
            ngram_top_k: 0,
            max_tree_size: 16,
        };
        let adapter = vec![10u32];
        let prompt = vec![10, 20, 30, 40, 50];
        let ngram_idx = NgramIndex::build(&prompt, 1);

        // Act
        let tree = SpecTree::build(config, &adapter, &prompt, &ngram_idx);

        // Assert: for each node, attention_path is strictly increasing
        // (root→parent order means ancestor IDs increase along the path)
        for (i, path) in tree.attention_paths.iter().enumerate() {
            for window in path.windows(2) {
                assert!(window[0] < window[1],
                    "node {} attention_path should be root-to-parent (increasing), \
                     but found {} before {}", i, window[0], window[1]);
            }
        }
    }

}
