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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftSource {
    /// PLD spine: 从 prompt 中 n-gram 匹配延续 (接受率 60-80%)
    PldSpine,
    /// Adapter top-k: 模型 adapter 输出的 top-k 候选 (接受率 70%+ for top-1)
    AdapterTopK { k: u8 },
    /// N-gram recurrence: 从 prompt 频率表提取的替代候选 (接受率 5-15%)
    NgramBranch,
}

/// 推测树配置
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
        let cols = total_seq_len + tree_size;
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
#[derive(Debug, Clone)]
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

        let (indptr, indices) = tree.tree_attention_mask_csr(10);
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
        let (count, accepted) = tree.accepted_from_spine(&target);
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
}
