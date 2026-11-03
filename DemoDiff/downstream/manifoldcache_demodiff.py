import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusion_model import GraphDiffusionTransformer
from models.transformer import precompute_freqs_cis

from diffusion.diffusion_utils import sample_discrete_features, sample_discrete_feature_noise, reverse_diffusion
from utils.data_io import PlaceHolder
from utils.molecule import convert_graph_to_smiles


def compute_tstar(noise_schedule):
    safe = torch.nonzero(noise_schedule.alphas_bar >= 0.5).flatten()
    return int(safe.max()) if safe.numel() > 0 else 0


class CachedTransformerForward:
    def __init__(self, transformer, total_timesteps=500,
                 shallow_stride=2, deep_stride=3,
                 warmup_frac=0.10, cooldown_frac=0.05, tstar=None):
        self.transformer = transformer
        self.T = total_timesteps
        self.shallow_stride = shallow_stride
        self.deep_stride = deep_stride
        self.warmup_frac = warmup_frac
        self.cooldown_frac = cooldown_frac
        self.tstar = tstar

        self.n_blocks = len(transformer.blocks)

        # Per-block caches
        self.block_cache = {}        # block_idx -> cached output tensor
        self.block_last_step = {}    # block_idx -> last step_idx when computed

        # Stats
        self.total_block_evals = 0
        self.cached_block_evals = 0

    def get_stride(self, block_idx, step_idx, unconditioned=False):
        """
        Per-block stride. Deeper blocks (higher idx) get longer strides.
        Unconditioned pass gets 2x stride (simpler input dynamics).
        """
        progress = 1.0 - (step_idx / self.T)  # 0 at start, 1 at end

        # No caching in warmup/cooldown
        if self.tstar is not None:
            if step_idx + 1 > self.tstar:  # the network denoises x_t with t = step_idx + 1
                return 1
        elif progress < self.warmup_frac:
            return 1
        if progress > (1.0 - self.cooldown_frac):
            return 1

        depth = block_idx / max(self.n_blocks - 1, 1)  # 0=first, 1=last
        base = self.shallow_stride + (self.deep_stride - self.shallow_stride) * depth

        # Unconditioned pass: double the stride (it's more stable)
        if unconditioned:
            base *= 2

        return max(1, int(base))

    def should_compute(self, block_idx, step_idx, unconditioned=False):
        """Check if block needs recomputation at this step."""
        key = (block_idx, unconditioned)
        last = self.block_last_step.get(key, -999)
        stride = self.get_stride(block_idx, step_idx, unconditioned)
        steps_since = last - step_idx  # step_idx counts down
        return steps_since >= stride or last < 0

    def reset(self):
        self.block_cache.clear()
        self.block_last_step.clear()
        self.total_block_evals = 0
        self.cached_block_evals = 0

    def forward(self, X_in, E_in, pos_in, relations, node_mask, target_n_nodes, t,
                step_idx=0, unconditioned=False):
        """
        Cached version of Transformer.forward().
        Identical signature plus step_idx and unconditioned flag.
        """
        model = self.transformer
        bs, n, _ = X_in.size()

        # Padding (same as original)
        left_E_dim = model.E_total_dim - E_in.size(-1)
        if left_E_dim > 0:
            E_in = F.pad(E_in, (0, left_E_dim), mode="constant", value=0)
        left_pos_dim = model.pos_total_dim - pos_in.size(-1)
        if left_pos_dim > 0:
            pos_in = F.pad(pos_in, (0, left_pos_dim), mode="constant", value=0)

        # Embeddings (always computed — they're cheap)
        x_atom = model.atom_embedder(X_in)
        x_bond = model.bond_embedder(E_in)
        x_pos = model.pos_embedder(pos_in)
        x = torch.cat([x_atom, x_bond + x_pos], dim=-1)

        r_emb = model.r_embedder(relations)
        t_emb = model.t_embedder(t).unsqueeze(1)
        c = t_emb + r_emb

        freqs_cis = precompute_freqs_cis(model.head_dim, relations)

        # ── Transformer blocks with stride caching ──
        for i, block in enumerate(model.blocks):
            cache_key = (i, unconditioned)
            self.total_block_evals += 1

            if not self.should_compute(i, step_idx, unconditioned) and cache_key in self.block_cache:
                x = self.block_cache[cache_key]
                self.cached_block_evals += 1
            else:
                x = block(x, c, freqs_cis, node_mask['all'])
                self.block_cache[cache_key] = x
                self.block_last_step[cache_key] = step_idx

        # Final layer (always computed — it's the output head)
        X, E, pos, y = model.final_layer(x, X_in, E_in, pos_in, c, node_mask['target'], target_n_nodes)
        out = PlaceHolder(X=X, E=E, y=y, pos=pos).mask(node_mask['target'])
        return out.X, out.E, out.y, out.pos

    def print_stats(self):
        rate = self.cached_block_evals / max(self.total_block_evals, 1) * 100
        print(f"  ManifoldCache: {rate:.1f}% blocks cached "
              f"({self.cached_block_evals}/{self.total_block_evals})")


class CachedGraphDiffusionTransformer(GraphDiffusionTransformer):
    def __init__(self, path_to_model, device, data_dir='../data',
                 guide_scale=2, sample_timesteps=None,
                 shallow_stride=2, deep_stride=3,
                 warmup_frac=0.10, cooldown_frac=0.05, warmup="frac"):

        super().__init__(path_to_model, device, data_dir, guide_scale, sample_timesteps)

        tstar = compute_tstar(self.noise_schedule) if warmup == "tstar" else None
        self.cached_forward = CachedTransformerForward(
            self.model,
            total_timesteps=self.T,
            shallow_stride=shallow_stride,
            deep_stride=deep_stride,
            warmup_frac=warmup_frac,
            cooldown_frac=cooldown_frac,
            tstar=tstar,
        )

        self.shallow_stride = shallow_stride
        self.deep_stride = deep_stride
        warmup_desc = (f"T*={tstar} (caching for t <= {tstar} of {self.T})"
                       if tstar is not None else f"warmup={warmup_frac}")
        print(f"[ManifoldCache] {len(self.model.blocks)} blocks, {self.T} steps, "
              f"shallow_stride={shallow_stride}, deep_stride={deep_stride}, "
              f"{warmup_desc}, cooldown={cooldown_frac}")

    def _forward_cached(self, noisy_target, dense_context_dict, step_idx, unconditioned=False):
        """
        Same as _forward() but routes through cached transformer.
        """
        if unconditioned:
            noisy_X = noisy_target['X_t'].float()
            noisy_E = noisy_target['E_t'].float()
            noisy_pos = noisy_target['pos_t'].float()
            bs, n, _, de = noisy_E.size()
            noisy_E = noisy_E.view(bs, n, n * self.max_bond_type)
            noisy_pos = noisy_pos.view(bs, n, n * self.max_position_type)
            node_mask_dict = {'target': noisy_target['node_mask'], 'all': noisy_target['node_mask']}
            relation_to_target = torch.zeros(noisy_X.size(0), noisy_X.size(1), 1).type_as(noisy_X)
        else:
            from utils.data_io import replace_first_indices, to_dense
            from torch_geometric.utils import to_dense_batch

            indicator_mol_to_ctx = dense_context_dict['indicator_mol_to_ctx']
            indicator_node_to_mol = dense_context_dict['indicator_node_to_mol']
            relation_to_target = dense_context_dict['relation_to_target']
            node_mask_context = dense_context_dict['node_mask_context']
            node_mask_target = dense_context_dict['node_mask_target']
            dense_data = dense_context_dict['dense_data']

            noisy_data = replace_first_indices(dense_data, node_mask_target, noisy_target)
            indicator_node_to_ctx = indicator_mol_to_ctx[indicator_node_to_mol]
            relation_to_target = relation_to_target[indicator_node_to_mol].view(-1, 1)

            noisy_X = noisy_data.X
            bs, n, _, de = noisy_data.E.size()
            noisy_E = noisy_data.E.view(bs, n, n * self.max_bond_type)
            noisy_pos = noisy_data.pos.view(bs, n, n * self.max_position_type)

            relation_to_target, _ = to_dense_batch(
                x=relation_to_target, batch=indicator_node_to_ctx,
                max_num_nodes=self.max_n_nodes
            )
            noisy_X, noisy_E, noisy_pos = noisy_X.float(), noisy_E.float(), noisy_pos.float()
            node_mask_dict = {'target': node_mask_target, 'all': node_mask_context}

        target_n_nodes = noisy_target['X_t'].size(1)
        t = noisy_target['t'] * self.T
        relation_to_target[~node_mask_dict['all']] = 1

        pred_X, pred_E, pred_y, pred_pos = self.cached_forward.forward(
            noisy_X, noisy_E, noisy_pos, relation_to_target,
            node_mask_dict, target_n_nodes, t=t,
            step_idx=step_idx, unconditioned=unconditioned,
        )
        pred = PlaceHolder(X=pred_X, E=pred_E, y=pred_y, pos=pred_pos)
        return pred

    def sample_p_zs_given_zt_cached(
        self, s, t, X_t, E_t, pos_t, node_mask, dense_context_dict,
        step_idx, deterministic=False
    ):
        """Same as sample_p_zs_given_zt but uses cached forward."""
        bs, n, _ = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        noisy_data = {
            "X_t": X_t, "E_t": E_t, "pos_t": pos_t,
            "t": t, "node_mask": node_mask,
        }

        def get_prob_cached(noisy_data, step_idx, unconditioned=False):
            pred = self._forward_cached(noisy_data, dense_context_dict, step_idx, unconditioned)
            pred_X = F.softmax(pred.X, dim=-1)
            pred_E = F.softmax(pred.E, dim=-1)
            pred_pos = F.softmax(pred.pos, dim=-1)

            n_nodes = pred.X.size(1)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device, n_nodes=n_nodes)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device, n_nodes=n_nodes)
            Qt = self.transition_model.get_Qt(beta_t, self.device, n_nodes=n_nodes)

            Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
            pred_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)

            unnormalized_probX_all = reverse_diffusion(
                predX_0=pred_all, X_t=Xt_all, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
            unnormalized_prob_X = unnormalized_probX_all[:, :, :self.max_node_type]
            unnormalized_prob_E = unnormalized_probX_all[:, :, self.max_node_type:].reshape(bs, n * n, -1)
            unnormalized_prob_pos = reverse_diffusion(
                predX_0=pred_pos.flatten(1, -2), X_t=pos_t.flatten(1, -2),
                Qt=Qt.pos, Qsb=Qsb.pos, Qtb=Qtb.pos
            ).reshape(bs, n, n, pred_pos.shape[-1])

            unnormalized_prob_X[unnormalized_prob_X.sum(-1) == 0] = 1e-5
            unnormalized_prob_E[unnormalized_prob_E.sum(-1) == 0] = 1e-5
            unnormalized_prob_pos[unnormalized_prob_pos.sum(-1) == 0] = 1e-5

            prob_X = unnormalized_prob_X / unnormalized_prob_X.sum(-1, keepdim=True)
            prob_E = (unnormalized_prob_E / unnormalized_prob_E.sum(-1, keepdim=True)).reshape(bs, n, n, pred_E.shape[-1])
            prob_pos = (unnormalized_prob_pos / unnormalized_prob_pos.sum(-1, keepdim=True)).reshape(bs, n, n, pred_pos.shape[-1])
            return prob_X, prob_E, prob_pos

        prob_X, prob_E, prob_pos = get_prob_cached(noisy_data, step_idx, unconditioned=False)

        if self.guide_scale != 1:
            uX, uE, uP = get_prob_cached(noisy_data, step_idx, unconditioned=True)
            prob_X = uX * (prob_X / uX.clamp_min(1e-10)) ** self.guide_scale
            prob_E = uE * (prob_E / uE.clamp_min(1e-10)) ** self.guide_scale
            prob_pos = uP * (prob_pos / uP.clamp_min(1e-10)) ** self.guide_scale
            prob_X = prob_X / prob_X.sum(-1, keepdim=True).clamp_min(1e-10)
            prob_E = prob_E / prob_E.sum(-1, keepdim=True).clamp_min(1e-10)
            prob_pos = prob_pos / prob_pos.sum(-1, keepdim=True).clamp_min(1e-10)

        assert ((prob_X.sum(-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(-1) - 1).abs() < 1e-4).all()
        assert ((prob_pos.sum(-1) - 1).abs() < 1e-4).all()

        sampled_s = sample_discrete_features(prob_X, prob_E, prob_pos, node_mask=node_mask, deterministic=deterministic)
        X_s = F.one_hot(sampled_s.X, num_classes=self.max_node_type).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.max_bond_type).float()
        pos_s = F.one_hot(sampled_s.pos, num_classes=self.max_position_type).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = PlaceHolder(X=X_s, E=E_s, y=None, pos=pos_s)
        return out_one_hot.mask(node_mask)

    @torch.no_grad()
    def generate(self, task_name=None, deterministic=False):
        """
        Cached version of generate(). Same API, same outputs.
        """
        dense_context_dict = self.dense_context_dict
        node_mask = self.target_node_mask
        n_nodes = self.target_n_nodes
        batch_size = self.batch_size

        node_mask = dense_context_dict['node_mask_target']
        sampled_s = sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        ).to(self.device)
        assert (sampled_s.E == torch.transpose(sampled_s.E, 1, 2)).all()

        # Reset caches for this generation
        self.cached_forward.reset()

        for s_int in tqdm(reversed(range(0, self.T)), desc='sampling (ManifoldCache)', total=self.T):
            s_array = s_int * torch.ones((batch_size, 1)).to(self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            sampled_s = self.sample_p_zs_given_zt_cached(
                s_norm, t_norm, sampled_s.X, sampled_s.E, sampled_s.pos,
                node_mask, dense_context_dict,
                step_idx=s_int,
                deterministic=deterministic,
            )

        # Print caching stats
        self.cached_forward.print_stats()

        # Convert to molecules (same as original)
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = sampled_s.X[i, :n].cpu()
            edge_types = sampled_s.E[i, :n, :n].cpu()
            positions = sampled_s.pos[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types, positions])

        _, valid_unique_list, all_smiles_list = convert_graph_to_smiles(
            molecule_list, self.tokenizer, self.excluded_smiles
        )

        # Build context_target_map (same as original)
        context_smiles = {'pos': [], 'med': [], 'neg': []}
        pos_df, med_df, neg_df = self.pos_df, self.med_df, self.neg_df
        for b in range(batch_size):
            pos_smiles = [(pos_df.iloc[idx]['smiles'], pos_df.iloc[idx]['score'])
                         for idx in dense_context_dict['context_indices']['pos'][b]]
            med_smiles = [(med_df.iloc[idx]['smiles'], med_df.iloc[idx]['score'])
                         for idx in dense_context_dict['context_indices']['med'][b]]
            neg_smiles = [(neg_df.iloc[idx]['smiles'], neg_df.iloc[idx]['score'])
                         for idx in dense_context_dict['context_indices']['neg'][b]]
            context_smiles['pos'].append(pos_smiles)
            context_smiles['med'].append(med_smiles)
            context_smiles['neg'].append(neg_smiles)

        context_target_map = {}
        for i, smiles in enumerate(valid_unique_list):
            key = smiles if smiles is not None else f"None_{i}"
            context_target_map[key] = {
                'pos': context_smiles['pos'][i],
                'med': context_smiles['med'][i],
                'neg': context_smiles['neg'][i]
            }

        return context_target_map
