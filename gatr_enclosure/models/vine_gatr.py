# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import gatr
import torch
import torch_geometric as pyg
import wandb
from gatr.interface import (
    embed_oriented_plane,
    embed_point,
    embed_scalar,
    extract_point,
)
from gatr.layers import EquiLinear, GeometricBilinear
from lab_gatr.models.lab_gatr import CrossAttentionHatchling, interp
from lab_gatr.nn.mlp.geometric_algebra import (
    MLP,
    EquiLayerNorm,
    ScalarGatedNonlinearity,
)
from torch.utils.checkpoint import checkpoint
from torch_cluster import knn

from gatr_enclosure.nn import LearnedVirtualNodes
from gatr_enclosure.transforms import PointCloudSingularValueDecomposition

from .utils import (
    GATrSequential,
    ProjectiveGeometricAlgebraInterface,
    Stopwatch,
    construct_join_reference,
    get_attention_mask,
    get_decoder_query,
)


class ViNEGATr(torch.nn.Module):

    def __init__(
        self,
        pga_interface: ProjectiveGeometricAlgebraInterface,
        num_virtual_nodes: int,
        hidden_mv_channels: int,
        num_heads: int,
        num_blocks: int,
        use_grade_mixer_pga_interface: bool = False,
        virtual_s_channels: int = 1,
        virtual_nodes_use_orientation: bool = False,
        virtual_nodes_init_distribution: Literal[
            "normal", "uniform", "truncated_normal"
        ] = "uniform",
        virtual_nodes_init_distribution_std: float = 1.0,
        encoder_num_layers: int = 1,
        encoder_only: bool = False,
        decoder_use_checkpointing: bool = False,
        decoder_id_module: Literal[
            "cross_attention", "interpolation"
        ] = "cross_attention",
        decoder_id_query_idcs: Optional[str] = None,
        dropout_prob: Optional[float] = None,
        broken: bool = False,
        tweak_reference: bool = False,
        online_pca: bool = False,
        **kwargs: Any,
    ):
        """Initialises ViNE-GATr model.

        Args:
            pga_interface (ProjectiveGeometricAlgebraInterface): Object bundling numbers of channels
                and interface logic.
            num_virtual_nodes (int): Number of virtual nodes.
            hidden_mv_channels (int): Number of hidden multivector channels.
            num_heads (int): Number of attention heads.
            num_blocks (int): Number of transformer blocks in the GATr backend.
            use_grade_mixer_pga_interface (bool): Whether to use a grade mixer also in the interface
                pathway. Default: False
            virtual_s_channels (int): Number of channels of the virtual scalars. Default: 1
            virtual_nodes_use_orientation (bool): Whether virtual nodes use surface normal and
                geodesic distance. Requires according preprocessing of input data. Default: False
            virtual_nodes_init_distribution (str): Distribution for initialising virtual node
                coordinates ("normal", "uniform" or "truncated_normal"). Default: "uniform"
            virtual_nodes_init_distribution_std (float): Standard deviation of distribution for
                initialising virtual node coordinates. Default: 1.0
            encoder_num_layers (int): Number of cross-attention layers in the encoder. Default: 1
            encoder_only (bool): Whether to skip GATr backend and cross-attention decoder
                altogether. Useful for debugging virtual node positions. Default: False
            decoder_use_checkpointing (bool): Whether to use activation checkpointing in the
                cross-attention decoder. Default: False
            decoder_id_module (str): Identifier of the decoder module ("cross_attention" or
                "interpolation"). Default: "cross_attention"
            decoder_id_query_idcs (str, optional): Identifier of indices for selecting queries in
                the cross-attention decoder, e.g., "dirichlet_boundary" where
                "dirichlet_boundary_index" is present in the input data.
            dropout_prob (float, optional): Dropout probability (0 to 1).
            **kwargs (optional): Additional keyword arguments which are ignored. Workaround for ease
                of experiment implementation.
        """
        super().__init__()

        self.pga_interface = pga_interface

        self.broken = broken
        self.tweak_reference = tweak_reference

        # Virtual nodes
        self.virtual_nodes_use_orientation = virtual_nodes_use_orientation
        num_dim = 3 + self.virtual_nodes_use_orientation * 4

        self.learned_virtual_nodes = LearnedVirtualNodes(
            num_virtual_nodes,
            num_dim,
            init_distribution=virtual_nodes_init_distribution,
            init_distribution_std=virtual_nodes_init_distribution_std,
            broken=broken,
        )
        setattr(
            self.learned_virtual_nodes.linear_combination.weight, "var_lr_flag", True
        )

        self.virtual_s = torch.nn.Parameter(
            torch.empty(
                self.learned_virtual_nodes.linear_combination.weight.size(0),
                virtual_s_channels,
            )
        )
        self.learned_virtual_nodes._init_fun(self.virtual_s)
        setattr(self.virtual_s, "var_lr_flag", True)

        virtual_s_channels += num_dim
        self.virtual_s_layer_norm = torch.nn.LayerNorm(virtual_s_channels)
        # setattr(self.virtual_s_lyer_norm, "var_lr_flag", True)

        # Grade mixing
        virtual_nodes_mv_channels = (
            num_dim * (1 + self.virtual_nodes_use_orientation * 2) + 1
        )
        if self.broken:
            virtual_nodes_mv_channels *= self.learned_virtual_nodes._num_frames

        self.grade_mixer_virtual_nodes = self._get_grade_mixer(
            virtual_nodes_mv_channels, hidden_mv_channels, virtual_s_channels
        )
        # for param in self.grade_mixer_virtual_nodes.parameters():
        #     setattr(param, "var_lr_flag", True)
        self.grade_mixer_virtual_nodes_skip_linear = EquiLinear(
            in_mv_channels=hidden_mv_channels + virtual_nodes_mv_channels,
            out_mv_channels=hidden_mv_channels,
            in_s_channels=2 * virtual_s_channels,
            out_s_channels=virtual_s_channels,
        )
        # for param in self.grade_mixer_virtual_nodes_skip_linear.parameters():
        #     setattr(param, "var_lr_flag", True)

        self.use_grade_mixer_pga_interface = use_grade_mixer_pga_interface
        if self.use_grade_mixer_pga_interface is True:
            self.grade_mixer_pga_interface = self._get_grade_mixer(
                pga_interface.in_mv_channels,
                hidden_mv_channels,
                pga_interface.in_s_channels,
            )
            # for param in self.grade_mixer_pga_interface.parameters():
            #     setattr(param, "var_lr_flag", True)

        # Encoding
        in_mv_channels = (
            hidden_mv_channels
            if self.use_grade_mixer_pga_interface is True
            else pga_interface.in_mv_channels
        )
        encoder_out_mv_channels = (
            1 if decoder_id_module == "interpolation" else hidden_mv_channels
        )
        self.encoder_layers = torch.nn.ModuleList()
        for _ in range(encoder_num_layers):
            self.encoder_layers.append(
                CrossAttentionHatchling(
                    num_input_channels_source=in_mv_channels,
                    num_input_channels_target=hidden_mv_channels,
                    num_output_channels=encoder_out_mv_channels,
                    num_input_scalars_source=pga_interface.in_s_channels,
                    num_input_scalars_target=virtual_s_channels,
                    num_output_scalars=virtual_s_channels,
                    num_attn_heads=num_heads,
                    num_latent_channels=hidden_mv_channels,
                    dropout_probability=dropout_prob,
                )
            )
        # for param in self.encoder_layers.parameters():
        #     setattr(param, "var_lr_flag", True)

        # self._register_hook_crossattention_layer(self.encoder_layers[-1], 'encoder')

        names_modules = ["setup", "grade_mixing", "encoding"]

        self.encoder_only = encoder_only
        if self.encoder_only is False:

            # Backend
            self.backend = gatr.GATr(
                in_mv_channels=encoder_out_mv_channels,
                out_mv_channels=hidden_mv_channels,
                hidden_mv_channels=hidden_mv_channels,
                in_s_channels=virtual_s_channels,
                out_s_channels=virtual_s_channels,
                hidden_s_channels=4 * hidden_mv_channels,
                attention=gatr.SelfAttentionConfig(num_heads=num_heads),
                mlp=gatr.MLPConfig(),
                num_blocks=num_blocks,
                dropout_prob=dropout_prob,
            )

            # Decoding
            self.decoder_id_module = decoder_id_module
            match self.decoder_id_module:

                case "cross_attention":
                    self.decoder = CrossAttentionHatchling(
                        num_input_channels_source=hidden_mv_channels,
                        num_input_channels_target=in_mv_channels,
                        num_output_channels=pga_interface.out_mv_channels,
                        num_input_scalars_source=virtual_s_channels,
                        num_input_scalars_target=pga_interface.in_s_channels,
                        num_output_scalars=1,  # soothe the 🐊
                        num_attn_heads=num_heads,
                        num_latent_channels=hidden_mv_channels,
                        dropout_probability=dropout_prob,
                    )
                    self.decoder_use_checkpointing = decoder_use_checkpointing
                    self.decoder_id_query_idcs = decoder_id_query_idcs

                case "interpolation":
                    self.decoder_mlp = MLP(
                        (
                            hidden_mv_channels + in_mv_channels,
                            hidden_mv_channels,
                            hidden_mv_channels,
                            pga_interface.out_mv_channels,
                        ),
                        num_input_scalars=virtual_s_channels
                        + pga_interface.in_s_channels,
                        num_output_scalars=pga_interface.in_s_channels,
                        use_norm_in_first=False,
                        dropout_probability=dropout_prob,
                    )

            # for param in self.decoder.parameters():
            #     setattr(param, "var_lr_flag", True)

            names_modules.extend(("backend", "decoding"))

        if online_pca:
            self.pca = PointCloudSingularValueDecomposition()
            names_modules += ["preprocessing"]
        else:
            self.pca = None

        self.num_param = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        print(f"ViNE-GATr ({self.num_param} parameters)")

        self.stopwatch = Stopwatch(names_splits=names_modules)

        self._log_cache: Any = None

    @staticmethod
    def _get_grade_mixer(
        in_mv_channels: int, out_mv_channels: int, s_channels: int
    ) -> torch.nn.Module:

        kwargs = {"in_s_channels": s_channels, "out_s_channels": s_channels}
        grade_mixer = GATrSequential(
            GeometricBilinear(in_mv_channels, out_mv_channels, **kwargs),
            EquiLayerNorm(),
            ScalarGatedNonlinearity("gelu"),
            GeometricBilinear(out_mv_channels, out_mv_channels, **kwargs),
            EquiLayerNorm(),
            ScalarGatedNonlinearity("gelu"),
        )

        return grade_mixer

    def forward(
        self, data: pyg.data.Data, batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if self.training is False:
            self.stopwatch.reset()

        if self.pca is not None:
            data = self.pca(data)
            if self.training is False:
                self.stopwatch.time_split("preprocessing")

        mv_source, s_source = self.pga_interface.embed(data)
        batch_source = (
            None if batch is not None and batch.unique().numel() == 1 else batch
        )

        # Virtual nodes
        num_pos = self._get_num_pos(data.pos, batch)
        virtual_nodes_bases = self.learned_virtual_nodes.get_bases(
            data.singular_values, data.right_singular_vectors, num_pos
        )
        virtual_nodes_coord = self.learned_virtual_nodes.linear_combination.weight

        mv_target, s_target, batch_target, frame_id = self._embed_virtual_nodes(
            virtual_nodes_bases, virtual_nodes_coord, data.origin, return_frame_id=True
        )

        batch_size, num_frames, _, _ = virtual_nodes_bases.shape
        s_target = torch.cat(
            (
                s_target,
                self.virtual_s.tile(
                    batch_size * num_frames if not self.broken else batch_size, 1
                ),
            ),
            dim=1,
        )

        s_target = self.virtual_s_layer_norm(s_target)

        join_reference_source, join_reference_target = self._get_join_references(
            mv_source, mv_target, batch_source, batch_target
        )

        if self.tweak_reference:
            join_reference_source = join_reference_source * 2.0
            join_reference_target = join_reference_target * 2.0

        if self.training is False:
            self.stopwatch.time_split("setup")

        # Grade mixing
        if self.use_grade_mixer_pga_interface is True:
            mv_source, s_source = self._call_tensor_product(
                self.grade_mixer_pga_interface,
                mv_source,
                s_source,
                join_reference_source,
            )

        mv, s = self._call_tensor_product(
            self.grade_mixer_virtual_nodes, mv_target, s_target, join_reference_target
        )
        mv, s = self.grade_mixer_virtual_nodes_skip_linear(
            torch.cat((mv, mv_target), dim=1), torch.cat((s, s_target), dim=1)
        )

        if self.training is False:
            self.stopwatch.time_split("grade_mixing")

        # Encoding
        for layer in self.encoder_layers:
            mv, s = self._call_cross_attention(
                layer,
                mv_source,
                mv,
                s_source,
                s,
                batch_source,
                batch_target,
                join_reference_target,
            )

        self._log_cache = (mv, s, batch_size, num_frames)
        data.virtual_nodes_pos, data.virtual_nodes_batch = self._extract_pos(
            mv, batch_target
        )
        data.cross_attention_pos = self.get_cross_attention_pos(
            batch_source, batch_target
        )
        data.frame_id = (
            frame_id.reshape(mv.shape[0], 1)
            .expand(mv.shape[0], mv.shape[1])
            .reshape(-1)
        )

        if self.training is False:
            self.stopwatch.time_split("encoding")

        if self.encoder_only is False:

            # Backend
            mv, s = self._call_self_attention(
                self.backend, mv, s, batch_target, join_reference_target
            )

            if self.training is False:
                self.stopwatch.time_split("backend")

            # Decoding
            match self.decoder_id_module:

                case "cross_attention":
                    if self.decoder_id_query_idcs is not None:
                        mv_source, s_source, batch_source, join_reference_source = (
                            get_decoder_query(
                                data[f"{self.decoder_id_query_idcs}_index"],
                                mv_source,
                                s_source,
                                batch_source,
                                join_reference_source,
                            )
                        )

                    mv, s = self._call_cross_attention(
                        self.decoder,
                        mv,
                        mv_source,
                        s,
                        s_source,
                        batch_target,
                        batch_source,
                        join_reference_source,
                        use_checkpointing=self.decoder_use_checkpointing,
                    )

                case "interpolation":
                    mv, s = self._call_interpolation(
                        data.virtual_nodes_pos,
                        data.pos,
                        self.decoder_mlp,
                        mv,
                        mv_source,
                        s,
                        s_source,
                        (
                            data.virtual_nodes_batch
                            if hasattr(data, "virtual_nodes_batch")
                            else None
                        ),
                        batch_source,
                        join_reference_source,
                    )

            if self.training is False:
                self.stopwatch.time_split("decoding")

        return self.pga_interface.extract(mv, s)

    @staticmethod
    def _get_num_pos(
        pos: torch.Tensor, batch: Union[None, torch.Tensor]
    ) -> torch.Tensor:

        if batch is None:
            num_pos = torch.tensor(pos.size(0), device=pos.device)

        else:
            num_pos = torch.bincount(batch)

        return num_pos

    @staticmethod
    def _get_join_references(
        mv_source: torch.Tensor,
        mv_target: torch.Tensor,
        batch_source: Union[None, torch.Tensor],
        batch_target: Union[None, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        join_reference = construct_join_reference(
            mv_source, batch_source, expand_batch=False
        )

        if batch_source is None:
            batch_source = torch.zeros(
                mv_source.size(0), dtype=torch.int, device=mv_source.device
            )

        if batch_target is None:
            batch_target = torch.zeros(
                mv_target.size(0), dtype=torch.int, device=mv_source.device
            )

        return (join_reference[batch_source], join_reference[batch_target])

    @overload
    def _embed_virtual_nodes(
        self,
        virtual_nodes_bases: torch.Tensor,
        virtual_nodes_coord: torch.Tensor,
        origin: torch.Tensor,
        return_frame_id: Literal[False],
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor]]: ...

    @overload
    def _embed_virtual_nodes(
        self,
        virtual_nodes_bases: torch.Tensor,
        virtual_nodes_coord: torch.Tensor,
        origin: torch.Tensor,
        return_frame_id: Literal[True],
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor], torch.Tensor]: ...

    def _embed_virtual_nodes(
        self,
        virtual_nodes_bases: torch.Tensor,
        virtual_nodes_coord: torch.Tensor,
        origin: torch.Tensor,
        return_frame_id: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor], torch.Tensor],
    ]:

        batch_size, num_frames, _, _ = virtual_nodes_bases.shape
        num_virtual_nodes_coord, num_dim = virtual_nodes_coord.shape

        # Position(-orientation) space
        virtual_nodes_pos_bases = virtual_nodes_bases[:, :, :, :3]
        mv = embed_oriented_plane(
            virtual_nodes_pos_bases, torch.zeros_like(virtual_nodes_pos_bases)
        )

        num_channels = num_dim
        assert mv.shape == (batch_size, num_frames, num_channels, 16)

        if self.virtual_nodes_use_orientation is True:

            # Ambient
            virtual_nodes_orientation_bases = virtual_nodes_bases[:, :, :, 3:6]
            mv_ = embed_oriented_plane(
                virtual_nodes_orientation_bases,
                torch.zeros_like(virtual_nodes_orientation_bases),
            )
            mv = torch.cat((mv, mv_), dim=2)

            num_channels += num_dim

            # Intrinsic
            mv_ = embed_scalar(virtual_nodes_bases[:, :, :, 6:7])
            mv = torch.cat((mv, mv_), dim=2)

            num_channels += num_dim

        # Origin ("center of mass")
        mv_ = (
            embed_point(origin[:, :3])
            .view(batch_size, 1, 1, 16)
            .expand(batch_size, num_frames, 1, 16)
        )
        mv = torch.cat((mv, mv_), dim=2)

        num_channels += 1
        assert mv.shape == (batch_size, num_frames, num_channels, 16)

        if self.broken:
            # PyG-style batching
            mv = (
                mv.view(batch_size, 1, num_frames, num_channels, 16)
                .expand(
                    batch_size, num_virtual_nodes_coord, num_frames, num_channels, 16
                )
                .reshape(
                    batch_size * num_virtual_nodes_coord, num_frames * num_channels, 16
                )
            )
            s = virtual_nodes_coord.tile(batch_size, 1)

            assert s.shape[0] == mv.shape[0], (
                s.shape,
                mv.shape,
                batch_size,
                num_virtual_nodes_coord,
                num_dim,
            )
            num_virtual_nodes = num_virtual_nodes_coord
        else:
            # PyG-style batching
            mv = mv.repeat_interleave(num_virtual_nodes_coord, dim=1).view(
                # batch_size * num_virtual_nodes_coord * num_frames, -1, 16
                batch_size * num_virtual_nodes_coord * num_frames,
                num_channels,
                16,
            )
            s = virtual_nodes_coord.tile(batch_size * num_frames, 1)
            num_virtual_nodes = num_virtual_nodes_coord * num_frames

        assert s.shape == (batch_size * num_virtual_nodes, num_dim), (
            s.shape,
            batch_size,
            num_virtual_nodes,
            virtual_nodes_coord.shape,
        )

        if batch_size > 1:
            batch = torch.arange(
                batch_size, device=virtual_nodes_bases.device
            ).repeat_interleave(num_virtual_nodes)
        else:
            batch = None

        if return_frame_id:
            if self.broken:
                frame_id = torch.zeros(
                    batch_size * num_virtual_nodes, 1, 1, dtype=torch.long
                )
            else:
                frame_id = (
                    torch.arange(num_frames)
                    .view(1, num_frames, 1, 1)
                    .expand(batch_size, num_frames, 1, 1)
                    # .reshape(batch_size * num_frames, 1, 1)
                    .repeat_interleave(num_virtual_nodes_coord, dim=1)
                    .reshape(batch_size * num_virtual_nodes_coord * num_frames, 1, 1)
                )
            return mv, s, batch, frame_id
        else:
            return mv, s, batch

    @staticmethod
    def _call_tensor_product(
        module: torch.nn.Module,
        mv: torch.Tensor,
        s: torch.Tensor,
        join_reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mv, s = module(mv, s, join_reference)

        return mv, s

    @staticmethod
    def _call_cross_attention(
        module: torch.nn.Module,
        mv_source: torch.Tensor,
        mv_target: torch.Tensor,
        s_source: torch.Tensor,
        s_target: torch.Tensor,
        batch_source: Union[None, torch.Tensor],
        batch_target: Union[None, torch.Tensor],
        join_reference_target: torch.Tensor,
        use_checkpointing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        attention_mask = get_attention_mask(batch_target, batch_source)

        if use_checkpointing is True:
            mv, s = checkpoint(
                module,
                mv_source,
                mv_target,
                s_source,
                s_target,
                attention_mask,
                join_reference_target,
                use_reentrant=False,
            )

        else:
            mv, s = module(
                mv_source,
                mv_target,
                s_source,
                s_target,
                attention_mask,
                join_reference_target,
            )

        return mv, s

    @staticmethod
    def _call_self_attention(
        module: torch.nn.Module,
        mv: torch.Tensor,
        s: torch.Tensor,
        batch: Union[None, torch.Tensor],
        join_reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        attention_mask = get_attention_mask(batch)
        mv, s = module(mv, s, attention_mask, join_reference)

        return mv, s

    @staticmethod
    def _extract_pos(
        mv: torch.Tensor, batch: Union[None, torch.Tensor]
    ) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        num_tokens, num_channels, _ = mv.shape

        pos = extract_point(mv).view(num_tokens * num_channels, 3)

        if batch is not None:
            batch = batch.repeat_interleave(num_channels)

        return pos, batch

    @staticmethod
    def _call_interpolation(
        pos_source: torch.Tensor,
        pos_target: torch.Tensor,
        mlp: torch.nn.Module,
        mv_source: torch.Tensor,
        mv_target: torch.Tensor,
        s_source: torch.Tensor,
        s_target: torch.Tensor,
        batch_source: Union[None, torch.Tensor],
        batch_target: Union[None, torch.Tensor],
        join_reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        idcs_target, idcs_source = knn(
            pos_source, pos_target, 3, batch_source, batch_target
        )
        dummy_data = pyg.data.Data(
            scale0_interp_source=idcs_source, scale0_interp_target=idcs_target
        )

        mv, s = interp(
            mlp,
            multivectors=mv_source,
            multivectors_skip=mv_target,
            scalars=s_source,
            scalars_skip=s_target,
            pos_source=pos_source,
            pos_target=pos_target,
            data=dummy_data,
            scale_id=0,
            reference_multivector=join_reference,
        )

        return mv, s

    def get_log_dict(self) -> Dict[str, Any]:

        # Norm of virtual node embedding (ViNE) vectors
        vine = torch.cat(
            (self.learned_virtual_nodes.linear_combination.weight, self.virtual_s),
            dim=1,
        )
        norm_vine = wandb.Histogram(vine.norm(dim=1).tolist())

        # Frame (standard) deviation
        mv, s, batch_size, num_frames = self._log_cache
        frame_std_mv, frame_std_s = self._compute_frame_std(
            mv, s, batch_size, num_frames
        )

        return {
            "norm_vine": norm_vine,
            "frame_std_mv": frame_std_mv,
            "frame_std_s": frame_std_s,
        }

    def _compute_frame_std(
        self, mv: torch.Tensor, s: torch.Tensor, batch_size: int, num_frames: int
    ) -> Tuple[float, float]:

        num_virtual_nodes_coord = int(mv.size(0) / batch_size / num_frames)
        num_channels_mv, num_channels_s = mv.size(1), s.size(1)

        mv = self._unflatten(mv, (batch_size, num_frames))
        assert mv.shape == (
            batch_size,
            num_frames,
            num_virtual_nodes_coord,
            num_channels_mv,
            16,
        )

        s = self._unflatten(s, (batch_size, num_frames))
        assert s.shape == (
            batch_size,
            num_frames,
            num_virtual_nodes_coord,
            num_channels_s,
        )

        frame_std_mv = mv.std(dim=1).quantile(0.94, dim=-1).mean().item()
        frame_std_s = s.std(dim=1).mean().item()

        return frame_std_mv, frame_std_s

    @staticmethod
    def _unflatten(tensor: torch.Tensor, num_dim: Iterable[int]) -> torch.Tensor:
        return tensor.view(*num_dim, -1, *tensor.shape[1:])

    def _register_hook_crossattention_layer(
        self, layer: CrossAttentionHatchling, name: str
    ) -> None:
        if (
            not hasattr(self, "_registered_crossattention_pos")
            or self._registered_crossattention_pos is None
        ):
            self._registered_crossattention_pos: Dict[str, Sequence[Any]] = dict()

        def build_hook(storage: Dict[str, Sequence[Any]], name: str) -> Callable:
            def hook(
                model: torch.nn.Module, _input: Sequence[Any], _output: Sequence[Any]
            ) -> None:
                # q_mv, k_mv, _, q_s, k_s, _ = _input[:6]

                # _input = (q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask)
                storage[name] = _input
                # storage[name] = _input[:2]

            return hook

        layer.block.attention.attention.register_forward_hook(
            build_hook(self._registered_crossattention_pos, name)
        )

    def get_cross_attention_pos(
        self,
        batch_source: Union[None, torch.Tensor],
        batch_target: Union[None, torch.Tensor],
    ) -> Dict[str, Any]:

        if (
            not hasattr(self, "_registered_crossattention_pos")
            or self._registered_crossattention_pos is None
        ):
            return dict()

        cross_attention_pos = dict()

        # for key, (q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask) in self._registered_crossattention_pos.items():
        # for key, (q_mv, k_mv, v_mv, q_s, k_s, v_s) in self._registered_crossattention_pos.items():
        for (
            key,
            cached_cross_attention_input,
        ) in self._registered_crossattention_pos.items():
            q_mv, k_mv = cached_cross_attention_input[:2]
            # k_mv has shape     1 x   input_tokens x hidden_channels x 16
            # q_mv has shape heads x virtual_tokens x hidden_channels x 16
            # hidden_channels = 4
            # self.encoder_layers[-1].block.attention.attention.log_weights.shape = (heads, 1, hidden_channels)

            hidden_channels = self.encoder_layers[
                -1
            ].block.attention.attention.log_weights.shape[2]
            heads = self.encoder_layers[-1].block.attention.attention.log_weights.shape[
                0
            ]
            _pos_source, _batch_source = self._extract_pos(
                k_mv.reshape(-1, hidden_channels, 16), batch_source
            )
            _pos_target, _batch_target = self._extract_pos(
                q_mv.reshape(-1, hidden_channels, 16),
                (
                    batch_target.reshape(1, -1).expand(heads, -1).reshape(-1)
                    if batch_target is not None
                    else None
                ),
            )
            pos_data = (_pos_source, _pos_target, _batch_source, _batch_target)

            cross_attention_pos[key] = (
                pos_data,
                cached_cross_attention_input,
                batch_target,
            )

        self._registered_crossattention_pos.clear()
        return cross_attention_pos


from gatr_enclosure.nn.loss import AttractiveLoss, RepulsiveLoss, SpectralLoss


class ViNEGATrWithRegularization(ViNEGATr):

    # def __init__(self, *args, attractive_k=8, repulsive_radius=1.5, **kwargs):
    # super().__init__(*args, **kwargs)
    # self.attractive = AttractiveLoss(num_nearest_pos=attractive_k)
    # self.repulsive = RepulsiveLoss(radius_repulsion=repulsive_radius)
    def __init__(
        self,
        *args: Any,
        attractive_radius: float = 1.5,
        repulsive_radius: float = 1.5,
        spectral_t: float = 2.0,
        spectral_n_neigh: int = 32,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        dense = False
        self.attractive = AttractiveLoss(
            radius_attraction=attractive_radius, dense=dense
        )
        self.repulsive = RepulsiveLoss(radius_repulsion=repulsive_radius, dense=dense)
        self.spectral = SpectralLoss(
            num_nearest_pos=32, diffusion_t=spectral_t, n_eig=spectral_n_neigh
        )

        self._register_hook_crossattention_layer(self.encoder_layers[-1], "encoder")

    def regularization_fn(
        self, data: pyg.data.Data, batch: Union[None, torch.Tensor]
    ) -> torch.Tensor:

        # pos_source = data.pos
        # pos_target = data.virtual_nodes_pos
        # batch_source = batch
        # batch_target = data.virtual_nodes_batch

        losses = {}

        # pos_source, pos_target, batch_source, batch_target = data.cross_attention_pos['encoder'][0]
        # losses['attractive'] = 1000 * self.attractive(pos_source, pos_target, batch_source, batch_target)
        # losses['repulsive'] = 1000 * self.repulsive(pos_source, batch_source)

        batch_target_virtualnodes = data.cross_attention_pos["encoder"][2]

        # if hasattr(data, 'spectral_subset_index'):
        #     spectral_subset_index = data.spectral_subset_index
        # else:
        #     spectral_subset_index = None
        spectral_subset_index = None

        loss_repulsive, loss_attractive = self.spectral(
            data,
            layer=self.encoder_layers[-1].block.attention.attention,
            batch_source=batch,
            batch_target=batch_target_virtualnodes,
            subset_index=spectral_subset_index,
        )

        losses["spectral"] = loss_repulsive - loss_attractive

        losses["spectral_rep"] = loss_repulsive
        losses["spectral_attr"] = loss_attractive

        return losses
