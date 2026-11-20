"""Question-Aware Hierarchical Multimodal Graph Reasoning for Document QA (reference implementation)."""
from .graph import HMDKG, EdgeKind, Node, NodeType
from .pipeline import HMDKGPipeline, PipelineConfig

__version__ = "0.1.0"
__all__ = ["HMDKG", "EdgeKind", "Node", "NodeType", "HMDKGPipeline", "PipelineConfig"]
